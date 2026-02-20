"""ColabFold MSA search backend (CPU-only).

Pre-fetches unpaired + paired MSAs via the ColabFold API and caches them
on CACHE_VOLUME so they can be reused across folding backends.

Cache layout:
    /cache/colabsearch/unpaired/{seq_hash}/
        uniref.a3m
        bfd.mgnify30.metaeuk30.smag30.a3m
        merged.a3m              # stripped lowercase, for backends that want one file
        _COMPLETE

    /cache/colabsearch/paired/{combined_hash}/    # only for 2+ protein chains
        pair.a3m
        _COMPLETE
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from modal import Image

from .common import CACHE_VOLUME, app

colabsearch_image = Image.debian_slim(python_version="3.12").pip_install("requests")

HOST_URL = "https://api.colabfold.com"


def _submit_and_download(query: str, mode: str) -> bytes:
    """Submit job to ColabFold API, poll until complete, download tar.gz."""
    import random
    import time

    import requests

    headers = {"User-Agent": "foldism/colabsearch"}
    endpoint = "pair" if "pair" in mode else "msa"

    def _post():
        while True:
            try:
                res = requests.post(
                    f"{HOST_URL}/ticket/{endpoint}",
                    data={"q": query, "mode": mode},
                    timeout=6.02,
                    headers=headers,
                )
                return res.json()
            except requests.exceptions.Timeout:
                print("Timeout submitting, retrying...")
                continue
            except Exception as e:
                print(f"Error submitting: {e}, retrying in 5s...")
                time.sleep(5)
                continue

    def _poll(ticket_id):
        while True:
            try:
                res = requests.get(
                    f"{HOST_URL}/ticket/{ticket_id}",
                    timeout=6.02,
                    headers=headers,
                )
                return res.json()
            except requests.exceptions.Timeout:
                print("Timeout polling, retrying...")
                continue
            except Exception as e:
                print(f"Error polling: {e}, retrying in 5s...")
                time.sleep(5)
                continue

    def _download(ticket_id):
        while True:
            try:
                res = requests.get(
                    f"{HOST_URL}/result/download/{ticket_id}",
                    timeout=6.02,
                    headers=headers,
                )
                return res.content
            except requests.exceptions.Timeout:
                print("Timeout downloading, retrying...")
                continue
            except Exception as e:
                print(f"Error downloading: {e}, retrying in 5s...")
                time.sleep(5)
                continue

    out = _post()
    while out.get("status") in ("UNKNOWN", "RATELIMIT"):
        t = 5 + random.randint(0, 5)
        print(f"Status: {out['status']}, sleeping {t}s...")
        time.sleep(t)
        out = _post()

    if out.get("status") == "ERROR":
        raise RuntimeError(f"ColabFold API error: {out}")
    if out.get("status") == "MAINTENANCE":
        raise RuntimeError("ColabFold API is under maintenance")

    ticket_id = out["id"]
    print(f"Submitted {endpoint} job {ticket_id}, polling...")

    while out.get("status") in ("UNKNOWN", "RUNNING", "PENDING"):
        t = 5 + random.randint(0, 5)
        time.sleep(t)
        out = _poll(ticket_id)
        print(f"Status: {out.get('status')}")

    if out.get("status") != "COMPLETE":
        raise RuntimeError(f"ColabFold API failed: {out}")

    return _download(ticket_id)


def _extract_a3m_for_id(a3m_content: str, target_id: int) -> str:
    """Extract A3M lines belonging to a specific sequence ID."""
    lines: list[str] = []
    current_id = None
    for line in a3m_content.splitlines(keepends=True):
        line = line.replace("\x00", "")
        if line.startswith(">"):
            try:
                current_id = int(line[1:].strip().split()[0])
            except ValueError:
                pass
        if current_id == target_id:
            lines.append(line)
    return "".join(lines)


def _skip_first_entry(a3m_content: str) -> str:
    """Skip the first header+sequence entry in an A3M string."""
    lines = a3m_content.splitlines(keepends=True)
    # Find the second header line; everything from there onward is hits
    header_count = 0
    for i, line in enumerate(lines):
        if line.startswith(">"):
            header_count += 1
            if header_count == 2:
                return "".join(lines[i:])
    return ""  # only had the query


def _strip_lowercase(a3m_content: str) -> str:
    """Strip lowercase insertions from A3M sequences."""
    lines: list[str] = []
    for line in a3m_content.splitlines():
        if line.startswith("#") or line.startswith(">"):
            lines.append(line)
        else:
            lines.append("".join(c for c in line if not c.islower()))
    return "\n".join(lines) + "\n"


def _fetch_unpaired(sequence: str, use_env: bool, cache_dir: Path) -> str:
    """Fetch unpaired MSA for one sequence. Saves raw + merged A3M files.

    Returns path to cache directory on volume.
    """
    import tarfile
    from io import BytesIO

    seq_hash = sha256(f"{sequence}|use_env={use_env}".encode()).hexdigest()[:16]
    cache_path = cache_dir / seq_hash
    marker = cache_path / "_COMPLETE"

    if marker.exists():
        print(f"[colabsearch] Unpaired cache HIT: {seq_hash} ({len(sequence)} aa)")
        return str(cache_path)

    print(f"[colabsearch] Unpaired cache MISS: {seq_hash} ({len(sequence)} aa)")

    query = f">101\n{sequence}\n"
    mode = "env" if use_env else "all"
    tar_bytes = _submit_and_download(query, mode)

    a3m_files = ["uniref.a3m"]
    if use_env:
        a3m_files.append("bfd.mgnify30.metaeuk30.smag30.a3m")

    cache_path.mkdir(parents=True, exist_ok=True)

    all_filtered: list[str] = []
    is_first_source = True
    with tarfile.open(fileobj=BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.name in a3m_files:
                f = tar.extractfile(member)
                if f:
                    content = f.read().decode()
                    (cache_path / member.name).write_text(content)
                    filtered = _extract_a3m_for_id(content, 101)
                    if is_first_source:
                        all_filtered.append(filtered)
                        is_first_source = False
                    else:
                        # Skip query (first entry) from subsequent sources
                        all_filtered.append(_skip_first_entry(filtered))

    if not all_filtered:
        raise RuntimeError(f"Empty MSA result for sequence ({len(sequence)} aa)")

    merged = _strip_lowercase("".join(all_filtered))
    (cache_path / "merged.a3m").write_text(merged)

    n_seqs = sum(1 for line in merged.splitlines() if line.startswith(">"))
    print(f"[colabsearch] Cached unpaired {n_seqs} sequences for {seq_hash}")

    marker.write_text(f"cached_at: {__import__('datetime').datetime.now().isoformat()}")
    return str(cache_path)


def _fetch_paired(sequences: list[str], use_env: bool, cache_dir: Path) -> str:
    """Fetch paired MSA for multiple sequences.

    Returns path to cache directory on volume.
    """
    import tarfile
    from io import BytesIO

    combined = "|".join(sequences) + f"|use_env={use_env}"
    combined_hash = sha256(combined.encode()).hexdigest()[:16]
    cache_path = cache_dir / combined_hash
    marker = cache_path / "_COMPLETE"

    if marker.exists():
        print(f"[colabsearch] Paired cache HIT: {combined_hash}")
        return str(cache_path)

    print(f"[colabsearch] Paired cache MISS: {combined_hash} ({len(sequences)} chains)")

    # Build multi-sequence query
    query = ""
    for i, seq in enumerate(sequences):
        query += f">{101 + i}\n{seq}\n"

    mode = "pairgreedy-env" if use_env else "pairgreedy"
    tar_bytes = _submit_and_download(query, mode)

    cache_path.mkdir(parents=True, exist_ok=True)

    pair_content = None
    with tarfile.open(fileobj=BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.name == "pair.a3m":
                f = tar.extractfile(member)
                if f:
                    # ColabFold pair API embeds NUL bytes at chain boundaries
                    pair_content = f.read().decode().replace("\x00", "")

    if not pair_content or not pair_content.strip():
        raise RuntimeError(f"pair.a3m not found or empty for {len(sequences)} chains")

    (cache_path / "pair.a3m").write_text(pair_content)
    marker.write_text(f"cached_at: {__import__('datetime').datetime.now().isoformat()}")
    print(f"[colabsearch] Cached paired MSA for {combined_hash}")
    return str(cache_path)


@app.function(image=colabsearch_image, timeout=30 * 60, volumes={"/cache": CACHE_VOLUME})
def colabsearch_fetch(sequences: list[str], use_env: bool = True) -> dict:
    """Fetch unpaired + paired MSAs for protein sequences.

    Returns MsaResult dict:
        {
            "unpaired": {seq: "/cache/colabsearch/unpaired/{hash}", ...},
            "paired_dir": "/cache/colabsearch/paired/{hash}" | None,
            "sequences": [seq1, seq2, ...]
        }
    """
    unpaired_base = Path("/cache/colabsearch/unpaired")
    paired_base = Path("/cache/colabsearch/paired")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Fetch all MSAs in parallel (unpaired per-chain + paired)
    unpaired = {}
    paired_path = None

    with ThreadPoolExecutor(max_workers=len(sequences) + 1) as pool:
        unpaired_futures = {
            pool.submit(_fetch_unpaired, seq, use_env, unpaired_base): seq
            for seq in sequences
        }
        paired_future = None
        if len(sequences) > 1:
            paired_future = pool.submit(_fetch_paired, sequences, use_env, paired_base)

        for fut in as_completed(unpaired_futures):
            seq = unpaired_futures[fut]
            unpaired[seq] = fut.result()

        if paired_future:
            paired_path = paired_future.result()

    CACHE_VOLUME.commit()

    return {
        "unpaired": unpaired,
        "paired_dir": paired_path,
        "sequences": sequences,
    }
