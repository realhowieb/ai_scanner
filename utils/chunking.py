# ai_scanner/utils/chunking.py
from __future__ import annotations
from typing import Iterable, List

def chunks(seq: List[str]|tuple[str,...], n: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def adaptive_chunk_size(total: int, base: int, max_workers: int) -> int:
    if total <= base:
        return total
    # keep total/buckets about equal
    buckets = max(1, min(max_workers*2, total // base + 1))
    return max(10, min(base, (total // buckets) or base))