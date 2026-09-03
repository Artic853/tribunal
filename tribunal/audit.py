"""
Tamper-evident audit log.

Payment risk decisions get argued about later: by the merchant whose customer was
declined, by the cardholder disputing a charge, and eventually by a regulator or an
auditor asking "on what basis". A log that can be quietly edited afterwards answers
none of those.

Each record carries the SHA-256 of the previous record, so the file is a hash chain.
Changing or deleting any past record breaks every hash after it, and `verify()`
reports the exact index where the chain first fails. This does not stop someone with
write access from rewriting the whole file - that needs an external anchor - but it
does mean a decision cannot be silently altered in place, which is the failure mode
that actually happens.

Records are append-only JSONL: one decision per line, greppable, no database needed.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

GENESIS = "0" * 64


def _canonical(obj: Dict[str, Any]) -> bytes:
    """Stable serialisation - key order and separators fixed, so the hash of a
    record does not depend on dict insertion order."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def record_hash(prev_hash: str, payload: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(prev_hash.encode())
    h.update(_canonical(payload))
    return h.hexdigest()


@dataclass
class AuditRecord:
    seq: int
    prev_hash: str
    hash: str
    payload: Dict[str, Any]

    def to_line(self) -> str:
        return json.dumps(
            {"seq": self.seq, "prev_hash": self.prev_hash,
             "hash": self.hash, "payload": self.payload},
            sort_keys=True, separators=(",", ":"), default=str,
        )


class AuditLog:
    """Append-only hash-chained decision log."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._seq, self._last_hash = self._resume()

    def _resume(self) -> Tuple[int, str]:
        """Pick up the chain where a previous process left off."""
        if not os.path.exists(self.path):
            return 0, GENESIS
        last = None
        with open(self.path, "r") as fh:
            for line in fh:
                if line.strip():
                    last = line
        if last is None:
            return 0, GENESIS
        rec = json.loads(last)
        return rec["seq"] + 1, rec["hash"]

    def append(self, payload: Dict[str, Any]) -> AuditRecord:
        with self._lock:
            h = record_hash(self._last_hash, payload)
            rec = AuditRecord(self._seq, self._last_hash, h, payload)
            with open(self.path, "a") as fh:
                fh.write(rec.to_line() + "\n")
            self._seq += 1
            self._last_hash = h
            return rec

    def __len__(self) -> int:
        return self._seq

    @property
    def head(self) -> str:
        """Current chain head. Publishing this somewhere you do not control (a
        timestamping service, another team's store) is what upgrades the log from
        tamper-evident to tamper-evident-against-yourself."""
        return self._last_hash


def read_records(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def verify(path: str) -> Tuple[bool, Optional[int], str]:
    """Recompute the chain.

    Returns (ok, first_bad_index, message).
    """
    prev = GENESIS
    n = 0
    for i, rec in enumerate(read_records(path)):
        if rec["prev_hash"] != prev:
            return False, i, f"record {i} points at {rec['prev_hash'][:12]}, expected {prev[:12]}"
        expected = record_hash(prev, rec["payload"])
        if rec["hash"] != expected:
            return False, i, f"record {i} payload does not match its hash - it was modified"
        if rec["seq"] != i:
            return False, i, f"record {i} has sequence number {rec['seq']}"
        prev = rec["hash"]
        n += 1
    return True, None, f"chain intact across {n:,} records, head {prev[:16]}"


def query(path: str, **filters: Any) -> List[Dict[str, Any]]:
    """Small convenience filter over payload fields, e.g. query(p, decision='block')."""
    out = []
    for rec in read_records(path):
        pl = rec["payload"]
        if all(pl.get(k) == v for k, v in filters.items()):
            out.append(rec)
    return out
