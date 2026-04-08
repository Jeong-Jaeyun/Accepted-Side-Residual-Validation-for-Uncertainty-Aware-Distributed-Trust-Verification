from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Dict, Optional, Tuple

from qbm.model import EvidenceBatch, VerificationHistory, VerifyResult


class BaseVerifier(ABC):
    name = "base"

    def __init__(
        self,
        quorum: float = 2.0 / 3.0,
        strict_window_sequence: bool = False,
        enable_replay_check: bool = True,
        enable_timestamp_check: bool = True,
    ) -> None:
        self.quorum = quorum
        self.strict_window_sequence = strict_window_sequence
        self.enable_replay_check = enable_replay_check
        self.enable_timestamp_check = enable_timestamp_check

    @abstractmethod
    def verify(self, batch: EvidenceBatch, history: VerificationHistory) -> VerifyResult:
        raise NotImplementedError

    def required_quorum(self, total_nodes: int) -> int:
        if total_nodes <= 0:
            return 0
        if self.quorum <= 1.0:
            return max(1, int(math.ceil(total_nodes * self.quorum)))
        return min(total_nodes, max(1, int(self.quorum)))

    def replay_check(self, batch: EvidenceBatch, history: VerificationHistory) -> Optional[str]:
        if not self.enable_replay_check:
            return None

        seen = set()
        for e in batch.evidences:
            if e.event_id in seen:
                return f"duplicate event_id in batch: {e.event_id}"
            seen.add(e.event_id)
            if e.event_id in history.committed_event_ids:
                return f"replay event_id detected: {e.event_id}"
            if e.payload_hash() in history.committed_payload_hashes:
                return "replay payload hash detected"
        return None

    def time_sequence_check(self, batch: EvidenceBatch, history: VerificationHistory) -> Optional[str]:
        if not self.enable_timestamp_check:
            return None

        if self.strict_window_sequence and history.last_window_id is not None:
            if batch.window_id != history.last_window_id + 1:
                return "window sequence continuity violated"

        for e in batch.evidences:
            prev_ts = history.last_ts_by_node.get(e.node_id)
            if prev_ts is not None and e.timestamp_ms <= prev_ts:
                return f"timestamp monotonicity violated for node={e.node_id}"
        return None

    @staticmethod
    def context_consistency_ratio(batch: EvidenceBatch) -> float:
        if not batch.evidences:
            return 1.0
        counts: Dict[str, int] = {}
        for e in batch.evidences:
            counts[e.context_hash] = counts.get(e.context_hash, 0) + 1
        top = max(counts.values())
        return top / max(len(batch.evidences), 1)
