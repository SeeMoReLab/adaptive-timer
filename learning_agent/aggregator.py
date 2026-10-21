from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np
from google.protobuf.descriptor import FieldDescriptor

import agent_pb2
from models import get_report_field_names, report_message_class


def _extract_state(report_local: agent_pb2.ReportLocal, protocol: int) -> object:
    if protocol == agent_pb2.PROTOCOL_PBFT:
        if not report_local.HasField("pbft_state"):
            raise ValueError(
                f"ReportLocal for node {report_local.node_id} missing pbft_state"
            )
        return report_local.pbft_state
    if protocol == agent_pb2.PROTOCOL_SBFT:
        if not report_local.HasField("sbft_state"):
            raise ValueError(
                f"ReportLocal for node {report_local.node_id} missing sbft_state"
            )
        return report_local.sbft_state
    if protocol == agent_pb2.PROTOCOL_TENDERMINT:
        if not report_local.HasField("tendermint_state"):
            raise ValueError(
                f"ReportLocal for node {report_local.node_id} missing tendermint_state"
            )
        return report_local.tendermint_state
    if protocol == agent_pb2.PROTOCOL_HOTSTUFF:
        if not report_local.HasField("hotstuff_state"):
            raise ValueError(
                f"ReportLocal for node {report_local.node_id} missing hotstuff_state"
            )
        return report_local.hotstuff_state
    if protocol == agent_pb2.PROTOCOL_AUTOBAHN:
        if not report_local.HasField("autobahn_state"):
            raise ValueError(
                f"ReportLocal for node {report_local.node_id} missing autobahn_state"
            )
        return report_local.autobahn_state
    if protocol == agent_pb2.PROTOCOL_SMARTBFT:
        if not report_local.HasField("smartbft_state"):
            raise ValueError(
                f"ReportLocal for node {report_local.node_id} missing smartbft_state"
            )
        return report_local.smartbft_state
    raise ValueError(f"Unsupported protocol: {protocol}")


SBFT_LEADER_MEASURED_FIELDS = (
    "nth_prepare_gap_ms",
    "nth_prepare_seen_count",
    "nth_prepare_unseen_count",
)


# The hold-free fields only some replicas measure (SmartBFT's followers, HotStuff's next
# leader of a round, Autobahn's non-leaders of a slot).
_HOLD_FREE_FIELDS = frozenset({"proposal_to_decision_ms", "proposal_to_qc_ms", "prepare_to_commit_ms"})


def measured_only_fields(protocol: int) -> Tuple[str, ...]:
    """The window statistics of samples (latencies, inter-commit gaps, batch sizes) and the
    hold-free fields, taken as the median of the reports that measured them (> 0): a replica
    with no sample in the window reports 0, which is no measurement. Every protocol alike
    (Kevin, 2026-10-19, restart; until then only the hold-free fields). Counts and rates keep
    the plain median; SBFT's leader-measured fields keep the leader's report."""
    return tuple(
        name for name in get_report_field_names(protocol)
        if (name.endswith("_latency_ms") or "_inter_commit_gap_ms" in name
            or name.endswith("_batch_size") or name in _HOLD_FREE_FIELDS)
        and not (protocol == agent_pb2.PROTOCOL_SBFT and name in SBFT_LEADER_MEASURED_FIELDS)
    )


def _report_to_array(protocol: int, report: object) -> np.ndarray:
    # Every report field, not only the model's inputs, so the aggregated report
    # carries all of them into the dataset (a field left out would read 0).
    fields = get_report_field_names(protocol)
    return np.array([float(getattr(report, field_name)) for field_name in fields], dtype=float)


_INTEGER_CPP_TYPES = frozenset(
    {
        FieldDescriptor.CPPTYPE_INT32,
        FieldDescriptor.CPPTYPE_INT64,
        FieldDescriptor.CPPTYPE_UINT32,
        FieldDescriptor.CPPTYPE_UINT64,
    }
)


def _integer_field_names(message_cls: type) -> frozenset:
    """Integer fields of a report message, read from the descriptor.

    A median is always a float, so every integer field needs rounding before it
    can be assigned. Deriving the set from the descriptor keeps this correct
    when report fields are added, which a hand maintained list does not.
    """
    return frozenset(
        field.name
        for field in message_cls.DESCRIPTOR.fields
        if field.cpp_type in _INTEGER_CPP_TYPES
    )


def _array_to_report(protocol: int, values: np.ndarray) -> object:
    fields = get_report_field_names(protocol)
    message_cls = report_message_class(protocol)
    integer_fields = _integer_field_names(message_cls)

    kwargs = {}
    for idx, field_name in enumerate(fields):
        value = float(values[idx])
        if field_name in integer_fields:
            value = max(0, int(round(value)))
        kwargs[field_name] = value
    return message_cls(**kwargs)
    raise ValueError(f"Unsupported protocol: {protocol}")


def _batch_to_reports_by_node(
    protocol: int, report_batch: agent_pb2.ReportBatch
) -> Dict[int, object]:
    reports_by_node: Dict[int, object] = {}
    for report_local in report_batch.reports:
        reports_by_node[report_local.node_id] = _extract_state(report_local, protocol)
    return reports_by_node


class BaseAggregator(ABC):
    def __init__(self, protocol: int):
        if protocol not in (
            agent_pb2.PROTOCOL_PBFT,
            agent_pb2.PROTOCOL_SBFT,
            agent_pb2.PROTOCOL_TENDERMINT,
            agent_pb2.PROTOCOL_HOTSTUFF,
            agent_pb2.PROTOCOL_AUTOBAHN,
            agent_pb2.PROTOCOL_SMARTBFT,
        ):
            raise ValueError(f"Unsupported protocol: {protocol}")
        self.protocol = protocol

    def aggregate(self, node_id: int, report_batch: agent_pb2.ReportBatch) -> object:
        if (
            report_batch.protocol != agent_pb2.PROTOCOL_UNSPECIFIED
            and report_batch.protocol != self.protocol
        ):
            raise ValueError(
                "ReportBatch protocol mismatch: "
                f"batch={agent_pb2.Protocol.Name(report_batch.protocol)}, "
                f"aggregator={agent_pb2.Protocol.Name(self.protocol)}"
            )

        reports_by_node = _batch_to_reports_by_node(self.protocol, report_batch)
        if not reports_by_node:
            raise ValueError("No reports provided for aggregation")

        data = {node: _report_to_array(self.protocol, report) for node, report in reports_by_node.items()}
        aggregated_values = self._calculate(data, node_id=node_id)
        if self.protocol == agent_pb2.PROTOCOL_SBFT:
            # Only the replica that led a sequence measures its n-th prepare, so a
            # median would read the replicas that led nothing: take these fields
            # from the report that led the most sequences (lowest node id on a
            # tie, so every agent aggregates the same batch the same way).
            leader = min(
                reports_by_node,
                key=lambda node: (
                    -(reports_by_node[node].nth_prepare_seen_count
                      + reports_by_node[node].nth_prepare_unseen_count),
                    node,
                ),
            )
            fields = get_report_field_names(self.protocol)
            for name in SBFT_LEADER_MEASURED_FIELDS:
                aggregated_values[fields.index(name)] = float(
                    getattr(reports_by_node[leader], name)
                )
        # A replica without a sample reports 0 (the hold-free fields: every replica that
        # does not measure them, or delivered nothing it verified); any such 0 pulled the
        # median toward a value the simulator only gives a window with no sample. Take the
        # median of the reports that measured the field, 0 only when none did.
        fields = get_report_field_names(self.protocol)
        for measured_field in measured_only_fields(self.protocol):
            measured = [
                float(getattr(report, measured_field))
                for node, report in reports_by_node.items()
                if getattr(report, measured_field) > 0
                for _ in range(self._copies(node, node_id))
            ]
            aggregated_values[fields.index(measured_field)] = (
                float(np.median(measured)) if measured else 0.0
            )
        return _array_to_report(self.protocol, aggregated_values)

    def _copies(self, node: int, node_id: int) -> int:
        """How many times a node's value enters a median (WeightedMedianAggregator
        counts the local node more than once)."""
        return 1

    @abstractmethod
    def _calculate(self, data: Dict[int, np.ndarray], node_id: int) -> np.ndarray:
        pass


class MedianAggregator(BaseAggregator):
    def _calculate(self, data: Dict[int, np.ndarray], node_id: int) -> np.ndarray:
        del node_id
        return np.median(np.stack(list(data.values())), axis=0)


class WeightedMedianAggregator(BaseAggregator):
    def __init__(self, protocol: int, weights: int):
        super().__init__(protocol=protocol)
        if weights < 1:
            raise ValueError(f"weights must be >= 1, got {weights}")
        self.weights = weights

    def _copies(self, node: int, node_id: int) -> int:
        return self.weights if node == node_id else 1

    def _calculate(self, data: Dict[int, np.ndarray], node_id: int) -> np.ndarray:
        if node_id not in data:
            return np.median(np.stack(list(data.values())), axis=0)

        others = [arr for k, arr in data.items() if k != node_id]
        weighted_node = ([data[node_id]] * self.weights) + others
        return np.median(np.stack(weighted_node), axis=0)
