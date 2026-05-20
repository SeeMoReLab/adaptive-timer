from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

import agent_pb2
from models import get_report_feature_field_names


def _extract_state(report_local: agent_pb2.ReportLocal, protocol: int) -> object:
    if protocol == agent_pb2.PROTOCOL_PBFT:
        if not report_local.HasField("pbft_state"):
            raise ValueError(
                f"ReportLocal for node {report_local.node_id} missing pbft_state"
            )
        return report_local.pbft_state
    if protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
        if not report_local.HasField("two_pc_over_pbft_state"):
            raise ValueError(
                "ReportLocal for node "
                f"{report_local.node_id} missing two_pc_over_pbft_state"
            )
        return report_local.two_pc_over_pbft_state
    if protocol == agent_pb2.PROTOCOL_SBFT:
        if not report_local.HasField("sbft_state"):
            raise ValueError(
                f"ReportLocal for node {report_local.node_id} missing sbft_state"
            )
        return report_local.sbft_state
    if protocol == agent_pb2.PROTOCOL_TENDERMINT:
        if not report_local.HasField("tendermint_report"):
            raise ValueError(
                f"ReportLocal for node {report_local.node_id} missing tendermint_report"
            )
        return report_local.tendermint_report
    if protocol in (
        agent_pb2.PROTOCOL_HOTSTUFF,
        agent_pb2.PROTOCOL_HOTSTUFF2,
        agent_pb2.PROTOCOL_CHAINED_HOTSTUFF,
    ):
        if not report_local.HasField("hotstuff_state"):
            raise ValueError(
                f"ReportLocal for node {report_local.node_id} missing hotstuff_state"
            )
        return report_local.hotstuff_state
    if protocol == agent_pb2.PROTOCOL_ZYZZYVA:
        if not report_local.HasField("zyzzyva_state"):
            raise ValueError(
                f"ReportLocal for node {report_local.node_id} missing zyzzyva_state"
            )
        return report_local.zyzzyva_state
    raise ValueError(f"Unsupported protocol: {protocol}")


def _report_to_array(protocol: int, report: object) -> np.ndarray:
    fields = get_report_feature_field_names(protocol)
    return np.array([float(getattr(report, field_name)) for field_name in fields], dtype=float)


def _array_to_report(protocol: int, values: np.ndarray) -> object:
    fields = get_report_feature_field_names(protocol)
    kwargs = {
        field_name: float(values[idx]) for idx, field_name in enumerate(fields)
    }

    # Integer counters need explicit integer conversion for protobuf uint32 fields.
    for int_field in (
        "total_transactions",
        "total_consensus_instances",
        "leader_change_count",
        "regency_change_count",
        "view_change_count",
        "single_shard_transactions",
        "cross_shard_transactions",
        "prepare_ok_count",
        "prepare_fail_count",
        "commit_ok_count",
        "abort_count",
        "in_doubt_count",
        "prepare_timeout_count",
        "commit_timeout_unknown_outcome_count",
        "abort_best_effort_timeout_count",
        "lock_contention_count",
        "reason_prepare_lock_busy_count",
        "reason_prepare_validation_fail_count",
        "reason_prepare_timeout_count",
        "reason_prepare_other_count",
        "reason_commit_timeout_unknown_outcome_count",
        "reason_abort_best_effort_timeout_count",
        "reason_other_count",
    ):
        if int_field in kwargs:
            kwargs[int_field] = max(0, int(round(float(kwargs[int_field]))))

    if protocol == agent_pb2.PROTOCOL_PBFT:
        return agent_pb2.PbftReport(**kwargs)
    if protocol == agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT:
        return agent_pb2.TwoPcOverPbftReport(**kwargs)
    if protocol == agent_pb2.PROTOCOL_SBFT:
        return agent_pb2.SbftReport(**kwargs)
    if protocol == agent_pb2.PROTOCOL_TENDERMINT:
        return agent_pb2.TendermintReport(**kwargs)
    if protocol in (
        agent_pb2.PROTOCOL_HOTSTUFF,
        agent_pb2.PROTOCOL_HOTSTUFF2,
        agent_pb2.PROTOCOL_CHAINED_HOTSTUFF,
    ):
        return agent_pb2.HotstuffReport(**kwargs)
    if protocol == agent_pb2.PROTOCOL_ZYZZYVA:
        return agent_pb2.ZyzzyvaReport(**kwargs)
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
            agent_pb2.PROTOCOL_TWO_PC_OVER_PBFT,
            agent_pb2.PROTOCOL_SBFT,
            agent_pb2.PROTOCOL_TENDERMINT,
            agent_pb2.PROTOCOL_HOTSTUFF,
            agent_pb2.PROTOCOL_HOTSTUFF2,
            agent_pb2.PROTOCOL_CHAINED_HOTSTUFF,
            agent_pb2.PROTOCOL_ZYZZYVA,
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
        return _array_to_report(self.protocol, aggregated_values)

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

    def _calculate(self, data: Dict[int, np.ndarray], node_id: int) -> np.ndarray:
        if node_id not in data:
            return np.median(np.stack(list(data.values())), axis=0)

        others = [arr for k, arr in data.items() if k != node_id]
        weighted_node = ([data[node_id]] * self.weights) + others
        return np.median(np.stack(weighted_node), axis=0)
