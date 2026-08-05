from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from google.protobuf.descriptor import FieldDescriptor

import agent_pb2
from models import get_report_feature_field_names


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
    raise ValueError(f"Unsupported protocol: {protocol}")


def _report_to_array(protocol: int, report: object) -> np.ndarray:
    fields = get_report_feature_field_names(protocol)
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


def _report_message_class(protocol: int) -> type:
    if protocol == agent_pb2.PROTOCOL_PBFT:
        return agent_pb2.PbftReport
    if protocol == agent_pb2.PROTOCOL_SBFT:
        return agent_pb2.SbftReport
    if protocol == agent_pb2.PROTOCOL_TENDERMINT:
        return agent_pb2.TendermintReport
    raise ValueError(f"Unsupported protocol: {protocol}")


def _array_to_report(protocol: int, values: np.ndarray) -> object:
    fields = get_report_feature_field_names(protocol)
    message_cls = _report_message_class(protocol)
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
