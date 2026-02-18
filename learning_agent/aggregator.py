from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
import agent_pb2


def _report_to_array(report: agent_pb2.Report) -> np.ndarray:
    return np.array(
        [
            float(report.processed_transactions),
            float(report.avg_message_delay),
            float(report.max_message_delay),
            float(report.min_message_delay),
            float(report.std_message_delay),
            float(report.p95_message_delay),
            float(report.p99_message_delay),
            float(report.throughput_tps),
            float(report.timeout_violation_rate),
            float(report.no_op_rate),
            float(report.avg_batch_size),
            float(report.p95_batch_size),
            float(report.leader_change_count),
            float(report.regency_change_count),
            float(report.avg_propose_delay),
            float(report.avg_write_delay),
            float(report.avg_accept_delay),
            float(report.avg_post_decision_delay),
        ],
        dtype=float,
    )


def _array_to_report(values: np.ndarray) -> agent_pb2.Report:
    return agent_pb2.Report(
        processed_transactions=max(0, int(round(float(values[0])))),
        avg_message_delay=float(values[1]),
        max_message_delay=float(values[2]),
        min_message_delay=float(values[3]),
        std_message_delay=float(values[4]),
        p95_message_delay=float(values[5]),
        p99_message_delay=float(values[6]),
        throughput_tps=float(values[7]),
        timeout_violation_rate=float(values[8]),
        no_op_rate=float(values[9]),
        avg_batch_size=float(values[10]),
        p95_batch_size=float(values[11]),
        leader_change_count=max(0, int(round(float(values[12])))),
        regency_change_count=max(0, int(round(float(values[13])))),
        avg_propose_delay=float(values[14]),
        avg_write_delay=float(values[15]),
        avg_accept_delay=float(values[16]),
        avg_post_decision_delay=float(values[17]),
    )


def _batch_to_reports_by_node(report_batch: agent_pb2.ReportBatch) -> Dict[int, agent_pb2.Report]:
    reports_by_node: Dict[int, agent_pb2.Report] = {}
    for report_local in report_batch.reports:
        reports_by_node[report_local.node_id] = report_local.state
    return reports_by_node


class BaseAggregator(ABC):
    def aggregate(self, node_id: int, report_batch: agent_pb2.ReportBatch) -> agent_pb2.Report:
        reports_by_node = _batch_to_reports_by_node(report_batch)
        if not reports_by_node:
            raise ValueError("No reports provided for aggregation")

        data = {node: _report_to_array(report) for node, report in reports_by_node.items()}
        aggregated_values = self._calculate(data, node_id=node_id)
        return _array_to_report(aggregated_values)

    @abstractmethod
    def _calculate(self, data: Dict[int, np.ndarray], node_id: int) -> np.ndarray:
        pass


class MedianAggregator(BaseAggregator):
    def _calculate(self, data: Dict[int, np.ndarray], node_id: int) -> np.ndarray:
        _ = node_id
        return np.median(np.stack(list(data.values())), axis=0)


class WeightedMedianAggregator(BaseAggregator):
    def __init__(self, weights: int):
        if weights < 1:
            raise ValueError(f"weights must be >= 1, got {weights}")
        self.weights = weights

    def _calculate(self, data: Dict[int, np.ndarray], node_id: int) -> np.ndarray:
        if node_id not in data.keys():
            return np.median(np.stack(list(data.values())), axis=0)

        others = [arr for k, arr in data.items() if k != node_id]
        weighted_node = ([data[node_id]] * self.weights) + others

        return np.median(np.stack(weighted_node), axis=0)


def _make_sample_report(
    processed_transactions: int,
    avg_message_delay: float,
    throughput_tps: float,
    leader_change_count: int,
    regency_change_count: int,
) -> agent_pb2.Report:
    return agent_pb2.Report(
        processed_transactions=processed_transactions,
        avg_message_delay=avg_message_delay,
        max_message_delay=avg_message_delay + 15.0,
        min_message_delay=max(0.0, avg_message_delay - 10.0),
        std_message_delay=max(0.0, avg_message_delay / 8.0),
        p95_message_delay=avg_message_delay + 20.0,
        p99_message_delay=avg_message_delay + 35.0,
        throughput_tps=throughput_tps,
        timeout_violation_rate=0.01 * leader_change_count,
        no_op_rate=0.02 * regency_change_count,
        avg_batch_size=8.0 + (processed_transactions % 5),
        p95_batch_size=12.0 + (processed_transactions % 7),
        leader_change_count=leader_change_count,
        regency_change_count=regency_change_count,
        avg_propose_delay=avg_message_delay / 2.0,
        avg_write_delay=avg_message_delay / 3.0,
        avg_accept_delay=avg_message_delay / 4.0,
        avg_post_decision_delay=avg_message_delay / 5.0,
    )


def _print_result(label: str, report: agent_pb2.Report) -> None:
    print(
        f"{label}: processed={report.processed_transactions}, "
        f"avg_delay={report.avg_message_delay:.2f}, "
        f"throughput={report.throughput_tps:.2f}, "
        f"leader_changes={report.leader_change_count}, "
        f"regency_changes={report.regency_change_count}"
    )


def _make_report_batch(
    reports_by_node: Dict[int, agent_pb2.Report], episode: int
) -> agent_pb2.ReportBatch:
    batch = agent_pb2.ReportBatch(episode=episode)
    for node_id, report in reports_by_node.items():
        entry = batch.reports.add(node_id=node_id, episode=episode)
        entry.state.CopyFrom(report)
    return batch


def main() -> None:
    reports_by_node = {
        0: _make_sample_report(1200, 80.0, 4200.0, 0, 0),
        1: _make_sample_report(900, 110.0, 3500.0, 1, 1),
        2: _make_sample_report(1400, 75.0, 4600.0, 0, 1),
        3: _make_sample_report(1000, 95.0, 3900.0, 2, 2),
        4: _make_sample_report(1300, 85.0, 4300.0, 1, 0),
    }

    node_id = 0
    batch = _make_report_batch(reports_by_node, episode=1)
    median_aggregator = MedianAggregator()
    weighted_aggregator = WeightedMedianAggregator(weights=3)

    median_report = median_aggregator.aggregate(node_id=node_id, report_batch=batch)
    weighted_report = weighted_aggregator.aggregate(node_id=node_id, report_batch=batch)

    _print_result("median", median_report)
    _print_result("weighted_median(node present)", weighted_report)

    reports_without_node = {k: v for k, v in reports_by_node.items() if k != node_id}
    fallback_batch = _make_report_batch(reports_without_node, episode=2)
    weighted_fallback_report = weighted_aggregator.aggregate(
        node_id=node_id, report_batch=fallback_batch
    )
    median_fallback_report = median_aggregator.aggregate(
        node_id=node_id, report_batch=fallback_batch
    )

    _print_result("weighted_median(node missing -> median fallback)", weighted_fallback_report)
    _print_result("median(same input)", median_fallback_report)


if __name__ == "__main__":
    main()
