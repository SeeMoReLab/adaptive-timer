#!/usr/bin/env python3
"""Generate keys, committee.json, and parameters.json for the Rust BFT
baselines (autobahn/baselines workspace), or keys and committee.json for the
CppBedrock replicas (--workspace bedrock).

Runs on node0 after the remote build (invoked by lib_rustbft.sh or
lib_bedrock.sh). For the Rust workspaces the freshly built `node` binary
generates the keys so the key format always matches the binary; for bedrock
the keys are RSA PEM files produced by openssl. Can also be run locally.

Replica i gets address ips[i] with ports (front, mempool, consensus) =
(front_port, mempool_port, consensus_port). On CloudLab every replica has its
own overlay IP, so the IPs must be distinct. CppBedrock uses a single port per
replica (--front-port) for peers and clients alike.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", choices=("baselines", "autobahn", "bedrock"), default="baselines", help="Which workspace's config format to generate")
    parser.add_argument("--node-bin", default=None, help="Path to the built node binary (for key generation; Rust workspaces only)")
    parser.add_argument("--ips", required=True, help="Comma-separated replica IPs, replica order (replica_id 0..n-1)")
    parser.add_argument("--out-dir", required=True, help="Directory for node_<i>.json, committee.json, parameters.json")
    parser.add_argument("--front-port", type=int, default=9000, help="Client transactions port (default 9000)")
    parser.add_argument("--mempool-port", type=int, default=9100, help="Mempool-to-mempool port (default 9100)")
    parser.add_argument("--consensus-port", type=int, default=9200, help="Consensus port (default 9200)")
    parser.add_argument("--parameters-template", default=None, help="Checked-in parameters.json template (config/<protocol>/parameters.json); the source of truth for all knobs not explicitly overridden below (Rust workspaces only)")
    # Every knob below defaults to None = "take the template's value"; a
    # given flag overrides the template field(s) it maps to.
    parser.add_argument("--timeout-delay-ms", type=int, default=None, help="Consensus timeout_delay in ms (the adaptive knob's initial value)")
    parser.add_argument("--max-payload-size", type=int, default=None, help="Mempool payload seal size in bytes (baselines) or worker batch size (autobahn)")
    parser.add_argument("--min-block-delay-ms", type=int, default=None)
    return parser.parse_args()


def generate_bedrock(args: argparse.Namespace, ips: list, out_dir: Path) -> int:
    """CppBedrock: RSA keys per replica plus the client key, and a committee
    with one endpoint per replica (ids 0..n-1)."""
    keys_dir = out_dir / "keys"
    keys_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(ips)):
        private = keys_dir / f"server_{i}_private.pem"
        public = keys_dir / f"server_{i}_public.pem"
        subprocess.run(
            ["openssl", "genpkey", "-algorithm", "RSA", "-out", str(private), "-pkeyopt", "rsa_keygen_bits:2048"],
            check=True, stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["openssl", "rsa", "-pubout", "-in", str(private), "-out", str(public)],
            check=True, stderr=subprocess.DEVNULL,
        )
    subprocess.run(
        ["openssl", "genpkey", "-algorithm", "RSA", "-out", str(keys_dir / "client_private.pem"), "-pkeyopt", "rsa_keygen_bits:2048"],
        check=True, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["openssl", "rsa", "-pubout", "-in", str(keys_dir / "client_private.pem"), "-out", str(keys_dir / "client_public.pem")],
        check=True, stderr=subprocess.DEVNULL,
    )

    replicas = [
        {"id": i, "host": ip, "port": args.front_port}
        for i, ip in enumerate(ips)
    ]
    with (out_dir / "committee.json").open("w") as f:
        json.dump({"replicas": replicas}, f, indent=2)
        f.write("\n")
    with (out_dir / "client_targets.txt").open("w") as f:
        for r in replicas:
            f.write(f"{r['host']}:{r['port']}\n")
    print(f"wrote {len(ips)} RSA key pairs, the client key, committee.json, client_targets.txt to {out_dir}")
    return 0


def build_parameters(args: argparse.Namespace) -> dict:
    with open(args.parameters_template) as f:
        parameters = json.load(f)

    def override(target: dict, field: str, value):
        if value is not None:
            target[field] = value

    if args.workspace == "baselines":
        if "consensus" not in parameters or "mempool" not in parameters:
            raise SystemExit(
                f"{args.parameters_template} is not a baselines parameters template "
                "(expected top-level 'consensus' and 'mempool' sections)"
            )
        consensus = parameters["consensus"]
        mempool = parameters["mempool"]
        override(consensus, "timeout_delay", args.timeout_delay_ms)
        override(consensus, "max_payload_size", args.max_payload_size)
        override(consensus, "min_block_delay", args.min_block_delay_ms)
        override(mempool, "max_payload_size", args.max_payload_size)
        override(mempool, "min_block_delay", args.min_block_delay_ms)
    else:
        if "consensus" in parameters or "timeout_delay" not in parameters:
            raise SystemExit(
                f"{args.parameters_template} is not an autobahn parameters template "
                "(expected a flat parameter object with 'timeout_delay')"
            )
        override(parameters, "timeout_delay", args.timeout_delay_ms)
        override(parameters, "batch_size", args.max_payload_size)
    return parameters


# Per-replica port roles for the autobahn workspace, added to the base
# offset. The transactions port equals --front-port so client_targets.txt has
# the same meaning in both workspaces.
AUTOBAHN_PORT_WORKER_TO_WORKER = 101
AUTOBAHN_PORT_PRIMARY_TO_WORKER = 102
AUTOBAHN_PORT_PRIMARY_TO_PRIMARY = 103
AUTOBAHN_PORT_WORKER_TO_PRIMARY = 104
AUTOBAHN_PORT_CONSENSUS = 105


def main() -> int:
    args = parse_args()
    ips = [ip.strip() for ip in args.ips.split(",") if ip.strip()]
    if not ips:
        print("no replica IPs given", file=sys.stderr)
        return 2
    if len(set(ips)) != len(ips):
        print("duplicate replica IPs", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.workspace == "bedrock":
        return generate_bedrock(args, ips, out_dir)
    if args.node_bin is None or args.parameters_template is None:
        print("--node-bin and --parameters-template are required for Rust workspaces", file=sys.stderr)
        return 2

    # Validate the template before the (slower) key generation step.
    parameters = build_parameters(args)

    keygen_subcommand = "keys" if args.workspace == "baselines" else "generate_keys"
    names = []
    for i in range(len(ips)):
        key_file = out_dir / f"node_{i}.json"
        key_file.unlink(missing_ok=True)
        subprocess.run(
            [args.node_bin, keygen_subcommand, "--filename", str(key_file)],
            check=True,
        )
        with key_file.open() as f:
            key = json.load(f)
        if "name" not in key or "secret" not in key:
            print(f"unexpected key file format in {key_file}", file=sys.stderr)
            return 1
        names.append(key["name"])

    if args.workspace == "baselines":
        consensus_authorities = {}
        mempool_authorities = {}
        for name, ip in zip(names, ips):
            consensus_authorities[name] = {
                "name": name,
                "stake": 1,
                "address": f"{ip}:{args.consensus_port}",
            }
            mempool_authorities[name] = {
                "name": name,
                "front_address": f"{ip}:{args.front_port}",
                "mempool_address": f"{ip}:{args.mempool_port}",
            }

        committee = {
            "consensus": {"authorities": consensus_authorities, "epoch": 1},
            "mempool": {"authorities": mempool_authorities, "epoch": 1},
        }
    else:
        authorities = {}
        for name, ip in zip(names, ips):
            authorities[name] = {
                "stake": 1,
                "consensus": {
                    "consensus_to_consensus": f"{ip}:{args.consensus_port + AUTOBAHN_PORT_CONSENSUS}",
                },
                "primary": {
                    "primary_to_primary": f"{ip}:{args.consensus_port + AUTOBAHN_PORT_PRIMARY_TO_PRIMARY}",
                    "worker_to_primary": f"{ip}:{args.consensus_port + AUTOBAHN_PORT_WORKER_TO_PRIMARY}",
                },
                "workers": {
                    "0": {
                        "transactions": f"{ip}:{args.front_port}",
                        "worker_to_worker": f"{ip}:{args.consensus_port + AUTOBAHN_PORT_WORKER_TO_WORKER}",
                        "primary_to_worker": f"{ip}:{args.consensus_port + AUTOBAHN_PORT_PRIMARY_TO_WORKER}",
                    }
                },
            }
        committee = {"authorities": authorities}

    with (out_dir / "committee.json").open("w") as f:
        json.dump(committee, f, indent=2)
        f.write("\n")
    with (out_dir / "parameters.json").open("w") as f:
        json.dump(parameters, f, indent=2)
        f.write("\n")

    # Client targets: one front endpoint per replica, replica order.
    with (out_dir / "client_targets.txt").open("w") as f:
        for ip in ips:
            f.write(f"{ip}:{args.front_port}\n")

    # Public key -> global replica id, for failure-spec targeting and
    # learning reports.
    replica_map = {name: i for i, name in enumerate(names)}
    with (out_dir / "replica_map.json").open("w") as f:
        json.dump(replica_map, f, indent=2)
        f.write("\n")

    print(f"wrote {len(ips)} keys, committee.json, parameters.json, replica_map.json, client_targets.txt to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
