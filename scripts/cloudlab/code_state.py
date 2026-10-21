#!/usr/bin/env python3
"""code_state.py: the code a CloudLab run ships, as key=value lines for run.env.

Kevin, 2026-10-19 (restart from scratch): every run of the new pipeline must be traceable
to one code state. The root repository is not committed, so a commit alone does not
identify it: code_hash is a sha256 over the content of every file a run ships (the root's
learning_agent, config, proto and harness scripts, and each engine submodule's tracked and
untracked, non-ignored files), path by path, so it changes with any edit, committed or not.
The analysis tooling under scripts/cloudlab/tmp is not shipped and not hashed.

  code_state.py                 print the state (root and submodule HEADs, dirty flags, hash)
  code_state.py --freeze NAME   write it to FREEZE (repo root) as the frozen state
  code_state.py --check         exit 1 unless the current code_hash equals FREEZE's
  --prior PATH                  also print the prior's md5 and the code_hash it was trained
                                under (every prior records one); with --check, exit 1 unless
                                that equals FREEZE's too
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / "FREEZE"
# Root paths a run ships (lib_bedrock.sh / lib_rustbft.sh / run_smartbft.sh / run_tendermint.sh
# tar these, or a subset of them).
ROOT_PATHS = ("learning_agent", "config", "proto", "scripts/apply_network_failures.py", "scripts/cloudlab")
# Priors are installed after the freeze (pretraining); each run records its prior md5 instead.
ROOT_EXCLUDE = ("scripts/cloudlab/tmp/", "learning_agent/pretrained_models/")
SUBMODULES = ("CppBedrock", "SmartBFT", "autobahn", "cometbft")
SKIP_SUFFIXES = (".pyc", ".DS_Store")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def _files(repo: Path, paths=()) -> list:
    out = _git(repo, "ls-files", "-co", "--exclude-standard", "-z", "--", *paths)
    names = [n for n in out.split("\0") if n]
    return sorted(n for n in names if not n.endswith(SKIP_SUFFIXES) and (repo / n).is_file())


def state() -> dict:
    digest = hashlib.sha256()
    result = {"root_head": _git(ROOT, "rev-parse", "HEAD").strip()}
    for name in _files(ROOT, ROOT_PATHS):
        if name.startswith(ROOT_EXCLUDE) or any(name == s or name.startswith(s + "/") for s in SUBMODULES):
            continue
        digest.update(name.encode() + b"\0" + hashlib.sha256((ROOT / name).read_bytes()).digest())
    for sub in SUBMODULES:
        repo = ROOT / sub
        result[f"{sub.lower()}_head"] = _git(repo, "rev-parse", "HEAD").strip()
        result[f"{sub.lower()}_dirty"] = "1" if _git(repo, "status", "--porcelain", "--untracked-files=no").strip() else "0"
        for name in _files(repo):
            digest.update(f"{sub}/{name}".encode() + b"\0" + hashlib.sha256((repo / name).read_bytes()).digest())
    result["code_hash"] = digest.hexdigest()
    return result


def prior_state(path: str) -> dict:
    """md5 and recorded code_hash of a prior: a GP prior's code_hash, or a LinUCB/PPO
    checkpoint's provenance.code_hash (continuous_bandit.train_gp_prior, pretrain.py)."""
    import numpy as np

    data = Path(path).read_bytes()
    with np.load(path, allow_pickle=False) as npz:
        keys = [k for k in ("code_hash", "provenance.code_hash") if k in npz.files]
        if not keys:
            raise SystemExit(f"code_state: prior {path} records no code_hash; retrain it")
        code_hash = str(npz[keys[0]])
    return {"prior_md5": hashlib.md5(data).hexdigest(), "prior_code_hash": code_hash}


def read_freeze() -> dict:
    if not FREEZE.exists():
        raise SystemExit(f"no {FREEZE}: freeze first (code_state.py --freeze NAME)")
    return dict(line.split("=", 1) for line in FREEZE.read_text().splitlines() if "=" in line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--freeze", metavar="NAME")
    group.add_argument("--check", action="store_true")
    parser.add_argument("--prior", metavar="PATH")
    args = parser.parse_args()
    current = state()
    if args.prior:
        if args.freeze:
            parser.error("--prior does not go with --freeze")
        current.update(prior_state(args.prior))
    if args.freeze:
        lines = [f"freeze_name={args.freeze}"] + [f"{k}={v}" for k, v in current.items()]
        FREEZE.write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return
    if args.check:
        frozen = read_freeze()
        if frozen.get("code_hash") != current["code_hash"]:
            print(f"code_state: the shipped code differs from {FREEZE} ({frozen.get('freeze_name')}): "
                  f"frozen {frozen.get('code_hash', '?')[:12]}, now {current['code_hash'][:12]}", file=sys.stderr)
            sys.exit(1)
        if args.prior and current["prior_code_hash"] != frozen.get("code_hash"):
            print(f"code_state: prior {args.prior} was trained under code {current['prior_code_hash'][:12]}, "
                  f"not the frozen {frozen.get('code_hash', '?')[:12]} ({frozen.get('freeze_name')})", file=sys.stderr)
            sys.exit(1)
        print(f"freeze_name={frozen.get('freeze_name')}")
    for k, v in current.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
