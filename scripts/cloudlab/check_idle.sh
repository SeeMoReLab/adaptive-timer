#!/usr/bin/env bash
# check_idle.sh <ssh key> <user@host>: exit 1 when the host still runs something from an earlier run: any tmux
# session (every harness runs replicas, agents, appliers and clients in tmux) or any process inside a
# network namespace (every harness puts replicas and agents in one). Each harness calls it for every
# target after taking the target lock (Kevin, 2026-10-19, restart): a leftover is an explicit error,
# never killed silently, because another protocol's run on the same host would share its CPU and
# network. Clean up with the harness's own teardown, or by hand after reading what is listed.
set -uo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 <ssh key> <user@host>" >&2; exit 2; }
key="$1"; shift
out="$(ssh -i "$key" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$1" 'bash -s' <<'REMOTE'
tmux ls 2>/dev/null | sed 's/^/tmux session: /'
for ns in $(ip netns list 2>/dev/null | awk '{print $1}'); do
  for pid in $(sudo -n ip netns pids "$ns" 2>/dev/null); do
    echo "netns $ns: $(ps -o pid=,etime=,args= -p "$pid" | cut -c1-160)"
  done
done
echo "check_idle: done"
REMOTE
)"
rc=$?
if [[ $rc -ne 0 || "$out" != *"check_idle: done"* ]]; then
  echo "check_idle: cannot inspect $1 (ssh rc=$rc)" >&2; exit 1
fi
left="$(printf '%s\n' "$out" | grep -v '^check_idle: done$')"
if [[ -n "$left" ]]; then
  echo "check_idle: $1 is not idle; leftover from an earlier run:" >&2
  printf '%s\n' "$left" | sed 's/^/  /' >&2
  exit 1
fi
