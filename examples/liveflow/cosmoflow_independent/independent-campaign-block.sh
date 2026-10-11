#!/bin/bash
# User approved n3 campaign after clean LF smoke, total48.5nh ceiling.
set -euo pipefail
OWN=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
block=${1:?}
DLIO_STATIC_W=${DLIO_STATIC_W:-12};DLIO_STATIC_P=${DLIO_STATIC_P:-2}
export DLIO_STATIC_W DLIO_STATIC_P
export DLIO_SCREEN_COMPUTE=0.00551 DLIO_CAMPAIGN_EPOCHS=5 DLIO_CAMPAIGN_STEPS=4096 DLIO_CONTROL_STEPS=512
# Identical passive reader observation in every timed arm, starting with block 1.
# The motivation lane supplies the opt-in implementation before any submission.
export LIVEFLOW_MOTIVATION_TIMERS=1
export LIVEFLOW_MOTIVATION_CAPTURE_STEPS=400
export LIVEFLOW_MOTIVATION_WARMUP_STEPS=20
export LIVEFLOW_MOTIVATION_GPU_TRACE=0
ROOT=${DLIO_CAMPAIGN_ROOT:?Set an absolute results directory outside the checkout}/b$block
[[ $ROOT == /* ]]
mkdir -p "$ROOT"
flux exec -r all hostname | sort -u > "$ROOT/hosts.txt"
mapfile -t hosts < "$ROOT/hosts.txt"
[[ ${#hosts[@]} == 16 ]]
case "$block" in
 1) arms=(original maxresource-adapted static-best liveflow);;
 2) arms=(maxresource-adapted static-best liveflow original);;
 3) arms=(static-best liveflow original maxresource-adapted);;
 *) exit 2;;
esac
printf '%s\n' "${arms[@]}" > "$ROOT/order.txt"
for arm in "${arms[@]}";do
 pids=()
 for host in "${hosts[@]}";do
  bash "$OWN/run-arm-independent-campaign.sh" "$ROOT/$arm/$host" "$arm" "$host" > "$ROOT/$arm-$host-launch.log" 2>&1 &
  pids+=("$!")
 done
 rc=0
 for pid in "${pids[@]}";do wait "$pid" || rc=1;done
 [[ $rc == 0 ]] || exit 1
 printf '%s complete\n' "$arm"
done
# Parallel static replays: at most three single-worker analyses per host,
# only after all timed application arms finish. LF facts were made natively.
pids=()
for host in "${hosts[@]}";do
 for arm in original maxresource-adapted static-best;do
  dir=$ROOT/$arm/$host
  flux run --requires="host:$host" -N1 -n1 -c1 -o cpu-affinity=off -o mpibind=off -o exit-timeout=none bash "$OWN/replay-independent.sh" "$dir" "$host" > "$dir/replay.log" 2>&1 &
  pids+=("$!")
 done
done
rc=0
for pid in "${pids[@]}";do wait "$pid" || rc=1;done
[[ $rc == 0 ]]
