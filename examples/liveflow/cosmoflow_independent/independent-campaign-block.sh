#!/bin/bash
# User approved n3 campaign after clean LF smoke, total48.5nh ceiling.
set -euo pipefail
OWN=${DLIO_CAMPAIGN_CHECKOUT:?Set the pinned DLIO checkout}/examples/liveflow/cosmoflow_independent
[[ -d $OWN ]]
case ${DLIO_DEFER_REPLAY:-0} in 0|1) ;; *) echo 'DLIO_DEFER_REPLAY must be 0 or 1' >&2;exit 2;;esac
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
cp "$ROOT/hosts.txt" "$ROOT/paired-hosts.txt"
export DLIO_EXCLUDE_CRASHED_HOSTS=${DLIO_EXCLUDE_CRASHED_HOSTS:-1}
export DLIO_MIN_PAIRED_HOSTS=${DLIO_MIN_PAIRED_HOSTS:-14}
case "$block" in
 1) arms=(original maxresource-adapted static-best liveflow);;
 2) arms=(maxresource-adapted static-best liveflow original);;
 3) arms=(static-best liveflow original maxresource-adapted);;
 *) exit 2;;
esac
printf '%s\n' "${arms[@]}" > "$ROOT/order.txt"
for arm in "${arms[@]}";do
 if [[ -n ${DLIO_POOL_CONFIG:-} ]];then
  mapfile -t selected < <(python3 "$OWN/pool-selection.py" "$DLIO_POOL_CONFIG" "$block" "$arm")
  [[ ${#selected[@]} == 2 ]] || { echo 'Invalid campaign data pool selection' >&2;exit 1; }
  export DLIO_DATA_POOL=${selected[0]} DLIO_DATA_ROOT=${selected[1]}
 fi
 pids=();mapfile -t paired < "$ROOT/paired-hosts.txt"
 for host in "${paired[@]}";do
  bash "$OWN/run-arm-independent-campaign.sh" "$ROOT/$arm/$host" "$arm" "$host" > "$ROOT/$arm-$host-launch.log" 2>&1 &
  pids+=("$!")
 done
 rc=0
 for i in "${!pids[@]}";do
  if ! wait "${pids[$i]}";then
   if [[ $DLIO_EXCLUDE_CRASHED_HOSTS == 1 ]];then
    python3 "$OWN/paired-hosts.py" "$ROOT" "$arm" "${paired[$i]}" --minimum "$DLIO_MIN_PAIRED_HOSTS" >> "$ROOT/pair-exclusion.log" 2>&1 || rc=1
   else rc=1;fi
  fi
 done
 [[ $rc == 0 ]] || exit 1
 printf '%s complete\n' "$arm"
done
# Application wrappers have already finished EOS, drain, postflight and cleanup.
# Retain the same raw traces for replay with this checkout/runtime after release.
if [[ ${DLIO_DEFER_REPLAY:-0} == 1 ]];then
 printf 'checkout=%s\nruntime=%s\narms=original maxresource-adapted static-best\n' \
  "$DLIO_CAMPAIGN_CHECKOUT" "${DLIO_RUNTIME_ROOT:?}" > "$ROOT/replay-deferred.txt"
 exit 0
fi
# Parallel static replays: at most three single-worker analyses per host,
# only after all timed application arms finish. LF facts were made natively.
pids=()
mapfile -t paired < "$ROOT/paired-hosts.txt"
for host in "${paired[@]}";do
 for arm in original maxresource-adapted static-best;do
  dir=$ROOT/$arm/$host
  flux run --requires="host:$host" -N1 -n1 -c1 -o cpu-affinity=off -o mpibind=off -o exit-timeout=none bash "$OWN/replay-independent.sh" "$dir" "$host" > "$dir/replay.log" 2>&1 &
  pids+=("$!")
 done
done
rc=0
for pid in "${pids[@]}";do wait "$pid" || rc=1;done
[[ $rc == 0 ]]
