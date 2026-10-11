#!/bin/bash
# One diagnostic arm; campaign PAUSE is independent and must remain set.
set -euo pipefail
OWN=${DLIO_CAMPAIGN_CHECKOUT:?}/examples/liveflow/cosmoflow_independent
ROOT=${DLIO_PILOT_ROOT:?}
[[ $ROOT == /* && -d $OWN && -f ${DLIO_RUNTIME_ROOT:?}/runtime-cor.sh ]]
export DLIO_STATIC_W=12 DLIO_STATIC_P=2
export DLIO_SCREEN_COMPUTE=0.00551 DLIO_CAMPAIGN_EPOCHS=5 DLIO_CAMPAIGN_STEPS=4096 DLIO_CONTROL_STEPS=512
export LIVEFLOW_MOTIVATION_TIMERS=1 LIVEFLOW_MOTIVATION_CAPTURE_STEPS=400
export LIVEFLOW_MOTIVATION_WARMUP_STEPS=20 LIVEFLOW_MOTIVATION_GPU_TRACE=0
export DLIO_EXCLUDE_CRASHED_HOSTS=1 DLIO_MIN_PAIRED_HOSTS=14
mkdir -p "$ROOT"
flux exec -r all hostname | sort -u > "$ROOT/hosts.txt"
mapfile -t hosts < "$ROOT/hosts.txt"
[[ ${#hosts[@]} == 16 ]]
for host in "${hosts[@]}"; do [[ $host != corona265 ]]; done
cp "$ROOT/hosts.txt" "$ROOT/paired-hosts.txt"
printf 'liveflow\n' > "$ROOT/order.txt"
pids=()
for host in "${hosts[@]}"; do
    bash "$OWN/run-arm-independent-campaign.sh" "$ROOT/liveflow/$host" liveflow "$host" > "$ROOT/liveflow-$host-launch.log" 2>&1 &
    pids+=("$!")
done
rc=0
for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
        python3 "$OWN/paired-hosts.py" "$ROOT" liveflow "${hosts[$i]}" --minimum "$DLIO_MIN_PAIRED_HOSTS" >> "$ROOT/pair-exclusion.log" 2>&1 || rc=1
    fi
done
printf '%s\n' "$rc" > "$ROOT/pilot.rc"
exit "$rc"
