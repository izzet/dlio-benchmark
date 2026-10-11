#!/bin/bash
# One diagnostic H100 static screen, full shipped app masks in every arm.
set -euo pipefail
OWN=${DLIO_CAMPAIGN_CHECKOUT:?}/examples/liveflow/cosmoflow_independent
ROOT=${DLIO_H100_SCREEN_ROOT:?}
[[ $ROOT == /* && -d $OWN && -f ${DLIO_RUNTIME_ROOT:?}/runtime-cor.sh ]]
export DLIO_SCREEN_COMPUTE=0.00350 DLIO_CAMPAIGN_EPOCHS=5 DLIO_CAMPAIGN_STEPS=512 DLIO_CONTROL_STEPS=512
export LIVEFLOW_MOTIVATION_TIMERS=1 LIVEFLOW_MOTIVATION_CAPTURE_STEPS=400 LIVEFLOW_MOTIVATION_WARMUP_STEPS=20 LIVEFLOW_MOTIVATION_GPU_TRACE=0
export DLIO_LEAN_WORKER_RECORDS=1 DLIO_EXCLUDE_CRASHED_HOSTS=1 DLIO_MIN_PAIRED_HOSTS=14
unset DLIO_LF_ISOLATE_CORES LIVEFLOW_AGENT_CPUSET
mkdir -p "$ROOT"
flux exec -r all hostname | sort -u > "$ROOT/hosts.txt"
mapfile -t hosts < "$ROOT/hosts.txt"
[[ ${#hosts[@]} == 16 ]]
for host in "${hosts[@]}";do [[ $host != corona265 ]];done
cp "$ROOT/hosts.txt" "$ROOT/paired-hosts.txt"
printf 'original\nstatic-w12\nstatic-w24\n' > "$ROOT/order.txt"
# Admission is capacity based; report free space too, not a physical-cache claim.
for host in "${hosts[@]}";do
 flux run --requires="host:$host" -N1 -n1 -c1 -o cpu-affinity=off -o mpibind=off python3 -c '
import json,math,os,sys
s=os.statvfs("/dev/shm");capacity=s.f_blocks*s.f_frsize;free=s.f_bavail*s.f_frsize
rows=[]
for W in (4,12,24):
 limit=math.floor(.7*capacity/(8*W*16777216));factor=math.ceil(2/W)
 queue=8*W*factor*16777216
 assert factor<=limit and queue<free,(W,limit,free)
 rows.append(dict(workers=W,prefetch_total=2,effective_prefetch_per_worker=factor,shm_rule_factor_cap=limit,shm_rule_total_prefetch_cap=W*limit,conservative_queue_bytes=queue))
print(json.dumps(dict(capacity_bytes=capacity,free_bytes=free,sample_bytes=16777216,ranks_per_host=8,fraction=.7,arms=rows)))
' > "$ROOT/shm-$host.json"
done
for label in original static-w12 static-w24;do
 mapfile -t paired < "$ROOT/paired-hosts.txt";pids=()
 W=4;arm=original
 if [[ $label != original ]];then arm=static-best;W=${label#static-w};fi
 export DLIO_STATIC_W=$W DLIO_STATIC_P=2
 for host in "${paired[@]}";do
  bash "$OWN/run-arm-independent-campaign.sh" "$ROOT/$label/$host" "$arm" "$host" > "$ROOT/$label-$host-launch.log" 2>&1 &
  pids+=("$!")
 done
 rc=0
 for i in "${!pids[@]}";do
  if ! wait "${pids[$i]}";then
   python3 "$OWN/paired-hosts.py" "$ROOT" "$label" "${paired[$i]}" --minimum 14 >> "$ROOT/pair-exclusion.log" 2>&1 || rc=1
  fi
 done
 [[ $rc == 0 ]] || exit 1
 printf '%s complete\n' "$label"
done
# Replay only after every timed arm. Serial perhost avoids three concurrent frames.
pids=();mapfile -t paired < "$ROOT/paired-hosts.txt"
for host in "${paired[@]}";do
 (
  for label in original static-w12 static-w24;do
   dir=$ROOT/$label/$host
   flux run --requires="host:$host" -N1 -n1 -c1 -o cpu-affinity=off -o mpibind=off bash "$OWN/replay-independent.sh" "$dir" "$host" > "$dir/replay.log" 2>&1
  done
 ) &
 pids+=("$!")
done
rc=0;for pid in "${pids[@]}";do wait "$pid" || rc=1;done
printf '%s\n' "$rc" > "$ROOT/screen.rc"
exit "$rc"
