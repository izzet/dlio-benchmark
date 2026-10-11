#!/bin/bash
# Adapted from the project's SHM dlio_tf/run_arm.sh; CPU PyTorch HDF5 on Corona.
set -euo pipefail
FREEZE=${DLIO_RUNTIME_ROOT:-/p/vast1/izzet/lfmr/liveflow-campaign-v4.0.6-agentperf-20261010}
TREE=$FREEZE/source
OWN=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$TREE/liveflow_shm/tests/cluster_env.sh"
source "$TREE/optimization/setup_env_corona.sh"
source "$FREEZE/runtime-cor.sh"
# Clean committed DLIO app and cell scripts; shared pipeline remains v4.0.6.
DLIO_APP_ROOT=$(realpath "$OWN/../../..");export DLIO_APP_ROOT
DLIO_COMMIT=$(git -C "$DLIO_APP_ROOT" rev-parse HEAD)
[[ -z $(git -C "$DLIO_APP_ROOT" status --porcelain --untracked-files=all) ]] || { echo "DLIO checkout must be clean";exit 1; }
export DLIO_COMMIT
export PYTHONPATH="$DLIO_APP_ROOT:$PYTHONPATH:$OWN"
export LIVEFLOW_AGENT_FACTS_ONLY=1 LIVEFLOW_ANALYZER_FASTPATH=1
export LIVEFLOW_WINDOW_TARGET_S=10 LIVEFLOW_SIZING=geometric
# All arms: enable verbs and MVAPICH2 fork safety before MPI initialization.
export IBV_FORK_SAFE=1 RDMAV_FORK_SAFE=1 MV2_SUPPORT_FORK_SAFETY=1
export LIVEFLOW_WORKLOAD=dlio-pytorch LIVEFLOW_DLIO_HDF5_DECODE=1 DFOPTIMIZER_REGISTER_COMPUTATION_THREADS=0
export DLIO_DECODE_AGENT_ENTRY="$OWN/dlio_decode_node_agent.py"
RUN_DIR=$(realpath -m -- "${1:?}");ARM=${2:?};HOST=${3:?};
hostrun() { flux run --requires="host:$HOST" -N1 -n1 -c1 -o cpu-affinity=off -o mpibind=off -o exit-timeout=none "$@"; }
mkdir -p "$RUN_DIR";export RUN_DIR
printf '%s\n' "$DLIO_COMMIT" > "$RUN_DIR/dlio-commit.txt"
git -C "$DLIO_APP_ROOT" log -1 --format='%H%n%s' > "$RUN_DIR/dlio-source.txt"
sha256sum "$DLIO_APP_ROOT/dlio_benchmark/main.py" "$DLIO_APP_ROOT/dlio_benchmark/data_loader/live_data_loader.py" "$DLIO_APP_ROOT/dlio_benchmark/data_loader/motivation_timers.py" "$DLIO_APP_ROOT/dlio_benchmark/reader/hdf5_reader.py" "$OWN"/*.py "$OWN"/*.sh > "$RUN_DIR/dlio-sources.sha256"
if [[ ${LIVEFLOW_MOTIVATION_TIMERS:-0} == 1 ]];then
 export LIVEFLOW_MOTIVATION_OUTPUT_DIR="$RUN_DIR/motivation-timers"
 [[ -f "$DLIO_APP_ROOT/dlio_benchmark/data_loader/motivation_timers.py" ]] || { echo 'Committed DLIO motivation timers missing';exit 1; }
fi
DATA=/p/lustre3/izzet/dlio-benchmark-test/cosmoflow_official_validation_32768_20261010/data
prefix="dlio_smoke_${ARM}_$(date +%s)_$$"
export LIVEFLOW_SHM_PREFIX=$prefix DFTRACER_SHM_PREFIX=$prefix
export LIVEFLOW_SHM_LANES=2048 DFTRACER_SHM_LANES=2048
export LIVEFLOW_SHM_PYTHONPATH="$TREE/liveflow_shm/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export DFTRACER_WRITER_TYPE=SHM DFTRACER_INIT=FUNCTION DFTRACER_ENABLE=1
export DFTRACER_INC_METADATA=1 DFTRACER_DATA_DIR=$DATA DFTRACER_LOG_FILE="$RUN_DIR/dft"
export DFTRACER_SHM_CONTROL_EVENT_NAMES=window.start,window.stop
export DFTRACER_USE__EXIT_AFTER_FINALIZE=1 DFTRACER_TRACE_COMPRESSION=0
export LIVEFLOW_CTX_PER_RANK=1 LIVEFLOW_JOB_LEVEL=0 LIVEFLOW_JOB_INPUT_SYNC=0
export DLIO_WINDOW_EVERY_N=${DLIO_CONTROL_STEPS:-512} DLIO_KNOB_MIN_THREADS=1 DLIO_KNOB_MAX_THREADS=12
export DFOPTIMIZER_ALLOWED_KNOBS=dlio.read_threads
export DFOPTIMIZER_BOOTSTRAP_READ_THREADS=4 DFOPTIMIZER_BOOTSTRAP_PREFETCH_SIZE=2
export DFOPTIMIZER_EXPECTED_LOCAL_RANKS=8 DFOPTIMIZER_BOOTSTRAP_DLIO=0
export LIVEFLOW_REQUIRE_PLAN=0 LIVEFLOW_AGENT_DEBUG=1
export LIVEFLOW_SNAPSHOT_PROOF_FILE="$RUN_DIR/snapshot-import-proof"
export LIVEFLOW_PERSIST_TRACES="$RUN_DIR/traces" LIVEFLOW_PERSIST_TRACE_QUEUE_RECORDS=4096
export LIVEFLOW_EOS_PUBLISHER_RECEIPT_DIR="$RUN_DIR/eos-publisher-receipt"
case "$ARM" in
 original|maxresource-adapted|static-best) export DFOPTIMIZER_ENABLE=0;stages=raw-drain ;;
 liveflow) export DFOPTIMIZER_ENABLE=1;stages=analyzer,diagnoser,optimizer ;;
 *) exit 2 ;;
esac
printf 'arm=%s\nprefix=%s\nruntime=%s\ndlio_commit=%s\nanalysis=static raw-only offline analysis; LiveFlow native online node-local agent\ncache=advisory cold own rank shard before each epoch; first20 steps per epoch excluded; campaign\n' "$ARM" "$prefix" "$FREEZE" "$DLIO_COMMIT" > "$RUN_DIR/config.txt"
# Record relevant task settings, not login environment/secrets.
env | sort | rg '^(LIVEFLOW_|DFTRACER_|DFOPTIMIZER_|DLIO_|IBV_FORK_SAFE=|RDMAV_FORK_SAFE=|MV2_SUPPORT_FORK_SAFETY=|OMP_|MKL_|OPENBLAS_|PYTHONPATH=|LD_LIBRARY_PATH=|PYTHONFAULTHANDLER=)' > "$RUN_DIR/runtime-env.txt"
appy="$VENV_DLIO/bin/python";agentpy="$LIVEFLOW_AGENT_VENV_DIR/bin/python"
# LF pays for two dedicated physical agent cores. Comparators retain shipped masks.
if [[ "$ARM" == liveflow ]];then
 export DLIO_LF_ISOLATE_CORES=2
 LIVEFLOW_AGENT_CPUSET=$(hostrun "$agentpy" "$OWN/dlio_cpu_affinity.py" --output "$RUN_DIR/core-isolation.json")
 export LIVEFLOW_AGENT_CPUSET
 [[ -n "$LIVEFLOW_AGENT_CPUSET" ]]
else
 unset DLIO_LF_ISOLATE_CORES LIVEFLOW_AGENT_CPUSET
fi
# Record derived affinity variables after host-specific topology discovery.
env | sort | rg '^(DLIO_LF_ISOLATE_CORES=|LIVEFLOW_AGENT_CPUSET=)' >> "$RUN_DIR/runtime-env.txt" || true
agent_pid=''
if [[ -n "$stages" ]];then
 hostrun "$agentpy" -c 'import os;from shm_reader import ShmRingReader;p=os.environ["LIVEFLOW_SHM_PREFIX"];ShmRingReader(p+".trace",lane_count=2048,lane_bytes=8388608,create=True).close();ShmRingReader(p+".control",lane_count=2048,lane_bytes=65536,create=True).close()' > "$RUN_DIR/shm-create.log" 2>&1
 if [[ "$stages" == raw-drain ]];then
  agentcmd=("$agentpy" -u "$OWN/raw-drain.py");ready_text='[raw-drain] ready host='
 else
  agentcmd=("$agentpy" -u "$DLIO_DECODE_AGENT_ENTRY" --stages "$stages" --preset liveflow-v1 --workers 1 --num-ranks 8 --data-folder "$DATA" --hydra-dir "$RUN_DIR/agent" --host-suffix --timeout "${DLIO_AGENT_TIMEOUT:-2700}" --drain-grace-ms 4000 --debug);ready_text='[agent] ready stages='
 fi
 python3 "$OWN/run-host-job.py" "$RUN_DIR" "$HOST" agent -- "${agentcmd[@]}" > "$RUN_DIR/agents.log" 2>&1 &
 agent_pid=$!
 ready=0
 for ((i=0;i<90;i++));do
  if [[ $(rg -F -c "$ready_text" "$RUN_DIR/agents.log" || true) == 1 ]];then ready=1;break;fi
  kill -0 "$agent_pid" 2>/dev/null || break
  sleep 1
 done
 [[ $ready == 1 ]] || { echo 'Agent readiness failed';exit 1; }
fi
sampler_pid=''
if [[ ${DLIO_CRASH_FORENSICS:-0} == 1 ]];then
 [[ "$ARM" == liveflow && ${PYTHONFAULTHANDLER:-0} == 1 ]]
 hostrun "$agentpy" -u "$OWN/host-memory-sampler.py" "$RUN_DIR" > "$RUN_DIR/host-memory.jsonl" 2> "$RUN_DIR/host-memory-sampler.err" &
 sampler_pid=$!
 trap 'touch "$RUN_DIR/host-memory.stop"; wait "$sampler_pid" || true' EXIT
fi
arm_w=4;arm_p=2
case "$ARM" in maxresource) arm_w=12;arm_p=1608;;maxresource-adapted) arm_w=12;arm_p=696;;static6) arm_w=6;;static8) arm_w=8;;static-best) arm_w=${DLIO_STATIC_W:?};arm_p=${DLIO_STATIC_P:?};;esac
printf 'nodes=%s\nranks=%s\ncompute_s=%s\nworkers=%s\nprefetch_total=%s\nsteady_excluded_per_epoch=20\ncontroller=arm-dependent\nloader=fixed LiveDataLoader for stall telemetry (stock TorchDataLoader lacks emit_window_stall)\n' 1 8 "$DLIO_SCREEN_COMPUTE" "$arm_w" "$arm_p" >> "$RUN_DIR/config.txt"
printf 'epochs=%s\nsteps_per_epoch_per_rank=%s\ncontrol_steps=%s\n' "${DLIO_CAMPAIGN_EPOCHS:-5}" "${DLIO_CAMPAIGN_STEPS:-4096}" "$DLIO_WINDOW_EVERY_N" >> "$RUN_DIR/config.txt"
cmd=("$appy" -u "$OWN/rank-independent-campaign.py" workload=cosmoflow_v100
 ++workload.framework=pytorch ++workload.reader.data_loader=pytorch
 ++workload.reader.data_loader_sampler=index ++workload.reader.multiprocessing_context="${DLIO_WORKER_CONTEXT:-fork}"
 ++workload.workflow.generate_data=False ++workload.workflow.train=True ++workload.workflow.checkpoint=False
 ++workload.dataset.format=hdf5 ++workload.dataset.data_folder="$DATA"
 ++workload.dataset.num_files_train=32768 ++workload.dataset.num_files_eval=0 ++workload.dataset.num_samples_per_file=1
 ++workload.dataset.record_length_bytes=16777216 ++workload.dataset.record_length_bytes_stdev=0
 ++workload.reader.read_threads="$arm_w" ++workload.reader.prefetch_size="$arm_p" ++workload.reader.batch_size=1
 ++workload.train.epochs="${DLIO_CAMPAIGN_EPOCHS:-5}" ++workload.train.total_training_steps="${DLIO_CAMPAIGN_STEPS:-4096}"
 ++workload.reader.file_shuffle=seed ++workload.reader.sample_shuffle=seed ++workload.reader.shuffle_size=2 ++workload.metric.au=0.70 ++workload.train.computation_time="${DLIO_SCREEN_COMPUTE:?}"
 ++workload.output.folder="$RUN_DIR/dlio" hydra.run.dir="$RUN_DIR/hydra")
if [[ "$ARM" != original-stock ]];then
 cmd+=(++workload.reader.data_loader_classname=dlio_benchmark.data_loader.live_data_loader.LiveDataLoader)
fi
printf '%q ' "${cmd[@]}" > "$RUN_DIR/command.txt";printf '\n' >> "$RUN_DIR/command.txt"
set +e
app_start_s=$(date +%s.%N)
python3 "$OWN/run-host-job.py" "$RUN_DIR" "$HOST" app -- "${cmd[@]}" > "$RUN_DIR/dlio.log" 2>&1
app_rc=$?
app_end_s=$(date +%s.%N)
printf 'start_s=%s\nend_s=%s\n' "$app_start_s" "$app_end_s" > "$RUN_DIR/app-elapsed.txt"
echo "$app_rc" > "$RUN_DIR/app.rc"
agent_rc=0;eos_rc=0;post_rc=0
if [[ -n "$stages" ]];then
 if [[ $app_rc == 0 ]];then
  hostrun "$agentpy" "$TREE/liveflow_shm/python/eos_publisher.py" > "$RUN_DIR/eos.log" 2>&1;eos_rc=$?
  [[ "$stages" != raw-drain ]] || touch "$RUN_DIR/raw-drain.stop"
 else
  # Failed ranks cannot supply complete EOS. Abort only this host's collector.
  echo 'EOS omitted: application failed; traces retained as partial' > "$RUN_DIR/eos.log"
  eos_rc=1
  [[ "$stages" != raw-drain ]] || touch "$RUN_DIR/raw-drain.stop"
  if [[ -s "$RUN_DIR/agent-jobid.txt" ]];then
   flux cancel "$(cat "$RUN_DIR/agent-jobid.txt")" > "$RUN_DIR/agent-cancel.log" 2>&1
  fi
 fi
 wait "$agent_pid";agent_rc=$?
 hostrun "$agentpy" "$TREE/liveflow_shm/python/tuolumne_postflight.py" --prefix "$prefix" > "$RUN_DIR/postflight.jsonl" 2>&1;post_rc=$?
fi
hostrun bash -c 'prefix=$1; for p in /dev/shm/"$prefix".*;do [[ -e "$p" ]] && unlink "$p";done;exit 0' _ "$prefix" > "$RUN_DIR/cleanup.log" 2>&1
cleanup_rc=$?
printf 'app_rc=%s\nagent_rc=%s\neos_rc=%s\npostflight_rc=%s\ncleanup_rc=%s\n' "$app_rc" "$agent_rc" "$eos_rc" "$post_rc" "$cleanup_rc" > "$RUN_DIR/status.txt"
[[ $app_rc == 0 && $agent_rc == 0 && $eos_rc == 0 && $post_rc == 0 && $cleanup_rc == 0 ]]
