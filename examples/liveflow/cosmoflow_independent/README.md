# Independent-host DLIO CosmoFlow cell

This cell runs eight MPI ranks on each host, six physical cores per rank. Each
host has an independent application communicator; the sixteen-host campaign
has no cross-host training barrier or controller agreement. CPU DLIO simulates
A100 computation at5.51ms, batch1. This is a storage/decode experiment, not an
actual A100 training or an official passing MLPerf Storage result.

MLPerf Storage v1.0 config source: mlcommons/storage commit
cd848ba808b2c627261dae8635f13593d3cceecc; DLIO submodule
bc693c62cf28f896caf014e9089d9a7bcba5ef6a. The compute setting follows that
CosmoFlow A100 config. Explicit reader/data deviations: PyTorch/HDF5 rather
than the pinned TensorFlow/TFRecord reader,32768 official HPC validation HDF5
records (int16,16MiB logical, gzip6) rather than524288 smaller TFRecords.
DLIO returns its existing synthetic payload after real input reading and
inflation; it has no model/loss. Advisory POSIX_FADV_DONTNEED of each rank's
file shard precedes each epoch; this does not prove physical cold-cache reads.

Timed arms: OriginalW4/P2, adapted MaxResourceW12/P696 (58prefetched samples per
worker), StaticW12/P2 and LiveFlow startingW4/P2. Adapted MaxResource applies
GlanceD's0.7 memory fraction to node /dev/shm capacity125.7GiB rather than RAM:
floor(0.7*shm_bytes/(8ranks*12workers*16MiB))=58perworker. The unadaptedW12/P1608
requires about201GiB and failed on this tmpfs; keep that finding. Other sample,
batch, compute and sampler parameters stay identical. File and index order
use seeded123+epoch permutations; saved consumed indices are checked exactly.

All arms set IBV_FORK_SAFE=1, RDMAV_FORK_SAFE=1 and
MV2_SUPPORT_FORK_SAFETY=1 before MPI initialization. Default worker context
remains fork until the clean forkserver option is measured and selected.
DLIO_WORKER_CONTEXT=forkserver starts a torch/h5py/numpy-preloaded clean server
before DLIO/MPI/DFTracer imports. Assertions require no native MPI/tracer state,
one native thread and zero open HDF5 files. Workers persist within an epoch;
the clean server persists across epoch retirement and LF resizes. Spawn is an
implemented diagnostic option but was rejected for startup/resize cost.

The optional motivation observer is enabled identically in every campaign arm:
LIVEFLOW_MOTIVATION_TIMERS=1, CAPTURE_STEPS=400, WARMUP_STEPS=20, GPU_TRACE=0 and
unique absolute OUTPUT_DIR=<arm-host-directory>/motivation-timers. It captures global
steps21–420, before LF's observed first moves. Fused HDF5 waits remain unresolved
Other. Post-move mechanism uses the normal analyzer, not these initial timers.
Observer overhead and native lifecycle must be qualified before release.

External pinned LiveFlow runtime:
/p/vast1/izzet/lfmr/liveflow-campaign-v4.0.6-agentperf-20261010,
analyzer68f7f57aa9a263e2f7fba936ba1a966fffb668e5. Its project cluster environment,
setup launcher, native SHM binaries and installed dependencies are reused.
The DLIO app, cell wrappers and decode analyzer adapter come from this repository,
not the older v3 overlay. Each arm checks a clean checkout and records HEAD,
source hashes, exact command, config and relevant environment before launching.

Host-scoped Flux jobs use exit-on-error and a5s exit timeout. An application
fatal-log monitor cancels only that host's nested job, records its failure and
retains partial traces. Other hosts finish the current arm. The block then stops;
paired-host exclusion is a possible disclosed fallback, not implemented policy.
No script here automatically submits the next block or authorizes a campaign.

Run only after the orchestrator releases the allocation and any storage hold:
set DLIO_CAMPAIGN_ROOT to an absolute output directory outside the clean checkout,
then submit independent-campaign-block.sh BLOCK using the project's Flux batch
allocation with16hosts/128tasks/6cores. Blocks rotate the four arm order and each
arm uses five4096-step epochs (20480steps/rank). LF targets10s with frozen K and
geometric worker raises; legal worker values1..12 avoid range coarsening.
Static replays run after all timed arms. Report scripts retain per-host elapsed,
AU, CPU/physical-core normalization, exact identities, normal analyzer mechanisms
and actual LF first-apply/verification trajectories. Paper claims require three
blocks with raw per-rep data and pooled95% intervals.

Qualification entry points (no automatic submission): the CPU payload/RNG,
signal-mask and fake-Flux fail-fast tests live under tests/.
`test-clean-forkserver.py` checks the clean preloaded server with the application
interpreter. `forkserver-test.sh` is the authorized one-node stress follow-up;
set DLIO_FORKSERVER_CAP_S and an absolute DLIO_FORKSERVER_TEST_ROOT outside this
checkout. It tests W12 timers off/on for256steps each, then native LF/timers-on
for4096steps, all eight ranks. This supplies an overhead diagnostic if both
matched short arms finish, plus startup/resize/sample identity observations.
A single off/on pair has no statistical overhead guarantee. The parent
time limit must keep total stress allocation under the approved one node-hour.
