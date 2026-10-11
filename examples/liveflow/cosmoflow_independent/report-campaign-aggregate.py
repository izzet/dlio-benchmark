#!/usr/bin/env python3
import json,math,pathlib,statistics,sys
root=pathlib.Path(sys.argv[1]);blocks=[json.load((root/f'b{i}'/'block-results.json').open()) for i in [1,2,3]]
results={}
for arm in ['maxresource-adapted','static-best','liveflow']:
 ratios=[b['aggregate']['original']['mean_host_elapsed_s']/b['aggregate'][arm]['mean_host_elapsed_s'] for b in blocks]
 logs=list(map(math.log,ratios));mu=statistics.mean(logs);half=4.302652729911275*statistics.stdev(logs)/math.sqrt(3)
 results[arm]=dict(per_block_elapsed_speedup=ratios,geometric_mean=math.exp(mu),ci95=[math.exp(mu-half),math.exp(mu+half)],n=3,method='paired block ratios, log-space Student-t df2; hosts not reps')
output=dict(paired_hosts_per_block=[b.get('paired_hosts') for b in blocks],excluded_hosts_per_block=[b.get('excluded_hosts',[]) for b in blocks],results=results,per_block_aggregate=[b['aggregate'] for b in blocks],scope='independent-host A1005.51ms B1 adapted HDF5/PyTorch cell; official MLPerf format/size deviations recorded',source='b1,b2,b3/block-results.json; per-host and per-rank CSV, commands/config and raw traces retained')
(root/'campaign-results.json').write_text(json.dumps(output,indent=2)+'\n')
lines=['# Independent-host A100 paired campaign','', 'n=3 paired blocks; primary elapsed is mean of eligible paired independent host jobs (14–16 per block); crashes exclude that host from every arm, with raw data retained. Hosts are not repetitions. Pooled geometric mean speedup and95% Student-t CI in log space(df2).','', '| Arm vs Original | b1 | b2 | b3 | Geomean |95% CI|','|---|---:|---:|---:|---:|---|']
for arm,r in results.items():
 x=r['per_block_elapsed_speedup'];lo,hi=r['ci95'];lines.append(f"|{arm}|{x[0]:.4f}|{x[1]:.4f}|{x[2]:.4f}|{r['geometric_mean']:.4f}|[{lo:.4f},{hi:.4f}]|")
lines+=['','Per-block mechanism/AU/CPU/read/cadence and actual LiveFlow application/confirmation steps: b1/RESULTS.md, b2/RESULTS.md, b3/RESULTS.md and their raw block-results.json/per-host.csv/cpu-per-rank.csv/lf-trajectory.json. All raw paths retain commands/config/runtime version.','', 'Retained official HDF5 validation data, PyTorch reader, dataset sizing, seeded shuffle adaptation, simulated computation and advisory cold eviction remain explicit deviations from the pinned MLPerf Storage v1.0 TFRecord cell. No model/loss in DLIO. MaxResource adapts its memory fraction to binding node SHM capacity as approved. Original crashed in initial b1 before fork-safety correction: ../independent-campaign/b1/FAILURE.md. Retried campaign sets IBV_FORK_SAFE=1,RDMAV_FORK_SAFE=1,MV2_SUPPORT_FORK_SAFETY=1 equally in every arm, preserving fork start method and samples. Native invalid/missing windows provide no facts; report measured coverage. Do not claim official AU compliance or numerical model convergence.']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(results,indent=2))
