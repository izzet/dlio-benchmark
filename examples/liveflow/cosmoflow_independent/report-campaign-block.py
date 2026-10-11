#!/usr/bin/env python3
import csv,datetime,json,pathlib,statistics,sys
root=pathlib.Path(sys.argv[1]);nhosts=2 if len(sys.argv)>2 and sys.argv[2]=='smoke' else 16;narms=1 if nhosts==2 else 4;totalsteps=4096 if nhosts==2 else 20480;d=json.load((root/'block-results.json').open());nhosts=len(d['paired_hosts']) if d.get('paired_hosts') is not None else nhosts;trajectory=json.load((root/'lf-trajectory.json').open());rows=[]
assert len(d['per_host'])==nhosts*narms,('incomplete successful host-arm results',len(d['per_host']))
for key,v in d['per_host'].items():
 arm,host=key.split('/');p=root/arm/host
 post=json.loads((p/'postflight.jsonl').read_text().splitlines()[-1])
 for ring,r in post['rings'].items():
  assert all(r.get(k,0)==0 for k in ['dropped','backlog_bytes','unclaimed']),(key,ring,r)
 assert v['analyzer_complete'],key
 steps=[json.loads(x) for x in (p/'screen-metrics/steps-rank-0.jsonl').open()]
 assert len(steps)==totalsteps,key
 windows=[]
 for f in sorted(p.glob('agent_*/analysis_facts_window_*.json')):
  facts=json.load(f.open()).get('facts',[])
  m=next((x['evidence']['metrics'] for x in facts if 'lf_v1_decode_wait' in x.get('evidence',{}).get('metrics',{})),{})
  if not m:continue
  flat=f.with_name(f.name.replace('analysis_facts_window_','flat_view_window_window_').replace('.json','.csv'))
  r=next(csv.DictReader(flat.open()))
  boundary=json.loads(r['__liveflow_boundary_ts_ns_by_pid']);end_s=statistics.median(boundary.values())/1e9
  last=min(range(len(steps)),key=lambda i:abs(steps[i]['unix_s']-end_s))+1
  n=int(m.get('step_nunique',0))
  if n and last>=n:
   xs=steps[last-n:last];windows.append(dict(window=f.stem,steps=n,completed_steps=last,boundary_time_s=end_s,join_error_seconds=abs(steps[last-1]['unix_s']-end_s),seconds=xs[-1]['monotonic']-xs[0]['monotonic']+xs[0]['seconds'],actual_workers=sorted({x['workers'] for x in xs})))
 v['realized_valid_native_windows']=windows
 v['host']=host;v['arm']=arm
 if arm=='liveflow':
  t=trajectory[host];change=t['worker_changes']['0'];v['first_apply_step']=change[1]['completed_step'] if len(change)>1 else None
  confirmed=[x for x in t['verification_events'] if x.get('event')=='optimizer.cbandit.move_confirmed']
  final_workers=steps[-1]['workers'];matching=[x for x in confirmed if x.get('value')==final_workers]
  c=matching[-1] if matching else None
  v['final_workers']=final_workers;v['last_confirmation']=c
  v['convergence_step']=c['actual_completed_step_at_decision'] if c and c['decision_seconds_after_last_retained_step']<0 else None
 full=[w['seconds'] for w in windows if w['steps']>20]
 v['median_noncalibration_window_seconds']=statistics.median(full) if full else None
 rows.append(dict(arm=arm,host=host,elapsed_s=v['app_elapsed_s'],steady_step_ms=v['steady_critical_step_ms'],au_percent=v['nominal_au_percent'],decode_wait_ms=v['steady_decode_wait_ms'],storage_wait_ms=v['steady_storage_wait_ms'],read_GB_s=v['aggregate_read_GB_s'],cpu_percent_6_cores=100*v['cpu_mean_fraction_6_cores'],noncalibration_window_median_seconds=v['median_noncalibration_window_seconds'],first_apply_step=v.get('first_apply_step'),convergence_step=v.get('convergence_step'),final_workers=v.get('final_workers')))
 for r in v['cpu_per_rank']:r['host']=host;r['arm']=arm
with (root/'per-host.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
cpus=[r for v in d['per_host'].values() for r in v['cpu_per_rank']]
with (root/'cpu-per-rank.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(cpus[0]));w.writeheader();w.writerows(cpus)
for arm,a in d['aggregate'].items():
 assert a['hosts']==nhosts,(arm,a['hosts'])
 vals=[v for v in d['per_host'].values() if v['arm']==arm]
 window_seconds=[v['median_noncalibration_window_seconds'] for v in vals if v['median_noncalibration_window_seconds'] is not None]
 a['median_host_noncalibration_window_seconds']=statistics.median(window_seconds) if window_seconds else None
 if arm=='liveflow':
  a['hosts_with_first_apply']=sum(v['first_apply_step'] is not None for v in vals)
  a['first_apply_steps']={v['host']:v['first_apply_step'] for v in vals}
  a['convergence_steps']={v['host']:v['convergence_step'] for v in vals}
  a['hosts_with_confirmed_final_configuration_before_end']=sum(v['convergence_step'] is not None for v in vals)
  a['final_workers']={v['host']:v['final_workers'] for v in vals}
(root/'block-results.json').write_text(json.dumps(d,indent=2)+'\n')
excluded=d.get('excluded_hosts',[])
lines=[f'# Independent-host A100 campaign {root.name}', '', f'Eligible paired hosts: {nhosts}; excluded across all arms: {[x["host"] for x in excluded]}. All excluded raw data retained. One diagnostic block: hosts are not campaign repetitions. Raw per-host results: per-host.csv; CPU: cpu-per-rank.csv; actual native noncalibration windows (tails retained) and LF trajectory in block-results.json and lf-trajectory.json.','', '| Arm | Host-mean elapsed(s) | Steady step(ms) | AU(%) | Decode/storage wait(ms) | Read(GB/s) | CPU(% of6cores/rank) | Median host window(s) |','|---|---:|---:|---:|---|---:|---:|---:|']
for arm,a in d['aggregate'].items():
 fmt=lambda value:'NA' if value is None else f'{value:.3f}'
 lines.append(f"|{arm}|{fmt(a['mean_host_elapsed_s'])}|{fmt(a['mean_host_step_ms'])}|{fmt(a['mean_host_au_percent'])}|{fmt(a['mean_host_steady_decode_wait_ms'])}/{fmt(a['mean_host_steady_storage_wait_ms'])}|{fmt(a['sum_host_read_GB_s'])}|{100*a['mean_cpu_fraction_6_cores']:.2f}|{fmt(a['median_host_noncalibration_window_seconds'])}|")
lines+=['','Missing valid wait/window measurements are NA, never zero. Completed analysis does not imply valid wait coverage.']
for arm,a in d['aggregate'].items():
 lines.append(f"{arm}: mean valid step bins/host={a.get('analyzer_valid_step_bins_mean_host', 'NA')}; steady-window bins/host={a.get('analyzer_steady_window_step_bins_mean_host', 'NA')}.")
a=d['aggregate']['liveflow'];lines+=['',f"LiveFlow first-apply steps: {a['first_apply_steps']}",f"LiveFlow confirmation steps at publication (of{totalsteps}): {a['convergence_steps']}",f"Final workers: {a['final_workers']}",'',f'All{nhosts*narms} host-arm statuses0,{nhosts*narms*8} exact rank identities, all final SHM drops/backlog/unclaimed0. Wait coverage includes valid native windows only; detailed bin coverage retained. Decode/storage are exposed per-step waits, not service times; common MPI barrier is node-local. Read uses full application wall. AU uses nominal5.51ms numerator; thresholds70/90 reported. Compute is simulated CPU sleep; no model/loss. Cold eviction advisory, shared cache; retained HDF5/PyTorch format and dataset sizing are explicit MLPerf deviations. MaxResource uses user-adapted binding-SHM formula.']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:11]))
