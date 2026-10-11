#!/usr/bin/env python3
"""Join retained planner evidence to actual rank step timing and worker counts."""
import collections,datetime,json,pathlib,statistics,sys
root=pathlib.Path(sys.argv[1]);out={}
for host in sorted(root.glob('liveflow/*')):
 paths=list(host.glob('agent_*/dfdiagnoser_agent/dfdiagnoser.log'))
 if not paths:continue
 ranks={}
 for p in (host/'screen-metrics').glob('steps-rank-*.jsonl'):
  ranks[int(p.stem.rsplit('-',1)[1])]=[json.loads(l) for l in p.open()]
 if 0 not in ranks:continue
 steps=ranks[0];records=[];decisions=[];verification=[];grids=[]
 for line in paths[0].open():
  try:d=json.loads(line)
  except:continue
  event=d.get('event','')
  if event=='optimizer.cbandit.registered':grids.append(d.get('levels'))
  if event=='optimizer.cbandit.window_inputs':
   m=d['metrics'];last=int(m['completed_steps']);n=int(m['step_nunique']);first=last-n
   selected=steps[first:last]
   wall=selected[-1]['monotonic']-selected[0]['monotonic']+selected[0]['seconds'] if len(selected)==n else None
   records.append(dict(window=d['window'],completed_steps=last,step_bins=n,configured_steps=m.get('configured_window_steps'),actual_wall_s=wall,physical_workers=m.get('physical_read_threads'),physical_applied_step=m.get('physical_applied_step'),actual_workers=sorted({x['workers'] for x in selected}),timestamp=d.get('timestamp')))
  if event=='optimizer.cbandit.decision':decisions.append({k:d.get(k) for k in ['window','decision','current','chose','accept','timestamp']})
  if 'verif' in event or 'revert' in event or 'move_confirmed' in event:verification.append(d)
 for v in verification:
  stamp=datetime.datetime.fromisoformat(v['timestamp'].replace('Z','+00:00')).timestamp()
  v['actual_completed_step_at_decision']=max([i+1 for i,x in enumerate(steps) if x['unix_s']<=stamp],default=0)
  v['decision_seconds_after_last_retained_step']=stamp-steps[-1]['unix_s']
 changes={}
 for rank,xs in ranks.items():
  changes[str(rank)]=[dict(completed_step=i+1,epoch=x['epoch'],step=x['step'],workers=x['workers'],unix_s=x['unix_s']) for i,x in enumerate(xs) if i==0 or x['workers']!=xs[i-1]['workers']]
 full_by_workers={str(w):[r['actual_wall_s'] for r in records if r['step_bins']==r['configured_steps'] and r['actual_wall_s'] is not None and r['actual_workers']==[w]] for w in [4,8,12]}
 full=[r['actual_wall_s'] for r in records if r['step_bins']==r['configured_steps'] and r['actual_wall_s'] is not None]
 out[host.name]=dict(actual_steps_per_rank={str(r):len(x) for r,x in ranks.items()},registered_grid=grids[-1] if grids else None,worker_changes=changes,windows=records,full_window_median_wall_s=statistics.median(full) if full else None,full_windows_wall_s_by_actual_workers=full_by_workers,decisions=decisions,verification_events=verification,convergence='unproven unless recorded per-step verification supports final applied configuration')
(root/'lf-trajectory.json').write_text(json.dumps(out,indent=2)+'\n')
for h,d in out.items():print(h,'steps',d['actual_steps_per_rank'].get('0'),'changes',d['worker_changes'].get('0'),'cadence_median',d['full_window_median_wall_s'],'decisions',d['decisions'],'verify_events',len(d['verification_events']))
