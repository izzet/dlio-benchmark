#!/usr/bin/env python3
import csv,json,pathlib,statistics,sys
root=pathlib.Path(sys.argv[1]);results={};epochs=int(sys.argv[2]) if len(sys.argv)>2 else 5;nsteps=int(sys.argv[3]) if len(sys.argv)>3 else 4096
paired_file=root/'paired-hosts.txt';paired=set(paired_file.read_text().splitlines()) if paired_file.exists() else None
if paired is not None:assert 14<=len(paired)<=16,('invalid paired host count',len(paired))
for arm in sorted(root.glob('*/*')):
 if paired is not None and arm.name not in paired:continue
 files=sorted((arm/'screen-metrics').glob('rank-*.json'))
 if len(files)!=8 or not (arm/'status.txt').exists():continue
 status=dict(line.split('=',1) for line in (arm/'status.txt').read_text().splitlines())
 if any(x!='0' for x in status.values()):continue
 ranks=[json.loads(p.read_text()) for p in files];critical={};cpu=[];training_epoch_wall={}
 assert sorted(r['rank'] for r in ranks)==list(range(8)),arm
 identity_files=list((arm/'sample-identity').glob('rank-*.json'))
 assert len(identity_files)==8,(arm,len(identity_files))
 for identity_file in identity_files:
  identity=json.loads(identity_file.read_text());rank=identity['rank']
  assert set(identity['epoch_indices'])=={str(e) for e in range(1,epochs+1)},(arm,rank)
  assert identity['epoch_indices']==identity['expected_epoch_indices'],(arm,rank)
 for r in ranks:
  samples=[json.loads(x) for x in (arm/'screen-metrics'/f"cpu-rank-{r['rank']}.jsonl").open()]
  used=wall=0.
  for epoch in range(1,epochs+1):
   steps=[x for x in r['steps'] if x['epoch']==epoch]
   assert len(steps)==nsteps,(arm,r['rank'],epoch,len(steps))
   training_epoch_wall[epoch]=max(training_epoch_wall.get(epoch,0),steps[-1]['monotonic']-(steps[0]['monotonic']-steps[0]['seconds']))
   for s in steps:
    if s['step']>20:
     key=(epoch,s['step']);critical[key]=max(critical.get(key,0),s['seconds'])
   start=steps[19]['monotonic'];end=steps[-1]['monotonic']
   selected=[x for x in samples if start<=x['monotonic']<=end]
   if len(selected)>1:
    used+=selected[-1]['cpu_seconds']-selected[0]['cpu_seconds'];wall+=selected[-1]['monotonic']-selected[0]['monotonic']
  cpu.append(dict(rank=r['rank'],fraction_of_6_cores=used/(6*wall) if wall else None,observed_seconds=wall))
 waits=[];steady_waits=[];step_components=[];read_available=False;read_bytes=0.;valid=invalid=0
 agents=list(arm.glob('agent_*'));raw_replay=any(arm.glob('raw-drain-*.json'))
 native_complete=len(agents)==1 and (not raw_replay or all((a/'replay-provenance.json').exists() and json.loads((a/'replay-provenance.json').read_text()).get('analysis_completed',False) for a in agents))
 for agent in agents:
  valid_window_files=set()
  for f in agent.glob('analysis_facts_window_*.json'):
   facts=json.loads(f.read_text()).get('facts',[])
   m=next((x['evidence']['metrics'] for x in facts if 'lf_v1_decode_wait' in x.get('evidence',{}).get('metrics',{})),None)
   if m and m.get('lf_v1_valid')==1:
    valid_window_files.add(f.stem.removeprefix('analysis_facts_window_'))
    waits.append((m['lf_v1_decode_wait'],m['lf_v1_storage_wait'],m.get('step_nunique',1)));valid+=1
    step_components.append((m.get('lf_v1_comm_wait',0),m.get('lf_v1_compute_time',0),m.get('step_nunique',1)))
    if m.get('epoch_nunique')==1 and m.get('step_min',0)>20 and m.get('step_max',nsteps+1)<=nsteps:
     steady_waits.append(waits[-1])
   else:invalid+=1
  invalid+=sum(f.stem.removeprefix('window_integrity_ledger_window_') not in valid_window_files for f in agent.glob('window_integrity_ledger_window_*.csv'))
  for f in agent.glob('flat_view_window_window_*.csv'):
   for row in csv.DictReader(f.open()):
    value=row.get('reader_posix_read_size_sum')
    if value and value not in ['NA','nan']:
     read_bytes+=float(value);read_available=True
 partial_waits=dict(decode_wait_ms=1000*sum(x[0]*x[2] for x in waits)/sum(x[2] for x in waits) if waits else None,storage_wait_ms=1000*sum(x[1]*x[2] for x in waits)/sum(x[2] for x in waits) if waits else None,read_bytes=read_bytes if read_available else None)
 if not native_complete:waits=[];steady_waits=[];step_components=[];read_available=False
 denom=sum(x[2] for x in waits)
 decode=sum(x[0]*x[2] for x in waits)/denom if denom else None
 storage=sum(x[1]*x[2] for x in waits)/denom if denom else None
 comm=sum(x[0]*x[2] for x in step_components)/denom if denom else None
 compute=sum(x[1]*x[2] for x in step_components)/denom if denom else None
 component_values=dict(decode=decode,storage=storage,communication=comm,compute=compute)
 largest_component=max(component_values,key=component_values.get) if denom else 'unmeasured'
 steady_denom=sum(x[2] for x in steady_waits)
 steady_decode=sum(x[0]*x[2] for x in steady_waits)/steady_denom if steady_denom else None
 steady_storage=sum(x[1]*x[2] for x in steady_waits)/steady_denom if steady_denom else None
 elapsed=dict(x.split('=',1) for x in (arm/'app-elapsed.txt').read_text().splitlines());wall=float(elapsed['end_s'])-float(elapsed['start_s'])
 results[f'{arm.parent.name}/{arm.name}']=dict(status=status,analyzer_complete=native_complete,analyzer_hosts=len(agents),partial_analyzer_diagnostic=partial_waits if not native_complete else None,sample_identity=f'8 ranks x {epochs} exact seeded {nsteps}-sample sequences; no loss in DLIO',app_elapsed_s=wall,steady_decode_wait_ms=1000*steady_decode if steady_decode is not None else None,steady_storage_wait_ms=1000*steady_storage if steady_storage is not None else None,analyzer_steady_only_windows=len(steady_waits),steady_steps_per_rank=len(critical),steady_critical_step_ms=1000*statistics.mean(critical.values()),comm_wait_ms=1000*comm if comm is not None else None,compute_time_ms=1000*compute if compute is not None else None,largest_step_component=largest_component,decode_wait_ms=1000*decode if decode is not None else None,storage_wait_ms=1000*storage if storage is not None else None,larger_input_wait=('decode' if decode>storage else 'storage') if denom else 'unmeasured',analyzer_valid_windows=valid,analyzer_absent_or_invalid_windows=invalid,analyzer_scope='all valid native windows, including startup; per-step max-process exposure, step-count weighted across nodes',read_bytes=read_bytes if read_available else None,aggregate_read_GB_s=read_bytes/1e9/wall if read_available else None,aggregate_training_read_GB_s=read_bytes/1e9/sum(training_epoch_wall.values()) if read_available else None,training_wall_s=sum(training_epoch_wall.values()),training_read_rate_scope='all analyzer POSIX read bytes / sum over epochs of longest rank training interval; includes first20 steps, excludes application initialization and between-epoch cold eviction',read_rate_scope='analyzer reader POSIX read bytes / full application wall time',cpu_per_rank=cpu,cpu_mean_fraction_6_cores=statistics.mean(x['fraction_of_6_cores'] for x in cpu if x['fraction_of_6_cores'] is not None),cpu_caveat='0.2s process-tree samples; worker exit can omit final sub-interval CPU')

 # MLPerf Storage v1 uses configured nominal compute as numerator, rather than measured oversleep.
 actual=[];nominal=[];numerator=denominator=0.
 for p in sorted((arm/'dlio').glob('*_output.json')):
  d=json.loads(p.read_text())
  for epoch in range(1,epochs+1):
   e=d[str(epoch)];a=e['au']['block1'];comp=e['compute']['block1'][1:]
   n=len(comp)*.00551
   assert a>0 and len(comp)==nsteps-1,(p,epoch,len(comp),a)
   elapsed=sum(comp)/(a/100)
   actual.append(a);nominal.append(100*n/elapsed);numerator+=n;denominator+=elapsed
 v=results[f'{arm.parent.name}/{arm.name}']
 v.update(analyzer_valid_step_bins=denom,analyzer_steady_window_step_bins=steady_denom,host=arm.name,arm=arm.parent.name,nominal_au_percent=statistics.mean(nominal),observed_compute_au_percent=statistics.mean(actual),au_pooled_percent=100*numerator/denominator,au_nominal_compute_s=numerator,au_measured_denominator_s=denominator,passes_config70=statistics.mean(nominal)>=70,passes_guidelines90=statistics.mean(nominal)>=90)

aggregate={}
for armname in sorted({v['arm'] for v in results.values()}):
 values=[v for v in results.values() if v['arm']==armname]
 a=dict(hosts=len(values),mean_host_elapsed_s=statistics.mean(v['app_elapsed_s'] for v in values),max_host_elapsed_s=max(v['app_elapsed_s'] for v in values),mean_host_step_ms=statistics.mean(v['steady_critical_step_ms'] for v in values),mean_host_au_percent=statistics.mean(v['nominal_au_percent'] for v in values),pooled_au_percent=100*sum(v['au_nominal_compute_s'] for v in values)/sum(v['au_measured_denominator_s'] for v in values),sum_host_read_GB_s=sum(v['aggregate_read_GB_s'] or 0 for v in values),mean_cpu_fraction_6_cores=statistics.mean(v['cpu_mean_fraction_6_cores'] for v in values),hosts_passing70=sum(v['passes_config70'] for v in values),hosts_passing90=sum(v['passes_guidelines90'] for v in values))
 for key in ['decode_wait_ms','storage_wait_ms','steady_decode_wait_ms','steady_storage_wait_ms','comm_wait_ms']:
  present=[v[key] for v in values if v[key] is not None]
  a['mean_host_'+key]=statistics.mean(present) if present else None
 a['analyzer_hosts_complete']=sum(v['analyzer_complete'] for v in values)
 a['analyzer_valid_step_bins_mean_host']=statistics.mean(v['analyzer_valid_step_bins'] for v in values)
 a['analyzer_steady_window_step_bins_mean_host']=statistics.mean(v['analyzer_steady_window_step_bins'] for v in values)
 aggregate[armname]=a
exclusions=json.loads((root/'excluded-hosts.json').read_text()) if (root/'excluded-hosts.json').exists() else []
if paired is not None:
 for armname in ['original','maxresource-adapted','static-best','liveflow']:
  assert {v['host'] for v in results.values() if v['arm']==armname}==paired,('incomplete paired arm',armname)
output=dict(paired_hosts=sorted(paired) if paired is not None else None,excluded_hosts=exclusions,per_host=results,aggregate=aggregate,replications=1,scope='one paired campaign block; hosts are not campaign repetitions',epochs=epochs,steps_per_epoch=nsteps)
(root/'block-results.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(aggregate,indent=2))
