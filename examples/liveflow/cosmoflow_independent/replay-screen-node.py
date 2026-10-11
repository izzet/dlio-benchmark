#!/usr/bin/env python3
"""Replay complete retained static traces through the released native analyzer stream path.

Only batch-count targets are cleared: persisted event JSON has no original batch framing.
Native timestamp fences, eight-rank windows and fact integrity gates remain enabled.
"""
import argparse,base64,collections,json,os,pathlib,time,types
ap=argparse.ArgumentParser();ap.add_argument('arm');ap.add_argument('host');args=ap.parse_args()
arm=pathlib.Path(args.arm).resolve();host=args.host;out=arm/f'agent_{host}';out.mkdir(exist_ok=True)
report=json.loads((arm/f'raw-drain-{host}.json').read_text())
assert report['launcher_stop'] and not report['interrupted']
assert not report['sequence_gaps'] and report['persistence_dropped_records']==0
assert all(r['dropped']==r['unclaimed']==r['backlog_bytes']==0 for r in report['rings'].values())
# Retain all original control records; mutate only an independent replay copy.
controls=[json.loads(line) for line in (arm/f'control-{host}.jsonl').open()]
def control_time(c):
 m=c['metadata'];start=m.get('trace_ts_us',m.get('ts_unix_ns',0)//1000)
 return start+(m.get('duration_us',0) if m.get('trigger_event_name')=='window.stop' else 0) if m.get('trigger_event_name')!='liveflow.eos' else float('inf')
controls.sort(key=control_time)
assert controls[-1]['metadata']['trigger_event_name']=='liveflow.eos'
events=[]
for p in sorted((arm/'traces').glob(f'trace-{host}-lane*.pfw')):
 for line in p.open():
  if line.strip() not in ('[',']',''):events.append(json.loads(line))
assert len(events)==report['persisted_events'],(len(events),report['persisted_events'])
events.sort(key=lambda event:event.get('ts',0))
print(f'[replay] loaded host={host} events={len(events)} controls={len(controls)}',flush=True)
import node_agent
node_agent.bootstrap()
from dlio_pytorch_decode_adapter import install
install()
from liveflow_transport import shm_io
from dftracer.analyzer.streaming import transport,continuous
# All retained trace data is available before the first native control closes.
# The original continuous-drain wrapper is unnecessary for an immutable replay.
class Event:
 def __init__(self,metadata,data=b''):self.metadata=metadata;self.data=data
 def acknowledge(self):pass
class Future:
 def __init__(self,source):self.source=source
 def wait(self,timeout_ms=1000):
  if self.source.records:return self.source.records.popleft()
  if timeout_ms:time.sleep(min(timeout_ms/1000,.01))
  return None
class Consumer:
 def __init__(self,records):self.records=collections.deque(records)
 def pull(self):return Future(self)
 def start(self):pass
 def close(self):pass
trace_records=[Event({'type':'batch'},('\n'.join(json.dumps(e,separators=(',',':')) for e in events)+'\n').encode())]
control_records=[]
for c in controls:
 m=dict(c['metadata'])
 if 'events_written' in m:m['events_written']=0
 control_records.append(Event(m,base64.b64decode(c['data_base64'])))
sources={'replay_trace':Consumer(trace_records),'replay_control':Consumer(control_records)}
transport.open_consumer=lambda group_file,topic_name,**kwargs:(None,sources[topic_name])
# v4.0.8+ passes prepare/idle_wait_ms for the native background drainer.
# Immutable replay bypasses that drainer and keeps native synchronous normalization.
continuous.BufferedTraceConsumer=lambda source, **options:source
shm_io.ring_totals=lambda:{'replay_trace':(report['rings']['trace']['written'],report['rings']['trace']['dropped']+report['rings']['trace']['unclaimed'])}
# Use the recorded host as the native fact identity, never the replay machine.
import dftracer.analyzer.analyzer as analyzer_module
analyzer_module.socket=types.SimpleNamespace(gethostname=lambda:host)
cfg=types.SimpleNamespace(data_folder='/p/lustre3/izzet/dlio-benchmark-test/cosmoflow_official_validation_32768_20261010/data',checkpoint_folder=None,hydra_dir=str(out),preset='liveflow-v1',debug=True,workers=1,trace_topic='replay_trace',control_topic='replay_control',window_topic='replay_unused',view_types='[window,proc_name]',facts_rule_file='',fact_source_view='')
dfa=node_agent.build_analyzer(cfg)
provenance=dict(mode='native timestamp fence replay; original batch-count targets set to0 on replay copy only',host=host,runtime=os.environ.get('LIVEFLOW_SNAPSHOT_ROOT'),raw_event_count=len(events),control_record_count=len(controls),collector=report,original_controls=str(arm/f'control-{host}.jsonl'),native_rank_step_and_trace_integrity='enabled',ring_totals_source='recorded raw-drain node counters, measured zero drops')
(out/'replay-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
from omegaconf import OmegaConf
dfa.analyzer.analyze_mofka(group_file='',topic_name='replay_trace',control_topic_name='replay_control',view_types=['window','proc_name'],logical_view_types=dfa.hydra_config.logical_view_types,metric_boundaries=OmegaConf.to_object(dfa.hydra_config.metric_boundaries),trace_drain_grace_ms=100,process_key='pid',extra_columns=node_agent.LIVE_WINDOW_EXTRA_COLUMNS,output_handler=lambda result:None,num_ranks=8)
if getattr(dfa,'cluster',None) is not None and hasattr(dfa.cluster,'close'):dfa.cluster.close()
provenance['analysis_completed']=True
(out/'replay-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('[replay] complete host='+host,flush=True)
