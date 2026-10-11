#!/usr/bin/env python3
"""Static screen: continuously drain SHM using released reader/persistence; no analyzer/controller."""
import base64,json,os,pathlib,signal,socket,time
from shm_reader import ShmRingReader
from trace_persistence import TracePersistence
from liveflow_transport.shm_io import unpack
root=pathlib.Path(os.environ['RUN_DIR']);host=socket.gethostname();prefix=os.environ['LIVEFLOW_SHM_PREFIX']
trace=ShmRingReader(prefix+'.trace');control=ShmRingReader(prefix+'.control')
persist=TracePersistence.from_environment();assert persist is not None
stop=root/'raw-drain.stop';interrupt=False;last={};gaps=[];records=0

def terminate(*_):
 global interrupt
 interrupt=True
signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
control_file=(root/f'control-{host}.jsonl').open('w',buffering=1)

def drain():
 global records
 count=0
 for topic,reader in [('trace',trace),('control',control)]:
  for lane,seq,payload in reader.drain():
   key=(topic,lane)
   if key in last and seq!=last[key]+1:gaps.append(dict(topic=topic,lane=lane,previous=last[key],sequence=seq))
   last[key]=seq;metadata,data=unpack(payload);count+=1
   if topic=='trace':persist.enqueue(lane,data);records+=1
   else:control_file.write(json.dumps(dict(lane=lane,sequence=seq,metadata=metadata,data_base64=base64.b64encode(data).decode()))+'\n')
 return count

print('[raw-drain] ready host='+host,flush=True)
try:
 while not stop.exists() and not interrupt:
  drain();time.sleep(.02)
 # The launcher creates stop only after all app ranks exit and EOS publication.
 # Retain a finite committed prefix, then wait one quiet interval before closing.
 drain();time.sleep(.5);drain()
finally:
 persist.close();control_file.close()
 rings={}
 for topic,reader in [('trace',trace),('control',control)]:
  stats=reader.stats();unclaimed=reader.unclaimed()[0]
  rings[topic]=dict(written=sum(s.written for s in stats),dropped=sum(s.dropped for s in stats),unclaimed=unclaimed,backlog_bytes=sum(s.backlog for s in stats))
  reader.close()
 report=dict(host=host,launcher_stop=stop.exists(),interrupted=interrupt,trace_records=records,persisted_records=persist._persisted_records,persisted_events=persist._persisted_events,persistence_dropped_records=persist._dropped_records,sequence_gaps=gaps,rings=rings)
 (root/f'raw-drain-{host}.json').write_text(json.dumps(report,indent=2)+'\n')
 print('[raw-drain] complete '+json.dumps(report),flush=True)
 if interrupt or not stop.exists() or gaps or persist._dropped_records or any(r['dropped'] or r['unclaimed'] or r['backlog_bytes'] for r in rings.values()):raise SystemExit(1)
