#!/usr/bin/env python3
"""Sample only the rank's process tree; normalize CPU time by six cores in reduction."""
import json,os,pathlib,signal,sys,time
parent=int(sys.argv[1]);out=pathlib.Path(sys.argv[2]);own=os.getpid();hz=os.sysconf('SC_CLK_TCK');last={};total=0.;stop=False

def terminate(*_):
 global stop
 stop=True
signal.signal(signal.SIGTERM,terminate)
out.parent.mkdir(exist_ok=True)
bindings=None;recorded=set()
if os.environ.get('DLIO_LF_ISOLATE_CORES') == '2':
 bindings=out.with_name(out.name.replace('cpu-rank-','bindings-rank-')).open('w',buffering=1)
 expected=sorted(os.sched_getaffinity(parent))
with out.open('w',buffering=1) as f:
 while not stop and pathlib.Path(f'/proc/{parent}').exists():
  queue=[parent];seen=set();current={}
  while queue:
   pid=queue.pop()
   if pid==own or pid in seen:continue
   seen.add(pid)
   try:
    base=pathlib.Path(f'/proc/{pid}');s=(base/'stat').read_text().rsplit(')',1)[1].split()
    key=(pid,s[19])
    if bindings is not None and key not in recorded:
     mask=sorted(os.sched_getaffinity(pid))
     bindings.write(json.dumps(dict(pid=pid,start_ticks=s[19],parent_pid=int(s[1]),name=(base/'comm').read_text().strip(),cpus=mask,expected_rank_cpus=expected,matches_rank_mask=mask==expected))+'\n')
     recorded.add(key)
    ticks=(int(s[11])+int(s[12]))/hz;current[key]=ticks
    queue.extend(map(int,(base/f'task/{pid}/children').read_text().split()))
   except (FileNotFoundError,ProcessLookupError,PermissionError):pass
  for key,value in current.items():total+=max(0,value-last.get(key,0))
  last.update(current)
  f.write(json.dumps(dict(monotonic=time.monotonic(),cpu_seconds=total,processes=len(current)))+'\n')
  time.sleep(.2)

if bindings is not None:bindings.close()
