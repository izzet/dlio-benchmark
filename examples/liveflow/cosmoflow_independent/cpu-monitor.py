#!/usr/bin/env python3
"""Sample only the rank's process tree; normalize CPU time by six cores in reduction."""
import json,os,pathlib,signal,sys,time
parent=int(sys.argv[1]);out=pathlib.Path(sys.argv[2]);own=os.getpid();hz=os.sysconf('SC_CLK_TCK');last={};total=0.;stop=False

def terminate(*_):
 global stop
 stop=True
signal.signal(signal.SIGTERM,terminate)
out.parent.mkdir(exist_ok=True)
with out.open('w',buffering=1) as f:
 while not stop and pathlib.Path(f'/proc/{parent}').exists():
  queue=[parent];seen=set();current={}
  while queue:
   pid=queue.pop()
   if pid==own or pid in seen:continue
   seen.add(pid)
   try:
    base=pathlib.Path(f'/proc/{pid}');s=(base/'stat').read_text().rsplit(')',1)[1].split()
    key=(pid,s[19]);ticks=(int(s[11])+int(s[12]))/hz;current[key]=ticks
    queue.extend(map(int,(base/f'task/{pid}/children').read_text().split()))
   except (FileNotFoundError,ProcessLookupError,PermissionError):pass
  for key,value in current.items():total+=max(0,value-last.get(key,0))
  last.update(current)
  f.write(json.dumps(dict(monotonic=time.monotonic(),cpu_seconds=total,processes=len(current)))+'\n')
  time.sleep(.2)
