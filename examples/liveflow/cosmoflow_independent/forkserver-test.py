#!/usr/bin/env python3
import hashlib,json,os,pathlib,subprocess,time
own=pathlib.Path(__file__).resolve().parent;root=pathlib.Path(os.environ['DLIO_FORKSERVER_TEST_ROOT']);assert root.is_absolute();root.mkdir(parents=True,exist_ok=True)
cap=int(os.environ['DLIO_FORKSERVER_CAP_S']);started=time.monotonic();deadline=started+cap-15
host=subprocess.check_output(['flux','exec','-r','all','hostname'],text=True).strip();results=[]
launcher=own/'run-arm-independent-campaign.sh'
for key,arm,steps,timers in [('w12-timers-off','static-best',256,'0'),('w12-timers-on','static-best',256,'1'),('liveflow-timers-on','liveflow',4096,'1')]:
 remaining=int(deadline-time.monotonic());d=root/key/host;d.mkdir(parents=True,exist_ok=True)
 if remaining<60:break
 env=dict(os.environ,LIVEFLOW_MOTIVATION_TIMERS=timers,LIVEFLOW_MOTIVATION_CAPTURE_STEPS='400',LIVEFLOW_MOTIVATION_WARMUP_STEPS='20',LIVEFLOW_MOTIVATION_GPU_TRACE='0',DLIO_WORKER_CONTEXT='forkserver',DLIO_STATIC_W='12',DLIO_STATIC_P='2',DLIO_SCREEN_COMPUTE='.00551',DLIO_CAMPAIGN_EPOCHS='1',DLIO_CAMPAIGN_STEPS=str(steps),DLIO_CONTROL_STEPS='512',DLIO_APP_TIMEOUT_S=str(max(30,remaining-25)),DLIO_AGENT_TIMEOUT=str(remaining),DLIO_CORE_LIMIT_BYTES=str(2**31))
 (d/'wrapper-versions.json').write_text(json.dumps({n:hashlib.sha256((own/n).read_bytes()).hexdigest() for n in ['rank-independent-campaign.py','dlio_spawn_worker.py','dlio_forkserver_probe.py','run-host-job.py','run-arm-independent-campaign.sh']},indent=2)+'\n')
 begin=time.monotonic()
 with (root/(key+'-launch.log')).open('w') as log:
  proc=subprocess.Popen(['bash',str(launcher),str(d),arm,host],env=env,stdout=log,stderr=subprocess.STDOUT)
  try:rc=proc.wait(timeout=remaining)
  except subprocess.TimeoutExpired:
   for kind in ['app','agent']:
    p=d/(kind+'-jobid.txt')
    if p.exists():subprocess.run(['flux','cancel',p.read_text().strip()],check=False)
   proc.terminate();rc=124
 results.append(dict(key=key,timers_enabled=timers=='1',context='forkserver',arm=arm,epochs=1,steps_per_epoch=steps,returncode=rc,wall_s=time.monotonic()-begin,directory=str(d)))
 (root/'case-results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results[-1]),flush=True)
(root/'driver-elapsed.json').write_text(json.dumps(dict(elapsed_s=time.monotonic()-started,cap_s=cap,host=host))+'\n')
