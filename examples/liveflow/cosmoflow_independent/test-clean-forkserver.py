#!/usr/bin/env python3
import json,multiprocessing as mp,os,pathlib,tempfile,time
if __name__=='__main__':
 with tempfile.TemporaryDirectory() as temp:
  os.environ['RUN_DIR']=temp
  from dlio_forkserver_probe import warmup
  mp.set_forkserver_preload(['torch','h5py','numpy','dlio_forkserver_probe'])
  ctx=mp.get_context('forkserver');receive,send=ctx.Pipe(duplex=False);begin=time.monotonic()
  child=ctx.Process(target=warmup,args=(send,));child.start();send.close()
  assert receive.poll(45),'server did not warm up'
  state=receive.recv();receive.close();child.join(10);assert child.exitcode==0
  server=json.loads((pathlib.Path(temp)/'forkserver-state'/f'server-{state["parent_pid"]}.json').read_text())
  assert all(server['heavy_modules'].values()) and not server['mpi_python'] and not server['mpi_mappings']
  assert not server['tracer_python'] and not server['tracer_shm_mapped'] and server['native_threads']==1 and server['hdf5_open_files']==0
  print('PASS clean preloaded forkserver',json.dumps(server),'warmup_s',time.monotonic()-begin)
