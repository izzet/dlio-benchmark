#!/usr/bin/env python3
"""Bind six distinct physical cores per rank, then run the committed DLIO module."""
if __name__ == '__main__':
    import hashlib,json,os,pathlib,runpy,socket,sys
    allowed=os.sched_getaffinity(0);cores={}
    for cpu in sorted(allowed):
     p=pathlib.Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
     key=((p/'physical_package_id').read_text().strip(),(p/'core_id').read_text().strip())
     cores.setdefault(key,cpu)
    cpus=list(cores.values());rank=int(os.environ['FLUX_TASK_RANK']);local=rank%8
    os.environ['RANK']=str(rank)
    assert len(cpus)>=48,(socket.gethostname(),len(cpus))
    chosen=cpus[local*6:(local+1)*6];os.sched_setaffinity(0,chosen)
    out=pathlib.Path(os.environ['RUN_DIR'])/'rank-bindings';out.mkdir(exist_ok=True)
    source=pathlib.Path(os.environ['LIVEFLOW_SNAPSHOT_ROOT'])
    # Start an MPI/DFTracer-free server before importing the application.
    # Existing fork/spawn cases do not enter this branch.
    if os.environ.get('DLIO_WORKER_CONTEXT') == 'forkserver':
        import multiprocessing as mp,time
        from dlio_forkserver_probe import warmup
        mp.set_forkserver_preload(['torch','h5py','numpy','dlio_forkserver_probe'])
        started=time.monotonic()
        ctx=mp.get_context('forkserver');receive,send=ctx.Pipe(duplex=False)
        child=ctx.Process(target=warmup,args=(send,));child.start();send.close()
        assert receive.poll(45),'clean forkserver warmup timeout'
        warm=receive.recv();receive.close();child.join(10)
        assert child.exitcode==0 and not warm['mpi_python'] and not warm['tracer_python'],warm
        server_pid=warm['parent_pid']
        server_file=pathlib.Path(os.environ['RUN_DIR'])/'forkserver-state'/f'server-{server_pid}.json'
        server_state=json.loads(server_file.read_text())
        assert all(server_state['heavy_modules'].values()) and not server_state['mpi_mappings'],server_state
        (pathlib.Path(os.environ['RUN_DIR'])/f'forkserver-rank-{rank}.json').write_text(json.dumps(dict(rank=rank,server_pid=server_pid,warmup_s=time.monotonic()-started,server=server_state),indent=2)+'\n')
        print('DLIO clean forkserver before MPI rank='+str(rank)+' server_pid='+str(server_pid),flush=True)
    import dlio_benchmark.main as app
    # Wrapper-local registry configuration: retain every legal worker count.
    # A range alone is coarsened to odd levels; off-grid W4 snaps to W3, then raises to W5.
    # Preserve the released actuator and response rules; only its candidate values change.
    if os.environ.get('DFOPTIMIZER_ENABLE') == '1':
        import inspect
        register = inspect.getclosurevars(app.DLIOBenchmark.make_loader._tunable_register).nonlocals['_register_knobs']
        build = inspect.getclosurevars(register).nonlocals['_build_knob_defs']
        specs = inspect.getclosurevars(build).nonlocals['_knob_specs']
        spec = specs['read_threads']
        assert tuple(spec['range']) == (1,12), spec
        spec['values'] = list(range(1,13))
        spec['step_factor'] = 2.0
        print('DLIO registry read_threads values=1..12 step_factor=2 window_target_s='+os.environ['LIVEFLOW_WINDOW_TARGET_S'],flush=True)
    assert pathlib.Path(app.__file__).resolve()==pathlib.Path(os.environ['DLIO_APP_ROOT'])/'dlio_benchmark/main.py',app.__file__
    (out/f'rank-{rank}.json').write_text(json.dumps(dict(rank=rank,host=socket.gethostname(),cpus=chosen,physical_cores=6,module=app.__file__,sha256=hashlib.sha256(pathlib.Path(app.__file__).read_bytes()).hexdigest()),indent=2))
    # Honor the pinned seeded shuffle intent in the retained PyTorch index-reader adaptation.
    # TFRecord/TF shuffle-buffer semantics remain an explicit format/reader deviation.
    import numpy as np
    from dlio_benchmark.utils.config import ConfigArguments
    from dlio_benchmark.data_loader.live_data_loader import LiveSampler
    reconfigure=ConfigArguments.reconfigure
    def shuffled_reconfigure(self,epoch_number):
        if not hasattr(self,'_screen_base_files'):
            self._screen_base_files=list(self.file_list_train)
        self.file_list_train=list(self._screen_base_files)
        seed=self.seed+epoch_number if self.seed_change_epoch else self.seed
        np.random.RandomState(seed).shuffle(self.file_list_train)
        self._screen_epoch=epoch_number
        return reconfigure(self,epoch_number)
    ConfigArguments.reconfigure=shuffled_reconfigure
    sampler_init=LiveSampler.__init__
    def shuffled_sampler(self,*args,**kwargs):
        sampler_init(self,*args,**kwargs)
        cfg=ConfigArguments.get_instance()
        seed=cfg.seed+cfg._screen_epoch if cfg.seed_change_epoch else cfg.seed
        np.random.RandomState(seed).shuffle(self._indices)
    LiveSampler.__init__=shuffled_sampler

    # Arm-common sample identity instrument. Keep actual reader and yielded payload.
    # Index travels with its real batch, so prefetch/rebuild cannot hide a restart.
    from dlio_benchmark.data_loader.torch_data_loader import TorchDataset,TorchDataLoader
    from dlio_benchmark.data_loader.live_data_loader import LiveDataLoader
    # A module-qualified dataset is reconstructed in spawn children without app/MPI imports.
    from dlio_spawn_worker import IndexedTorchDataset
    import dlio_benchmark.data_loader.torch_data_loader as torch_module
    import dlio_benchmark.data_loader.live_data_loader as live_module
    torch_module.TorchDataset=IndexedTorchDataset
    live_module.TorchDataset=IndexedTorchDataset
    lifecycle_stream=None
    def lifecycle_record(event,loader,**fields):
        record=dict(event=event,rank=rank,epoch=loader.epoch_number,workers=loader._read_threads,
                    samples_yielded=loader._samples_yielded,context=loader._args.multiprocessing_context,
                    monotonic=time.monotonic(),unix_s=time.time(),**fields)
        if lifecycle_stream is not None:lifecycle_stream.write(json.dumps(record,separators=(',',':'))+'\n')
    real_build=LiveDataLoader._build_inner
    def measured_build(self):
        self._dlio_build_started=time.monotonic()
        lifecycle_record('build_start',self,mpi_module_present='mpi4py.MPI' in sys.modules,
                         h5py_present='h5py' in sys.modules,
                         hdf5_open_files=(sys.modules['h5py'].h5f.get_obj_count(types=sys.modules['h5py'].h5f.OBJ_FILE) if 'h5py' in sys.modules else 0))
        result=real_build(self)
        lifecycle_record('build_return',self,elapsed_s=time.monotonic()-self._dlio_build_started)
        self._dlio_first_batch_pending=True
        return result
    LiveDataLoader._build_inner=measured_build
    real_resize=LiveDataLoader._apply_reconfigure
    def measured_resize(self):
        self._dlio_resize_started=time.monotonic()
        old_workers=self._read_threads
        lifecycle_record('resize_start',self,target_workers=self._pending_reconfigure[0])
        result=real_resize(self)
        lifecycle_record('resize_return',self,old_workers=old_workers,elapsed_s=time.monotonic()-self._dlio_resize_started)
        return result
    LiveDataLoader._apply_reconfigure=measured_resize
    identities={}
    def wrap_next(original):
        def checked_next(self):
            epoch=str(self.epoch_number)
            iterator=iter(original(self))
            while True:
                started=time.monotonic()
                try:payload,index=next(iterator)
                except StopIteration:break
                if getattr(self,'_dlio_first_batch_pending',False):
                    fields=dict(build_to_first_batch_s=time.monotonic()-self._dlio_build_started,
                                first_fetch_s=time.monotonic()-started)
                    if hasattr(self,'_dlio_resize_started'):
                        fields['resize_to_first_batch_s']=time.monotonic()-self._dlio_resize_started
                        del self._dlio_resize_started
                    lifecycle_record('first_batch',self,**fields)
                    self._dlio_first_batch_pending=False
                ids=[int(v) for v in index.tolist()]
                identities.setdefault(epoch,[]).extend(ids)
                identity_stream.write(json.dumps(dict(epoch=int(epoch),indices=ids))+'\n')
                yield payload
        return checked_next
    TorchDataLoader.next=wrap_next(TorchDataLoader.next)
    LiveDataLoader.next=wrap_next(LiveDataLoader.next)
    # Arm-common screen diagnostics, no application math changes.
    import subprocess,time
    steps=[];cold=[];current_epoch=0;step_number=0;active_loader=None
    step_stream=None
    real_make=app.DLIOBenchmark.make_loader
    cold_epochs=set()
    def cold_make(self,epoch,*args,**kwargs):
        if epoch not in cold_epochs:
            if rank==0:print('DLIO worker_start_method='+self.args.multiprocessing_context+' fork_safety='+json.dumps({k:os.environ.get(k) for k in ['IBV_FORK_SAFE','RDMAV_FORK_SAFE','MV2_SUPPORT_FORK_SAFETY']}),flush=True)
            files=self.args.file_list_train
            n=len(files)//self.comm_size
            started=time.monotonic()
            for name in files[self.my_rank*n:(self.my_rank+1)*n]:
                fd=os.open(name,os.O_RDONLY)
                try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
                finally:os.close(fd)
            cold.append(dict(epoch=epoch,files=n,elapsed=time.monotonic()-started,method='POSIX_FADV_DONTNEED advisory; not certified cache eviction'))
            cold_epochs.add(epoch)
        return real_make(self,epoch,*args,**kwargs)
    app.DLIOBenchmark.make_loader=cold_make
    real_train=app.DLIOBenchmark._train
    def measured_train(self,epoch,loader):
        global current_epoch,step_number,active_loader
        current_epoch=epoch;step_number=0;active_loader=loader
        return real_train(self,epoch,loader)
    app.DLIOBenchmark._train=measured_train
    for cls in (app._StepDirect,app._NullStepDirect):
        original=cls.end_step
        def measured_end(self,seconds,_original=original):
            global step_number
            _original(self,seconds);step_number+=1
            record=dict(epoch=current_epoch,step=step_number,seconds=seconds,monotonic=time.monotonic(),unix_s=time.time(),workers=getattr(active_loader,'_read_threads',None),prefetch=getattr(active_loader,'_prefetch_size',None))
            steps.append(record)
            if step_stream is not None:step_stream.write(json.dumps(record,separators=(',',':'))+'\n')
        cls.end_step=measured_end
    metrics_out=pathlib.Path(os.environ['RUN_DIR'])/'screen-metrics';metrics_out.mkdir(exist_ok=True)
    step_stream=(metrics_out/f'steps-rank-{rank}.jsonl').open('w',buffering=1)
    lifecycle_stream=(metrics_out/f'lifecycle-rank-{rank}.jsonl').open('w',buffering=1)
    identity_stream=(metrics_out/f'identities-rank-{rank}.jsonl').open('w',buffering=1)
    monitor=subprocess.Popen([sys.executable,str(pathlib.Path(__file__).with_name('cpu-monitor.py')),str(os.getpid()),str(metrics_out/f'cpu-rank-{rank}.jsonl')])
    sys.argv=['dlio_benchmark.main',*sys.argv[1:]]
    # Invoke the real imported committed module as its CLI, retaining its Hydra entry.
    try:
        app.main()
    finally:
        monitor.terminate();monitor.wait();step_stream.close();lifecycle_stream.close();identity_stream.close()
        (metrics_out/f'rank-{rank}.json').write_text(json.dumps(dict(rank=rank,physical_cores=6,steps=steps,cold=cold,cpu_sampling_seconds=.2,cpu_caveat='worker exit between samples can omit final <0.2s CPU per worker'),separators=(',',':'))+'\n')

    expected={}
    for epoch in range(1,int(os.environ.get("DLIO_CAMPAIGN_EPOCHS","5"))+1):
        ids=list(range(rank*4096,(rank+1)*4096))
        np.random.RandomState(123+epoch).shuffle(ids)
        expected[str(epoch)]=ids[:int(os.environ.get("DLIO_CAMPAIGN_STEPS","4096"))]
    identity_out=pathlib.Path(os.environ['RUN_DIR'])/'sample-identity';identity_out.mkdir(exist_ok=True)
    (identity_out/f'rank-{rank}.json').write_text(json.dumps(dict(rank=rank,epoch_indices=identities,expected_epoch_indices=expected,loss=None,loss_reason='DLIO simulates computation; no model/loss'),separators=(',',':'))+'\n')
