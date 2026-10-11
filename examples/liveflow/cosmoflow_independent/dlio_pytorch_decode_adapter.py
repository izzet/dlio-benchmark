"""Explicit DLIO-PyTorch adapter registration; shared frozen files stay byte-identical."""
import os

from dftracer.analyzer.dftracer import DFTracerAnalyzer
from dlio_hdf5_decode import prepare_hdf5_decode


class DLIOHDF5Analyzer(DFTracerAnalyzer):
    def _decode_traces(self, traces):
        if not getattr(self.preset, 'liveflow_families', {}).get('hdf5_decode_residual'):
            return traces
        if hasattr(traces, 'compute'):
            traces = traces.compute()
        return prepare_hdf5_decode(traces, time_resolution=self.time_resolution)

    def _analyze_trace_fast(self, traces, *args, **kwargs):
        return super()._analyze_trace_fast(self._decode_traces(traces), *args, **kwargs)

    def _analyze_trace(self, traces, *args, **kwargs):
        return super()._analyze_trace(self._decode_traces(traces), *args, **kwargs)


def install():
    """Register the adapter only in an explicitly selected DLIO HDF5 agent process."""
    if (os.environ.get('LIVEFLOW_WORKLOAD') != 'dlio-pytorch'
            or os.environ.get('LIVEFLOW_DLIO_HDF5_DECODE') != '1'):
        raise RuntimeError('DLIO decode adapter requires explicit dlio-pytorch HDF5 selection')
    import dftracer.analyzer as package
    from dftracer.analyzer import liveflow_v1
    if getattr(liveflow_v1, '_dlio_hdf5_adapter_installed', False):
        return
    original_declaration = liveflow_v1.declaration
    original_prepare = liveflow_v1.prepare_views
    original_init = package.init_with_hydra

    def declaration(workload=None, tf_decode=None):
        declared = original_declaration(workload, tf_decode)
        if (declared['workload'] == 'dlio-pytorch'
                and os.environ.get('LIVEFLOW_DLIO_HDF5_DECODE') == '1'):
            declared = dict(declared, input_families=['storage', 'decode'], split_wait=True,
                            hdf5_decode_residual=True)
        return declared

    def prepare_views(flat_views, declared, context=None):
        if declared.get('hdf5_decode_residual') and ('window',) in flat_views:
            # DLIO calls its undivided input fetch wait stall.storage. Storage service
            # remains the original POSIX layer; residual decode apportions that same wait.
            flat_views = dict(flat_views)
            window = flat_views[('window',)].copy()
            if 'stall_storage_time_proc_max' in window:
                present = window['stall_storage_time_proc_max'].notna()
                window.loc[present, 'fetch_iter_time_proc_max'] = window.loc[present, 'stall_storage_time_proc_max']
            flat_views[('window',)] = window
        return original_prepare(flat_views, declared, context)

    def init_with_hydra(overrides, *args, **kwargs):
        return original_init(list(overrides)+[
            'analyzer._target_=dlio_pytorch_decode_adapter.DLIOHDF5Analyzer'], *args, **kwargs)

    liveflow_v1.declaration = declaration
    liveflow_v1.prepare_views = prepare_views
    package.init_with_hydra = init_with_hydra
    liveflow_v1._dlio_hdf5_adapter_installed = True
