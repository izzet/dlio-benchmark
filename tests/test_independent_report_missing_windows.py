"""A completed report with absent wait windows preserves missing values."""
import csv
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / 'examples/liveflow/cosmoflow_independent/report-campaign-block.py'


class Tests(unittest.TestCase):
    def test_complete_apps_without_wait_windows_report_na(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            per_host = {}
            trajectory = {}
            for host in ['h0', 'h1']:
                directory = root / 'liveflow' / host
                (directory / 'screen-metrics').mkdir(parents=True)
                (directory / 'postflight.jsonl').write_text(json.dumps({'rings': {'trace': {'dropped': 0, 'backlog_bytes': 0, 'unclaimed': 0}}}) + '\n')
                steps = [dict(workers=4 if n < 100 else 8, unix_s=n) for n in range(1, 4097)]
                (directory / 'screen-metrics/steps-rank-0.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in steps))
                per_host[f'liveflow/{host}'] = dict(analyzer_complete=True, app_elapsed_s=100,
                    steady_critical_step_ms=20, nominal_au_percent=25,
                    steady_decode_wait_ms=None, steady_storage_wait_ms=None,
                    aggregate_read_GB_s=1, cpu_mean_fraction_6_cores=.5,
                    cpu_per_rank=[dict(rank=0, fraction_of_6_cores=.5)])
                trajectory[host] = dict(worker_changes={'0': [{'completed_step': 1}, {'completed_step': 100}]}, verification_events=[])
            aggregate = dict(hosts=2, mean_host_elapsed_s=100, mean_host_step_ms=20,
                mean_host_au_percent=25, mean_host_steady_decode_wait_ms=None,
                mean_host_steady_storage_wait_ms=None, sum_host_read_GB_s=2,
                mean_cpu_fraction_6_cores=.5, analyzer_valid_step_bins_mean_host=0,
                analyzer_steady_window_step_bins_mean_host=0)
            (root / 'block-results.json').write_text(json.dumps(dict(per_host=per_host, aggregate={'liveflow': aggregate})))
            (root / 'lf-trajectory.json').write_text(json.dumps(trajectory))
            result = subprocess.run([sys.executable, str(SCRIPT), str(root), 'smoke'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            updated = json.loads((root / 'block-results.json').read_text())
            self.assertIsNone(updated['aggregate']['liveflow']['median_host_noncalibration_window_seconds'])
            self.assertEqual(updated['aggregate']['liveflow']['first_apply_steps'], {'h0': 100, 'h1': 100})
            self.assertEqual(updated['aggregate']['liveflow']['final_workers'], {'h0': 8, 'h1': 8})
            self.assertIn('NA/NA', (root / 'RESULTS.md').read_text())
            with (root / 'per-host.csv').open() as handle:
                self.assertTrue(all(row['decode_wait_ms'] == '' and row['noncalibration_window_median_seconds'] == '' for row in csv.DictReader(handle)))


if __name__ == '__main__':
    unittest.main()
