"""Deferring analysis must leave every timed arm and its cleanup intact."""
import os
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'examples/liveflow/cosmoflow_independent/independent-campaign-block.sh'


class Tests(unittest.TestCase):
    def run_block(self, defer):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            own = root / 'checkout/examples/liveflow/cosmoflow_independent'
            own.mkdir(parents=True)
            (own / 'run-arm-independent-campaign.sh').write_text(
                '#!/bin/bash\nmkdir -p "$1"\nprintf "%s %s\\n" "$2" "$3" >> "$CALLS"\n')
            flux = root / 'flux'
            flux.write_text('#!/bin/bash\n'
                'if [[ $1 == exec ]];then for n in {1..16};do echo corona$n;done;'
                'else echo replay >> "$REPLAYS";fi\n')
            flux.chmod(0o755)
            env = dict(os.environ, PATH=str(root)+':'+os.environ['PATH'],
                DLIO_CAMPAIGN_CHECKOUT=str(root/'checkout'),
                DLIO_CAMPAIGN_ROOT=str(root/'results'), DLIO_RUNTIME_ROOT='/fixed/runtime',
                CALLS=str(root/'calls'), REPLAYS=str(root/'replays'))
            if defer is not None:
                env['DLIO_DEFER_REPLAY'] = defer
            else:
                env.pop('DLIO_DEFER_REPLAY', None)
            result = subprocess.run(['bash', str(SCRIPT), '1'], env=env,
                                    text=True, capture_output=True)
            calls = (root/'calls').read_text().splitlines() if (root/'calls').exists() else []
            replays = (root/'replays').read_text().splitlines() if (root/'replays').exists() else []
            marker = root/'results/b1/replay-deferred.txt'
            return result, calls, replays, marker.read_text() if marker.exists() else None

    def test_defer_runs_all_64_host_arms_without_replay(self):
        result, calls, replays, marker = self.run_block('1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 64)
        self.assertEqual({line.split()[0] for line in calls},
                         {'original', 'maxresource-adapted', 'static-best', 'liveflow'})
        self.assertEqual(replays, [])
        self.assertIn('runtime=/fixed/runtime', marker)

    def test_default_still_replays_all_48_static_host_arms(self):
        result, calls, replays, marker = self.run_block(None)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 64)
        self.assertEqual(len(replays), 48)
        self.assertIsNone(marker)

    def test_invalid_option_stops_before_launch(self):
        result, calls, replays, marker = self.run_block('yes')
        self.assertEqual(result.returncode, 2)
        self.assertEqual(calls, [])
        self.assertEqual(replays, [])


if __name__ == '__main__':
    unittest.main()
