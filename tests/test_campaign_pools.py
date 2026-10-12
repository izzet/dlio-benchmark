import importlib.util
import pathlib
import tempfile
import unittest

PATH = pathlib.Path(__file__).resolve().parents[1] / 'examples/liveflow/cosmoflow_independent/pool-selection.py'
spec = importlib.util.spec_from_file_location('pool_selection', PATH)
pools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pools)

class PoolSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.arms = ['original', 'maxresource-adapted', 'static-best', 'liveflow']
        self.assignments = {}
        for i, arm in enumerate(self.arms):
            root = pathlib.Path(self.tmp.name) / f'C{i}'
            (root / 'train').mkdir(parents=True)
            self.assignments[arm] = dict(pool=f'C{i}', data_root=str(root))
        self.config = dict(blocks={'1': self.assignments})

    def test_selected_arm_gets_its_pool(self):
        for arm in self.arms:
            label, root = pools.select(self.config, 1, arm)
            self.assertEqual((label, root), (self.assignments[arm]['pool'], str(pathlib.Path(self.assignments[arm]['data_root']).resolve())))

    def test_shared_physical_directory_rejected_even_with_different_labels(self):
        self.assignments['liveflow']['data_root'] = self.assignments['original']['data_root']
        with self.assertRaises(ValueError):
            pools.select(self.config, 1, 'original')

    def test_shared_label_rejected(self):
        self.assignments['liveflow']['pool'] = 'C0'
        with self.assertRaises(ValueError):
            pools.select(self.config, 1, 'original')

    def test_missing_arm_rejected(self):
        del self.assignments['liveflow']
        with self.assertRaises(ValueError):
            pools.select(self.config, 1, 'original')

    def test_missing_train_directory_rejected(self):
        pathlib.Path(self.assignments['liveflow']['data_root'], 'train').rmdir()
        with self.assertRaises(ValueError):
            pools.select(self.config, 1, 'original')

if __name__ == '__main__':
    unittest.main()
