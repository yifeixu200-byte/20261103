import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


module_path = Path(__file__).resolve().parents[1] / 'agent/runners/reproduction_support.py'
spec = importlib.util.spec_from_file_location('reproduction_support', module_path)
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


class ReproductionSupportTests(unittest.TestCase):
    def test_nested_eval_seeds_are_distinct_and_training_unchanged(self):
        leaf = SimpleNamespace(seed=11, env_args={'random_seed': 11})
        train = SimpleNamespace(env_configs=[SimpleNamespace(env_configs=[leaf])])
        configs = [support.evaluation_env_config(train, i) for i in range(10)]
        leaves = [c.env_configs[0].env_configs[0] for c in configs]
        self.assertEqual(len(set(c.seed for c in leaves)), 10)
        self.assertEqual([c.seed for c in leaves], list(range(100011, 100021)))
        self.assertTrue(all(c.env_args['random_seed'] == c.seed for c in leaves))
        self.assertEqual((leaf.seed, leaf.env_args['random_seed']), (11, 11))
        again = support.evaluation_env_config(train, 0).env_configs[0].env_configs[0]
        self.assertEqual(again.seed, leaves[0].seed)

    def test_csv_is_readable_after_each_append_and_has_one_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics/evaluations.csv'
            for i in range(2):
                support.append_evaluation(path, {'steps': i * 1000, 'eval_returns': i + 0.5})
                with path.open(encoding='utf-8', newline='') as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), i + 1)
            self.assertEqual(rows[-1], {'steps': '1000', 'eval_returns': '1.5'})


if __name__ == '__main__':
    unittest.main()
