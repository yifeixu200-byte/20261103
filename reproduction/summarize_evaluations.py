"""Aggregate final evaluation CSV rows across seeds (not individual episodes)."""
import argparse
import csv
import json
import statistics
from pathlib import Path

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('files', nargs='+', type=Path)
    p.add_argument('--value-column', default='eval_returns')
    p.add_argument('--step-column', default='steps')
    args = p.parse_args()
    records = []
    for path in args.files:
        with path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError('Empty evaluation CSV: ' + str(path))
        final = max(rows, key=lambda row: int(row[args.step_column]))
        records.append({'file': str(path), 'steps': int(final[args.step_column]),
                        'return': float(final[args.value_column])})
    values = [row['return'] for row in records]
    print(json.dumps({'n_runs': len(values), 'mean_final_return': statistics.mean(values),
                      'sample_std_ddof1': statistics.stdev(values) if len(values)>1 else None,
                      'runs': records, 'warning': 'Supply one run per independent seed of the SAME task/profile.'}, indent=2))
