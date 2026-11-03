"""Print the 7 tasks x 4 seeds; --execute opts into long sequential runs."""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

TASKS = [('HalfCheetah-v2', '2x3'), ('HalfCheetah-v2', '3x2'),
         ('HalfCheetah-v2', '6x1'), ('Walker2d-v2', '2x3'),
         ('Walker2d-v2', '3x2'), ('Ant-v2', '2x4'), ('Ant-v2', '4x2')]

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--profile', choices=['paper', 'official'], default='paper')
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4])
    args = p.parse_args()
    for task, split in TASKS:
        for seed in args.seeds:
            command = [sys.executable, 'train.py', '--profile', args.profile,
                       '--env', 'mamujoco', '--env_name', task, '--agent_conf', split,
                       '--n_workers', '1', '--policy_class', 'gaussian', '--seed', str(seed),
                       '--steps', '1000000', '--sample_temp', '20', '--mode', 'disabled',
                       '--state_decoder_type', '1', '--use_tensorboard']
            print(shlex.join(command), flush=True)
            if args.execute:
                subprocess.run(command, cwd=Path(__file__).resolve().parent, check=True)
