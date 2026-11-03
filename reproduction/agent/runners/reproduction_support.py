"""Simulator-independent reproducibility helpers."""
import csv
import os
from copy import deepcopy
from pathlib import Path


def evaluation_env_config(env_config, worker_index):
    """Deep-copy a curriculum and offset all leaf seeds deterministically."""
    result = deepcopy(env_config)
    offset = 100_000 + int(worker_index)

    def update(config):
        children = getattr(config, 'env_configs', None)
        if children is not None:
            for child in children:
                update(child)
            return
        if hasattr(config, 'seed') and config.seed is not None:
            config.seed = (int(config.seed) + offset) % (2 ** 32)
        if hasattr(config, 'env_args') and 'random_seed' in config.env_args:
            seed = config.env_args['random_seed']
            if seed is not None:
                config.env_args['random_seed'] = (int(seed) + offset) % (2 ** 32)

    update(result)
    return result


def append_evaluation(path, row):
    """Persist each evaluation immediately, including when training later fails."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
