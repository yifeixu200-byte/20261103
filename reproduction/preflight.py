"""Read-only dependency report; never installs or launches training."""
import importlib.util
import json
import platform
import sys
from pathlib import Path

def inspect() -> dict:
    packages = ['torch', 'torchvision', 'numpy', 'einops', 'ray', 'wandb',
                'gym', 'mujoco_py', 'isaacgym', 'tensorboard', 'termcolor', 'ipdb']
    report = {'python': sys.version, 'platform': platform.platform(),
              'packages': {m: importlib.util.find_spec(m) is not None for m in packages},
              'mujoco210_directory': (Path.home()/'.mujoco'/'mujoco210').exists()}
    if report['packages']['torch']:
        import torch
        report.update(torch=torch.__version__, cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            report.update(gpu=torch.cuda.get_device_name(0),
                          vram_gib=torch.cuda.get_device_properties(0).total_memory/2**30)
    report['note'] = 'Presence is not simulator or training validation.'
    return report

if __name__ == '__main__':
    print(json.dumps(inspect(), indent=2))
