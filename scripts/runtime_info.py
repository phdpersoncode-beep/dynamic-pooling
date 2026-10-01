"""Record CPU/build context for backend-dependent numerical regressions."""
import json
import platform
from pathlib import Path
import torch

if __name__ == '__main__':
    cpuinfo = Path('/proc/cpuinfo')
    details = {}
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if ':' in line:
                key, value = (s.strip() for s in line.split(':', 1))
                if key in ('model name', 'flags'):
                    details.setdefault(key, value)
    print(json.dumps({'python': platform.python_version(), 'platform': platform.platform(),
                      'torch': str(torch.__version__), 'cpu': details,
                      'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                      'mkldnn': torch.backends.mkldnn.enabled, 'torch_build': torch.__config__.show()}, indent=2))
