"""Read-only preflight for the existing shared-server RTX environment."""
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def main():
    root = (Path.home() / "nhl_project").resolve()
    repo = Path(__file__).resolve().parents[1]
    errors = []
    if not repo.is_relative_to(root):
        errors.append("The checkout must be under ~/nhl_project.")
    if Path(sys.prefix).resolve() != root / "nhlpy3":
        errors.append("Activate ~/nhl_project/nhlpy3 using scripts/rtx6000d_env.sh.")
    cache_names = ("PIP_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
                   "TORCH_EXTENSIONS_DIR", "XDG_CACHE_HOME", "TMPDIR")
    paths = {key: os.environ.get(key) for key in cache_names}
    for key, value in paths.items():
        if not value or not Path(value).is_absolute() or not Path(value).resolve().is_relative_to(root):
            errors.append(f"{key} must resolve under ~/nhl_project; source the environment helper.")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        errors.append("PYTHONDONTWRITEBYTECODE must be 1; source the environment helper.")
    if errors:
        print(json.dumps({"ok": False, "python": sys.executable, "errors": errors}, indent=2))
        return 1

    # Check scope before importing modules that may initialize caches.
    import torch
    import triton
    import flash_mla_old
    import flash_mla_old.sm120 as sm120

    if Path(sm120.__file__).resolve() != repo / "flash_mla_old" / "sm120.py":
        errors.append("flash_mla_old is imported from a different checkout.")
    headers = root / "toolchains/python3.12-headers/usr/include/python3.12/Python.h"
    device = None
    if not torch.cuda.is_available():
        errors.append("No CUDA device is available to this process.")
    else:
        props = torch.cuda.get_device_properties(0)
        device = {"logical_index": 0, "name": props.name,
                  "capability": [props.major, props.minor],
                  "sm_count": props.multi_processor_count,
                  "visible_count": torch.cuda.device_count()}
        if (props.major, props.minor) != (12, 0):
            errors.append("This preflight is for the RTX capability-12.0 backend.")
    result = {
        "ok": not errors, "python": sys.executable, "python_version": sys.version.split()[0],
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "distribution_version": importlib.metadata.version("flash_mla_old"),
        "package": flash_mla_old.__file__, "rtx_source": sm120.__file__,
        "caches": paths, "local_python_header": str(headers),
        "header_present": headers.is_file(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": device, "errors": errors,
    }
    print(json.dumps(result, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
