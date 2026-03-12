import os
import shutil
import subprocess
import yaml
import torch


def setup_output_dir(base_dir: str, exp_name: str) -> str:
    """Create output directory structure for an experiment."""
    exp_dir = os.path.join(base_dir, exp_name)
    img_dir = os.path.join(exp_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    return exp_dir


def dump_config(config: dict, exp_dir: str):
    """Save experiment config to yaml."""
    path = os.path.join(exp_dir, "config.yaml")
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def dump_env(exp_dir: str):
    """Record environment info (torch/cuda/driver versions)."""
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_mb": torch.cuda.get_device_properties(0).total_memory // (1024 * 1024) if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=5)
        info["driver_version"] = result.stdout.strip()
    except Exception:
        info["driver_version"] = "unknown"

    path = os.path.join(exp_dir, "env.txt")
    with open(path, "w") as f:
        for k, v in info.items():
            f.write(f"{k}: {v}\n")
    return info


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_prompts(prompts_path: str) -> list:
    """Load prompts from text file (one per line)."""
    with open(prompts_path, "r") as f:
        return [line.strip() for line in f if line.strip()]
