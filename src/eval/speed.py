"""
Speed benchmarking with proper warmup and GPU timing.

Follows the plan spec:
  - Warmup: 5 runs (not counted)
  - Measurement: 30 runs
  - Records wall-clock (time.perf_counter) + GPU time (cuda.Event)
"""

import time
import csv
import os
import torch
import numpy as np


class SpeedBenchmark:
    """Standardized speed measurement for generation experiments."""

    def __init__(self, warmup_runs: int = 5, measure_runs: int = 30):
        self.warmup_runs = warmup_runs
        self.measure_runs = measure_runs

    def benchmark(self, generate_fn, **kwargs) -> dict:
        """
        Run speed benchmark on a generation function.

        Args:
            generate_fn: callable that performs one generation.
            **kwargs: passed to generate_fn.

        Returns:
            dict with wall_time and gpu_time statistics.
        """
        # Warmup
        for _ in range(self.warmup_runs):
            generate_fn(**kwargs)
        torch.cuda.synchronize()

        # Measurement
        wall_times = []
        gpu_times = []

        for _ in range(self.measure_runs):
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_wall = time.perf_counter()
            start_event.record()

            generate_fn(**kwargs)

            end_event.record()
            end_wall = time.perf_counter()

            torch.cuda.synchronize()

            wall_times.append(end_wall - start_wall)
            gpu_times.append(start_event.elapsed_time(end_event) / 1000.0)  # ms -> s

        wall_times = np.array(wall_times)
        gpu_times = np.array(gpu_times)

        return {
            "wall_mean": float(wall_times.mean()),
            "wall_std": float(wall_times.std()),
            "wall_median": float(np.median(wall_times)),
            "wall_min": float(wall_times.min()),
            "wall_max": float(wall_times.max()),
            "gpu_mean": float(gpu_times.mean()),
            "gpu_std": float(gpu_times.std()),
            "gpu_median": float(np.median(gpu_times)),
            "warmup_runs": self.warmup_runs,
            "measure_runs": self.measure_runs,
        }

    @staticmethod
    def save_csv(results: dict, path: str):
        """Save speed results to CSV."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results.keys())
            writer.writeheader()
            writer.writerow(results)


def setup_torch_determinism(cudnn_benchmark: bool = False, matmul_precision: str = "high"):
    """Apply standardized torch settings for benchmarking."""
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.set_float32_matmul_precision(matmul_precision)
