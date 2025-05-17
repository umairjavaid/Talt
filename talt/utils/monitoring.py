"""Performance monitoring utilities for the TALT optimizer."""

import os
import time
import psutil
import torch
from typing import Dict, List

class Timer:
    """Simple timer for measuring execution time of code blocks."""

    def __init__(self, name: str = "Operation"):
        """
        Initialize timer.

        Args:
            name: Name of the operation being timed
        """
        self.name = name
        self.start_time = None

    def __enter__(self):
        """Start timer when entering context."""
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        """Print elapsed time when exiting context."""
        elapsed = time.time() - self.start_time
        print(f"⏱️ {self.name} took {elapsed:.4f} seconds")


def print_memory_usage(prefix: str = ""):
    """
    Print current memory usage statistics.

    Args:
        prefix: Optional prefix for the output message
    """
    # System memory (RAM)
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024)  # MB

    # GPU memory if available
    gpu_memory = ""
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
        gpu_memory = f", GPU: {gpu_allocated:.1f}MB allocated, {gpu_reserved:.1f}MB reserved"

    print(f"🔄 {prefix}Memory - RAM: {ram_usage:.1f}MB{gpu_memory}")


class PerformanceTracker:
    """Tracks performance metrics during training."""

    def __init__(self):
        """Initialize performance tracker."""
        self.timings = {
            "forward_pass": [],
            "backward_pass": [],
            "optimizer_step": [],
            "topology_update": [],
            "batch_total": []
        }
        self.memory_usage = []

    def record_timing(self, operation: str, elapsed: float):
        """
        Record timing for an operation.

        Args:
            operation: Name of the operation
            elapsed: Time taken in seconds
        """
        if operation in self.timings:
            self.timings[operation].append(elapsed)

    def record_memory(self):
        """Record current memory usage."""
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            self.memory_usage.append(gpu_allocated)
        else:
            process = psutil.Process(os.getpid())
            ram_usage = process.memory_info().rss / (1024 * 1024)  # MB
            self.memory_usage.append(ram_usage)

    def print_summary(self):
        """Print summary of performance statistics."""
        print("\n===== PERFORMANCE SUMMARY =====")

        print("⏱️ Timing Statistics (in seconds):")
        for operation, times in self.timings.items():
            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                print(f"  - {operation:15s}: avg={avg_time:.4f}, max={max_time:.4f}")

        if self.memory_usage:
            avg_memory = sum(self.memory_usage) / len(self.memory_usage)
            peak_memory = max(self.memory_usage)
            print(f"🔄 Memory Usage (MB): avg={avg_memory:.1f}, peak={peak_memory:.1f}")

        print("===============================\n")
