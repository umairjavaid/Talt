"""
Integration tests for the TALT optimizer and components working together.
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import tempfile
import os
import gc
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from talt.model import SimpleCNN
from talt.optimizer import ImprovedTALTOptimizer
import shutil
import psutil
import time

class TestTALTIntegration(unittest.TestCase):
    """Integration tests for TALT optimizer."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Set up devices
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create a model
        self.model = SimpleCNN(num_classes=10).to(self.device)
        
        # Create dummy dataset
        self.batch_size = 32
        x = torch.randn(100, 3, 32, 32)
        y = torch.randint(0, 10, (100,))
        self.dataset = TensorDataset(x, y)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir)

    def test_complete_training_cycle(self):
        """Test a complete training cycle with the optimizer."""
        # Create base optimizer factory
        base_optimizer = lambda params, lr: torch.optim.SGD(
            params, lr=lr, momentum=0.9, weight_decay=1e-4
        )
        
        # Create TALT optimizer
        optimizer = ImprovedTALTOptimizer(
            model=self.model,
            base_optimizer=base_optimizer,
            lr=0.01,
            projection_dim=16,
            memory_size=5,
            update_interval=10,
            valley_strength=0.1,
            smoothing_factor=0.5,
            device=self.device
        )
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Train for a few epochs
        num_epochs = 2
        
        for epoch in range(num_epochs):
            for batch_idx, (inputs, labels) in enumerate(self.dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Reset gradients
                optimizer.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                optimizer.optimizer.step()
                
                # Call topology update occasionally
                if batch_idx % 10 == 0:
                    optimizer._update_topology_async()
                
                # Break early to keep test runtime reasonable
                if batch_idx >= 2:
                    break
        
        # Clean up properly
        optimizer.shutdown()
        gc.collect()
        
        # Test passed if we reached this point without errors
        self.assertTrue(True)

    def test_device_compatibility(self):
        """Test compatibility across CPU/CUDA scenarios."""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for device compatibility testing")
        
        # Test on CPU
        cpu_device = torch.device("cpu")
        cpu_model = SimpleCNN(num_classes=10).to(cpu_device)
        
        base_optimizer_cpu = lambda params, lr: torch.optim.SGD(
            params, lr=lr, momentum=0.9, weight_decay=1e-4
        )
        
        cpu_optimizer = ImprovedTALTOptimizer(
            model=cpu_model,
            base_optimizer=base_optimizer_cpu,
            lr=0.01,
            device=cpu_device
        )
        
        # Test on CUDA
        cuda_device = torch.device("cuda")
        cuda_model = SimpleCNN(num_classes=10).to(cuda_device)
        
        base_optimizer_cuda = lambda params, lr: torch.optim.SGD(
            params, lr=lr, momentum=0.9, weight_decay=1e-4
        )
        
        cuda_optimizer = ImprovedTALTOptimizer(
            model=cuda_model,
            base_optimizer=base_optimizer_cuda,
            lr=0.01,
            device=cuda_device
        )
        
        # Run on CPU
        criterion = nn.CrossEntropyLoss()
        x_cpu = torch.randn(4, 3, 32, 32).to(cpu_device)
        y_cpu = torch.randint(0, 10, (4,)).to(cpu_device)
        
        loss_fn_cpu = lambda pred, target: criterion(pred, target)
        
        cpu_passed = True
        try:
            cpu_optimizer.step(loss_fn_cpu, x_cpu, y_cpu)
        except Exception:
            cpu_passed = False
        
        # Run on CUDA
        x_cuda = torch.randn(4, 3, 32, 32).to(cuda_device)
        y_cuda = torch.randint(0, 10, (4,)).to(cuda_device)
        
        loss_fn_cuda = lambda pred, target: criterion(pred, target)
        
        cuda_passed = True
        try:
            cuda_optimizer.step(loss_fn_cuda, x_cuda, y_cuda)
        except Exception:
            cuda_passed = False
        
        # Clean up
        cpu_optimizer.shutdown()
        cuda_optimizer.shutdown()
        
        self.assertTrue(cpu_passed, "CPU device compatibility failed")
        self.assertTrue(cuda_passed, "CUDA device compatibility failed")

    def test_memory_leak_long_training(self):
        """Test for memory leaks during longer training runs."""
        # Only run on CUDA since we're testing GPU memory
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for memory leak testing")
        
        # Create base optimizer factory
        base_optimizer = lambda params, lr: torch.optim.SGD(
            params, lr=lr, momentum=0.9, weight_decay=1e-4
        )
        
        # Create TALT optimizer
        optimizer = ImprovedTALTOptimizer(
            model=self.model,
            base_optimizer=base_optimizer,
            lr=0.01,
            update_interval=3,  # Frequent updates to stress test
            device=self.device
        )
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Monitor memory
        torch.cuda.empty_cache()
        gc.collect()
        initial_memory = torch.cuda.memory_allocated()
        
        # Run for many iterations to spot leaks
        iterations = 20
        memory_samples = []
        
        for _ in range(iterations):
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Reset gradients
                optimizer.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                optimizer.optimizer.step()
                
                # Record memory usage
                if _ % 5 == 0:
                    memory_samples.append(torch.cuda.memory_allocated())
                
                break  # Just one batch per "epoch"
        
        # Cleanup
        optimizer.shutdown()
        gc.collect()
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        
        # Check for unbounded growth
        # Memory might fluctuate, but should not consistently grow
        memory_trend = np.polyfit(np.arange(len(memory_samples)), memory_samples, 1)[0]
        
        self.assertLess(memory_trend / initial_memory, 0.01, 
                       "Memory usage shows consistent growth, potential leak")
        
        # Final memory should be close to initial
        memory_ratio = float(final_memory) / max(initial_memory, 1)
        self.assertLess(memory_ratio, 1.5, 
                       "Memory significantly higher after test")


if __name__ == "__main__":
    unittest.main()
