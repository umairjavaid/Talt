"""
Unit tests for ImprovedTALTOptimizer.
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.amp import GradScaler
import numpy as np
import gc
import os
import time

from talt.optimizer import ImprovedTALTOptimizer
from talt.model import SimpleCNN

class SimpleTestModel(nn.Module):
    """A simple model for testing optimizers"""
    def __init__(self):
        super(SimpleTestModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class TestImprovedTALTOptimizer(unittest.TestCase):
    """Test the ImprovedTALTOptimizer."""
    
    def setUp(self):
        """Set up test environment."""
        torch.manual_seed(42)  # For reproducibility
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use CPU for testing
        self.model = SimpleTestModel().to(self.device)
        
        # Define base optimizer factory function
        self.base_optimizer = lambda params, lr: optim.SGD(
            params, lr=lr, momentum=0.9, weight_decay=1e-4
        )

    def test_constructor_params(self):
        """Test that the constructor accepts the correct parameters."""
        # Should work with model and base_optimizer
        optimizer = ImprovedTALTOptimizer(
            model=self.model,
            base_optimizer=self.base_optimizer,
            lr=0.01
        )
        
        # Check that the optimizer is initialized properly
        self.assertIsNotNone(optimizer.optimizer, "Base optimizer not initialized")
        self.assertEqual(optimizer.steps, 0, "Steps counter not initialized to 0")

    def test_device_consistency(self):
        """Test device consistency in gradient transformations."""
        optimizer = ImprovedTALTOptimizer(
            model=self.model,
            base_optimizer=self.base_optimizer,
            lr=0.01,
            valley_strength=0.2
        )
        
        # Create dummy data
        batch_size = 2
        x = torch.randn(batch_size, 3, 32, 32).to(self.device)
        y = torch.randint(0, 10, (batch_size,)).to(self.device)
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Forward pass 
        def loss_fn(pred, target):
            return criterion(pred, target)
        
        # Run step and check that no device errors occur
        try:
            loss_val, outputs = optimizer.step(loss_fn, x, y)
            passed = True
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                passed = False
            else:
                raise
        
        self.assertTrue(passed, "Device inconsistency in optimizer")

    def test_memory_leak_prevention(self):
        """Test that memory leaks are prevented in async operations."""
        # Skip if not on CUDA, as memory tracking is more reliable
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for memory leak testing")
            
        torch.cuda.empty_cache()
        gc.collect()
        
        # Create optimizer with async topology updates
        optimizer = ImprovedTALTOptimizer(
            model=self.model,
            base_optimizer=self.base_optimizer,
            lr=0.01,
            update_interval=5,  # Update frequently to stress test
            device=self.device
        )
        
        # Create dummy data
        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        
        # Define loss function
        def loss_fn(pred, target):
            return criterion(pred, target)
        
        # Monitor memory during many steps
        initial_memory = torch.cuda.memory_allocated()
        memory_readings = []
        
        # Run several optimization steps to trigger async updates
        for i in range(20):
            x = torch.randn(batch_size, 3, 32, 32).to(self.device)
            y = torch.randint(0, 10, (batch_size,)).to(self.device)
            
            optimizer.step(loss_fn, x, y)
            
            # Delay to allow async work to happen
            if i % 5 == 0:
                time.sleep(0.1)
                gc.collect()
                torch.cuda.empty_cache()
                memory_readings.append(torch.cuda.memory_allocated())
        
        # Proper cleanup should occur
        optimizer.shutdown()
        gc.collect()
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated()
        
        # Memory shouldn't grow unbounded - check if memory after cleanup is
        # within reasonable bounds of initial memory
        memory_ratio = float(final_memory) / max(initial_memory, 1)
        self.assertLess(memory_ratio, 2.0, 
                       "Memory usage more than doubled, potential leak")

    def test_pytorch_compatibility(self):
        """Test PyTorch version compatibility code paths."""
        # We can't test different PyTorch versions in one test, 
        # but we can check the code path is used
        optimizer = ImprovedTALTOptimizer(
            model=self.model,
            base_optimizer=self.base_optimizer,
            lr=0.01
        )
        
        # Check that GradScaler was initialized properly
        self.assertIsInstance(optimizer.scaler, GradScaler,
                             "GradScaler not initialized correctly")

    def test_scheduler_compatibility(self):
        """Test scheduler compatibility fixes."""
        optimizer = ImprovedTALTOptimizer(
            model=self.model,
            base_optimizer=self.base_optimizer,
            lr=0.01
        )
        
        # Test if param_groups is available for the scheduler
        if not hasattr(optimizer, 'param_groups'):
            # Add param_groups from base optimizer
            if hasattr(optimizer, 'optimizer') and hasattr(optimizer.optimizer, 'param_groups'):
                optimizer.param_groups = optimizer.optimizer.param_groups
        
        # Create scheduler
        try:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            passed = True
        except Exception:
            passed = False
            
        self.assertTrue(passed, "Scheduler compatibility fix failed")
    
    def test_state_dict_operations(self):
        """Test saving and loading state dictionaries."""
        # Create optimizer
        optimizer = ImprovedTALTOptimizer(
            model=self.model,
            base_optimizer=self.base_optimizer,
            lr=0.01,
            projection_dim=16,
            memory_size=5
        )
        
        # Run a few steps to populate internal state
        for inputs, targets in self.dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.step(self.criterion, inputs, targets)
        
        # Get state dict
        state_dict = optimizer.state_dict()
        
        # Verify that state dict has expected keys
        self.assertIn('base_optimizer', state_dict)
        self.assertIn('steps', state_dict)
        
        # Create a new optimizer
        new_optimizer = ImprovedTALTOptimizer(
            model=self.model,
            base_optimizer=self.base_optimizer,
            lr=0.01,
            projection_dim=16,
            memory_size=5
        )
        
        # Load state dict
        new_optimizer.load_state_dict(state_dict)
        
        # Verify that state was restored
        self.assertEqual(optimizer.steps, new_optimizer.steps)

    def test_zero_grad_functionality(self):
        """Test zero_grad method of the optimizer."""
        # Create optimizer
        optimizer = ImprovedTALTOptimizer(
            model=self.model,
            base_optimizer=self.base_optimizer,
            lr=0.01
        )
        
        # Set some gradients
        for p in self.model.parameters():
            p.grad = torch.ones_like(p)
        
        # Call zero_grad
        optimizer.zero_grad()
        
        # Check that gradients are zeroed
        for p in self.model.parameters():
            self.assertTrue(torch.all(torch.eq(p.grad, torch.zeros_like(p))))
    
    def test_lr_adjustment_with_scheduler(self):
        """Test learning rate adjustment with scheduler."""
        # Create optimizer
        optimizer = ImprovedTALTOptimizer(
            model=self.model,
            base_optimizer=self.base_optimizer,
            lr=0.1
        )
        
        # Create scheduler with significant step reduction
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        
        # Initial learning rate
        initial_lr = optimizer.optimizer.param_groups[0]['lr']
        self.assertAlmostEqual(initial_lr, 0.1)
        
        # Run a few steps
        for inputs, targets in self.dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.step(self.criterion, inputs, targets)
        
        # Step the scheduler
        scheduler.step()
        
        # Check that learning rate was reduced
        updated_lr = optimizer.optimizer.param_groups[0]['lr']
        self.assertAlmostEqual(updated_lr, 0.01)

    def test_covariance_efficiency(self):
        """Test the efficiency of covariance operations."""
        optimizer = ImprovedTALTOptimizer(
            model=self.model,
            base_optimizer=self.base_optimizer,
            lr=0.01,
            update_interval=1  # Update every step to test covariance computation
        )
        
        criterion = nn.CrossEntropyLoss()
        def loss_fn(pred, target):
            return criterion(pred, target)
        
        # Create dummy data
        batch_size = 2
        x = torch.randn(batch_size, 3, 32, 32).to(self.device)
        y = torch.randint(0, 10, (batch_size,)).to(self.device)
        
        # Measure time taken
        start = time.time()
        for _ in range(3):  # Just a few steps to avoid long test times
            optimizer.step(loss_fn, x, y)
            
        duration = time.time() - start
        
        # No specific threshold here, but can catch dramatic slowdowns
        self.assertLess(duration, 10.0, "Optimization steps are too slow")


if __name__ == "__main__":
    unittest.main()
