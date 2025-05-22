"""
Integration tests for the TALT optimizer and components working together.
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import tempfile
import os
import shutil
import sys

# Ensure that the parent directory is in the Python path 
# for proper imports during testing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from talt.optimizer import ImprovedTALTOptimizer, TALTOptimizer
from talt.visualization import ImprovedTALTVisualizer
from talt.model import SimpleCNN
from talt.train import get_loaders, train_and_evaluate_improved, evaluate

class SimpleModel(nn.Module):
    """A very simple model for testing"""
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class TestTALTIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment before each test method"""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Set the device for testing - use CPU for consistent testing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create temporary directory for outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple dataset
        self.input_dim = 10
        self.output_dim = 2
        self.batch_size = 4
        self.num_samples = 20
        
        # Create random data
        X = torch.randn(self.num_samples, self.input_dim)
        y = torch.randint(0, self.output_dim, (self.num_samples,))
        
        dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Create model
        self.model = SimpleModel(self.input_dim, 20, self.output_dim).to(self.device)
        
        # Define loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def tearDown(self):
        """Clean up after each test method"""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_talt_optimizer_integration(self):
        """Test that ImprovedTALTOptimizer works end-to-end"""
        # Create the TALT optimizer
        talt_opt = ImprovedTALTOptimizer(
            model=self.model,
            base_optimizer=lambda params, lr: optim.SGD(params, lr=lr, momentum=0.9),
            lr=0.01,
            projection_dim=8,  # Small for testing
            memory_size=5,
            update_interval=3,  # Small for testing
            valley_strength=0.2,
            smoothing_factor=0.3,
            device=self.device
        )
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(talt_opt, step_size=2, gamma=0.9)
        
        # Train for a few iterations
        self.model.train()
        losses = []
        
        for epoch in range(3):  # Just 3 epochs for testing
            epoch_loss = 0.0
            
            for inputs, targets in self.dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward and backward pass using TALT optimizer
                loss, outputs = talt_opt.step(self.criterion, inputs, targets)
                
                epoch_loss += loss
                
            # Update learning rate
            scheduler.step()
            
            # Store loss for assertion
            losses.append(epoch_loss)
            
            # Verify that the optimizer is collecting visualization data
            self.assertIn('loss_values', talt_opt._visualization_data)
            self.assertGreater(len(talt_opt._visualization_data['loss_values']), 0)
        
        # Verify that training loss generally decreases
        self.assertLessEqual(losses[-1], losses[0] * 1.5)  # Allow some fluctuation
        
        # Clean up
        talt_opt.shutdown()
    
    def test_visualization_integration(self):
        """Test that visualization components work with optimizer"""
        # Create optimizer
        talt_opt = ImprovedTALTOptimizer(
            model=self.model,
            base_optimizer=lambda params, lr: optim.SGD(params, lr=lr, momentum=0.9),
            lr=0.01,
            projection_dim=8,  # Small for testing
            memory_size=3,
            update_interval=2,  # Small for testing
            max_visualization_points=100,
            device=self.device
        )
        
        # Create visualizer
        visualizer = ImprovedTALTVisualizer(output_dir=self.temp_dir)
        
        # Train for a few steps
        self.model.train()
        
        for _ in range(2):  # Just 2 epochs for testing
            for inputs, targets in self.dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward and backward pass
                loss, outputs = talt_opt.step(self.criterion, inputs, targets)
                
                # Collect data for visualization
                visualizer.add_data({
                    'loss_values': talt_opt._visualization_data['loss_values'],
                    'valley_detections': talt_opt._visualization_data['valley_detections'],
                    'bifurcations': talt_opt.bifurcations,
                    'gradient_stats': {
                        name: list(stats) for name, stats in 
                        talt_opt._visualization_data['gradient_stats'].items()
                    }
                })
        
        # Generate visualizations - should not raise exceptions
        try:
            visualizer.visualize_loss_trajectory(save_path='loss.png', show=False)
            visualizer.visualize_gradient_norm_history(save_path='grad_norm.png', show=False)
            
            # Check that files were created
            self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'loss.png')))
            self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'grad_norm.png')))
            
            # Generate report
            visualizer.generate_report(experiment_name='Test_Experiment')
            
            # Check that the report directory was created
            self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'report')))
            
        except Exception as e:
            self.fail(f"Visualization raised exception: {e}")
        
        # Clean up
        talt_opt.shutdown()

    def test_train_and_evaluate_function(self):
        """Test the train_and_evaluate_improved function"""
        # Create a temporary CNN model for CIFAR10
        model = SimpleCNN(3, 10).to(self.device)  # 3 channels, 10 classes
        
        # Create simplified dataloaders with dummy data
        # In a real test, you would use a real dataset, but we just need the structure here
        dummy_data = torch.randn(16, 3, 32, 32)  # 16 samples, 3 channels, 32x32 images
        dummy_targets = torch.randint(0, 10, (16,))
        dummy_dataset = TensorDataset(dummy_data, dummy_targets)
        train_loader = DataLoader(dummy_dataset, batch_size=4)
        test_loader = DataLoader(dummy_dataset, batch_size=4)
        
        # Create visualization directory
        vis_dir = os.path.join(self.temp_dir, 'visualizations')
        
        # Run training with TALT optimizer - this should not raise any exceptions
        try:
            results = train_and_evaluate_improved(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                epochs=1,  # Just 1 epoch for testing
                learning_rate=0.01,
                use_improved_talt=True,
                device=self.device,
                projection_dim=8,
                update_interval=3,
                memory_size=3,
                valley_strength=0.2,
                smoothing_factor=0.3,
                visualization_dir=vis_dir,
                experiment_name="Test_Run"
            )
            
            # Verify that results contain expected keys
            self.assertIn('train_loss', results)
            self.assertIn('test_acc', results)
            self.assertIn('final_test_acc', results)
            
        except Exception as e:
            self.fail(f"train_and_evaluate_improved raised exception: {e}")

if __name__ == '__main__':
    unittest.main()
