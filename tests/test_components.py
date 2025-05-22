"""
Unit tests for TALT components.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from talt.components import PowerIteration, IncrementalCovariance, ValleyDetector

class TestPowerIteration(unittest.TestCase):
    """Test the PowerIteration component for eigendecomposition."""
    
    def test_matrix_not_modified(self):
        """Test that the original matrix is not modified."""
        # Create a symmetric test matrix
        n = 5
        matrix = torch.randn(n, n)
        matrix = torch.matmul(matrix, matrix.t())  # Make it symmetric
        
        # Keep a copy of the original matrix
        original = matrix.clone()
        
        # Run power iteration
        power_iter = PowerIteration(max_iter=10)
        eigenvalues, eigenvectors = power_iter.compute_eigenpairs(matrix, k=2)
        
        # Check if matrix was not modified
        self.assertTrue(torch.allclose(matrix, original, atol=1e-5), 
                       "Original matrix was modified during eigendecomposition")

    def test_eigensystem_quality(self):
        """Test that the eigenvalues and eigenvectors are correct."""
        # Create a matrix with known eigenvalues
        n = 5
        D = torch.diag(torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5]))
        Q, _ = torch.linalg.qr(torch.randn(n, n))
        matrix = Q @ D @ Q.t()
        
        # Get eigenpairs using power iteration
        power_iter = PowerIteration(max_iter=20, tol=1e-8)
        eigenvalues, eigenvectors = power_iter.compute_eigenpairs(matrix, k=3)
        
        # Check eigenvalue order (should be from largest to smallest)
        self.assertGreaterEqual(eigenvalues[0], eigenvalues[1], "Eigenvalues not sorted correctly")
        if len(eigenvalues) > 2:
            self.assertGreaterEqual(eigenvalues[1], eigenvalues[2], "Eigenvalues not sorted correctly")
        
        # Verify Av = λv relationship for the largest eigenvalue
        principal_eigenvector = eigenvectors[:, 0]
        Av = torch.mv(matrix, principal_eigenvector)
        lambda_v = eigenvalues[0] * principal_eigenvector
        
        rel_error = torch.norm(Av - lambda_v) / torch.norm(lambda_v)
        self.assertLess(rel_error, 1e-3, "Eigenvector equation Av = λv not satisfied within tolerance")


class TestIncrementalCovariance(unittest.TestCase):
    """Test the IncrementalCovariance component."""
    
    def test_device_consistency(self):
        """Test device consistency in covariance computation."""
        dim = 4
        cov = IncrementalCovariance(dim)
        
        # Add some data points
        for _ in range(10):
            x = torch.randn(dim)
            cov.update(x)
            
        # Get covariance with regularization
        result1 = cov.get_covariance(reg=1e-5)
        
        # Ensure device is consistent when explicitly set
        if torch.cuda.is_available():
            # Move covariance to GPU
            cov.cov = cov.cov.cuda()
            cov.mean = cov.mean.cuda()
            
            # Get covariance again - should also be on GPU
            result2 = cov.get_covariance(reg=1e-5)
            
            # Check device
            self.assertEqual(result2.device.type, "cuda", 
                           "Covariance matrix not on expected device")
            
            # Check that regularization term is on the same device
            self.assertEqual(result2.device, cov.cov.device, 
                           "Regularization not on the same device as covariance")

    def test_update_efficiency(self):
        """Test that covariance update is efficient."""
        dim = 100
        cov = IncrementalCovariance(dim)
        x = torch.randn(dim)
        
        # Add a data point and benchmark
        import time
        start = time.time()
        for _ in range(100):  # Repeat to get more reliable timing
            cov.update(x)
        duration = time.time() - start
        
        # No specific threshold, but this might identify dramatic performance regressions
        self.assertLess(duration, 1.0, "Covariance update is too slow")


class TestValleyDetector(unittest.TestCase):
    """Test the ValleyDetector component."""
    
    def test_normalization_robustness(self):
        """Test gradient normalization robustness."""
        detector = ValleyDetector(window_size=3)
        
        # Test with very small gradients
        tiny_grad = torch.ones(10) * 1e-10
        detector.update(tiny_grad)
        
        # Test with zero gradients - should not raise error
        zero_grad = torch.zeros(10)
        try:
            detector.update(zero_grad)
            # If we get here, no exception was raised
            passed = True
        except Exception as e:
            passed = False
        
        self.assertTrue(passed, "Valley detector failed to handle zero gradient")


if __name__ == "__main__":
    unittest.main()
