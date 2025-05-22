#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test runner for TALT optimizer tests.
"""

import unittest
import os
import sys
import argparse
from pathlib import Path

def main():
    """Run the test suite."""
    parser = argparse.ArgumentParser(description="Run TALT optimizer tests")
    parser.add_argument(
        "--test-type", 
        choices=["unit", "integration", "all"], 
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--device", 
        choices=["cpu", "cuda", "auto"], 
        default="auto",
        help="Device to run tests on"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Verbose test output"
    )
    args = parser.parse_args()
    
    # Set device environment variable if specified
    if args.device != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.device == "cpu" else "0"
        
    # Get current directory
    base_dir = Path(__file__).resolve().parent
        
    # Discover tests based on type
    if args.test_type == "unit" or args.test_type == "all":
        unit_tests = unittest.defaultTestLoader.discover(
            start_dir=base_dir, 
            pattern="test_[!integration]*.py"
        )
        
    if args.test_type == "integration" or args.test_type == "all":
        integration_tests = unittest.defaultTestLoader.discover(
            start_dir=base_dir, 
            pattern="test_integration*.py"
        )
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    if args.test_type == "unit" or args.test_type == "all":
        test_suite.addTest(unit_tests)
    if args.test_type == "integration" or args.test_type == "all":
        test_suite.addTest(integration_tests)
        
    # Run tests
    verbosity = 2 if args.verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(test_suite)
    
    # Return appropriate exit code
    sys.exit(0 if result.wasSuccessful() else 1)
    
if __name__ == "__main__":
    main()
