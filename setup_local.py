#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple setup script to make talt_evaluation importable as a local package.
"""

import os
import sys
from pathlib import Path

def setup_local_imports():
    """Add necessary paths to enable local imports."""
    project_root = Path(__file__).parent
    talt_evaluation_path = project_root / "talt_evaluation"
    
    # Add to Python path
    paths_to_add = [str(project_root), str(talt_evaluation_path)]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    print(f"Added to Python path:")
    for path in paths_to_add:
        print(f"  {path}")

if __name__ == "__main__":
    setup_local_imports()
    print("Local import setup complete. You can now import talt_evaluation modules.")
