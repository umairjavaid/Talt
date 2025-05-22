"""
Setup script for installing the TALT package.
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define package requirements
requirements = [
    "torch>=1.7.0",
    "torchvision>=0.8.0",
    "numpy>=1.19.0",
    "matplotlib>=3.3.0",
    "seaborn>=0.11.0",
    "datasets>=1.5.0",
    "transformers>=4.5.0",
    "huggingface-hub>=0.0.12",
    "optuna>=2.8.0",
]

setup(
    name="talt",
    version="0.1.0",
    author="TALT Team",
    description="TALT Optimization Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/talt",
    project_urls={
        "Bug Tracker": "https://github.com/example/talt/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=requirements,
)
