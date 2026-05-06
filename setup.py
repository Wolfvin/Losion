#!/usr/bin/env python3
"""
Losion — Setup script for legacy pip compatibility.
Prefer: pip install -e ".[all]"
"""

from setuptools import setup, find_packages

setup(
    name="losion",
    version="2.3.0",
    packages=find_packages(include=["losion*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "einops>=0.7.0",
    ],
)
