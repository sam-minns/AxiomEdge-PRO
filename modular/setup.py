#!/usr/bin/env python3
# =============================================================================
# AXIOM EDGE - SETUP CONFIGURATION
# =============================================================================

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Get version from __init__.py
def get_version():
    version_file = os.path.join("axiom_edge", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.0.1"

setup(
    name="axiom-edge",
    version=get_version(),
    author="AxiomEdge Team",
    author_email="contact@axiom-edge.com",
    description="AI-Powered Modular Trading Framework",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/axiom-edge/axiom-edge",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-cov>=3.0.0",
            "black>=22.6.0",
            "flake8>=5.0.0",
            "mypy>=0.971",
        ],
        "docs": [
            "sphinx>=5.1.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "gpu": [
            "torch-audio>=0.12.0",
            "torch-vision>=0.13.0",
        ],
        "advanced": [
            "tensorflow>=2.9.0",
            "keras>=2.9.0",
            "prophet>=1.1.0",
        ],
        "brokers": [
            "oandapyV20>=0.6.3",
            "ib-insync>=0.9.70",
            "ccxt>=2.0.0",
        ],
        "data": [
            "quandl>=3.7.0",
            "fredapi>=0.5.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "axiom-edge=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "axiom_edge": [
            "configs/*.json",
            "configs/*.yaml",
            "templates/*.txt",
        ],
    },
    keywords=[
        "trading",
        "finance",
        "machine-learning",
        "artificial-intelligence",
        "backtesting",
        "algorithmic-trading",
        "quantitative-finance",
        "feature-engineering",
        "time-series",
        "genetic-algorithm"
    ],
    project_urls={
        "Bug Reports": "https://github.com/axiom-edge/axiom-edge/issues",
        "Source": "https://github.com/axiom-edge/axiom-edge",
        "Documentation": "https://axiom-edge.readthedocs.io/",
    },
)
