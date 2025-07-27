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
    author_email="sam.minns@example.com",
    description="Professional AI-Powered Modular Trading Framework with Advanced Analytics",
    summary="Comprehensive trading framework featuring 200+ technical indicators, AI-powered strategy optimization, genetic programming, and enterprise-grade backtesting capabilities",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/sam-minns/AxiomEdge-PRO",
    license="GPL-3.0",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
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
            "pytest-asyncio>=0.21.0",
            "black>=22.6.0",
            "flake8>=5.0.0",
            "mypy>=0.971",
            "pre-commit>=2.20.0",
            "isort>=5.10.0",
        ],
        "docs": [
            "sphinx>=5.1.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.19.0",
            "myst-parser>=0.18.0",
        ],
        "gpu": [
            "torch>=1.12.0",
            "torch-audio>=0.12.0",
            "torch-vision>=0.13.0",
            "cupy-cuda11x>=11.0.0",
        ],
        "advanced": [
            "tensorflow>=2.9.0",
            "keras>=2.9.0",
            "prophet>=1.1.0",
            "xgboost>=1.6.0",
            "lightgbm>=3.3.0",
            "catboost>=1.0.0",
        ],
        "brokers": [
            "oandapyV20>=0.6.3",
            "ib-insync>=0.9.70",
            "ccxt>=2.0.0",
            "alpaca-trade-api>=2.3.0",
            "python-binance>=1.0.0",
        ],
        "data": [
            "quandl>=3.7.0",
            "fredapi>=0.5.0",
            "yfinance>=0.1.87",
            "alpha-vantage>=2.3.1",
            "polygon-api-client>=1.9.0",
        ],
        "ai": [
            "google-generativeai>=0.3.0",
            "openai>=0.27.0",
            "anthropic>=0.3.0",
        ],
        "visualization": [
            "plotly>=5.10.0",
            "bokeh>=2.4.0",
            "dash>=2.6.0",
            "streamlit>=1.12.0",
        ],
        "all": [
            "pytest>=7.1.0", "pytest-cov>=3.0.0", "pytest-asyncio>=0.21.0",
            "black>=22.6.0", "flake8>=5.0.0", "mypy>=0.971",
            "sphinx>=5.1.0", "sphinx-rtd-theme>=1.0.0",
            "torch>=1.12.0", "tensorflow>=2.9.0", "prophet>=1.1.0",
            "oandapyV20>=0.6.3", "ccxt>=2.0.0", "quandl>=3.7.0",
            "google-generativeai>=0.3.0", "plotly>=5.10.0", "streamlit>=1.12.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "axiom-edge=axiom_edge.main:main",
            "axiom-edge-framework=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "axiom_edge": [
            "configs/*.json",
            "configs/*.yaml",
            "configs/*.yml",
            "templates/*.txt",
            "templates/*.html",
            "templates/*.md",
            "data/*.csv",
            "data/*.json",
            "schemas/*.json",
            "*.md",
            "*.txt",
        ],
    },
    zip_safe=False,
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
        "Bug Reports": "https://github.com/sam-minns/AxiomEdge-PRO/issues",
        "Source": "https://github.com/sam-minns/AxiomEdge-PRO",
        "Documentation": "https://github.com/sam-minns/AxiomEdge-PRO/wiki",
        "Discussions": "https://github.com/sam-minns/AxiomEdge-PRO/discussions",
        "Releases": "https://github.com/sam-minns/AxiomEdge-PRO/releases",
    },
)
