"""Setup configuration for Time Series RAG package."""

import os
from setuptools import setup, find_packages

# Get package version
def get_version():
    """Get version from __init__.py."""
    with open(os.path.join("src", "timeseries_rag", "__init__.py")) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.strip().split("=")[1].strip(" '\"")
    raise RuntimeError("Version not found")

# Get long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Get install requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="timeseries_rag",
    version=get_version(),
    author="Minesh A. Jethva",
    author_email="minesh.1291@gmail.com",
    description="Time series similarity search and retrieval augmented generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/minesh1291/timeseries-rag",
    project_urls={
        "Bug Tracker": "https://github.com/minesh1291/timeseries-rag/issues",
        "Documentation": "https://minesh1291.github.io/timeseries-rag/",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "timeseries-rag=timeseries_rag.api:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)