from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="timeseries_rag",
    version="0.1.0",
    author="OpenHands",
    author_email="minesh.1291@gmail.com",
    description="Time series similarity search and retrieval augmented generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/minesh-1291/timeseries-rag",
    project_urls={
        "Bug Tracker": "https://github.com/minesh-1291/timeseries-rag/issues",
        "Documentation": "https://minesh-1291.github.io/timeseries-rag/",
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
    install_requires=[
        "fastapi",
        "uvicorn",
        "numpy",
        "pandas",
        "faiss-cpu",
        "plotly",
        "python-multipart",
        "scikit-learn",
        "scipy",
    ],
    entry_points={
        "console_scripts": [
            "timeseries-rag=timeseries_rag.api:main",
        ],
    },
)