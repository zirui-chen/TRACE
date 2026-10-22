from setuptools import setup

setup(
    name="trace",
    version="1.0.0",
    description="TRACE: Premise-Faithful Supervision for Zero-Shot "
                "Knowledge Graph Reasoning",
    packages=["trace", "trace.analysis"],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "torch>=2.0",
        "torch-geometric>=2.3",
        "pyyaml>=6.0",
        "tqdm>=4.65",
    ],
)
