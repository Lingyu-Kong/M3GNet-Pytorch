from setuptools import setup, find_packages

setup(
    name='m3gnet_torch',
    version='0.1',
    packages=find_packages(include=["m3gnet_torch", "m3gnet_torch.*"]),
    install_requires=[
        "numpy",
        "torch==2.0",
        "torch_scatter",
        "torch_sparse",
        "torch_geometric",
        "tqdm",
        "ase",
        "torch_ema",
        "torchmetrics",
    ],
)