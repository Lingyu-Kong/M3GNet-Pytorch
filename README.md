# M3GNet-Pytorch

M3GNet self-implemented with pytorch

## Environment Setup

```
conda create -n m3gnet python=3.10 -y
conda activate m3gnet
pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.0.0%2Bcu118.html
pip install torch_sparse==0.6.18 -f https://data.pyg.org/whl/torch-2.0.0%2Bcu118.html
pip install -e .
```