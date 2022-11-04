# Manual Installation with Miniconda

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) and create an environment with a python version that matches your blender python interpreter.
2. Install the dependencies with pip:
```
pip install -r requirements.txt
```
3. Optional: Install [PyTorch](https://pytorch.org) to enable the semantic segmentation functionality. Thanks to yanx27 et al. for creating and maintaining the [PointNet++ repository](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)
4. Compile the partition algorithms in [https://github.com/mati3230/3DPartitionAlgorithms](https://github.com/mati3230/3DPartitionAlgorithms) and copy the resulting dynamic libraries into this project.
4. Zip this folder (not only the contents).
5. Import the add-on in blender by importing the zip file. 