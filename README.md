# GeoInterpGCN
A general spatial interpolation method based on graph convolution.

# Environment Prepare
```shell
# create environment
conda create -n gcn python=3.11
# install pytorch 2.4
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  pytorch-cuda=11.8 -c pytorch -c nvidia
# install pytorch geometric 
# recommend for Linux
conda install pyg -c pyg
# recommend for Windows
pip install torch_geometric
# install scipy
pip install scipy
# install scikit-learn
pip install scikit-learn
```

# Train
```shell
conda activate gcn

python src/train.py
```
