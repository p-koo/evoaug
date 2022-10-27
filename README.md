# EvoAug

Evolution-inspired augmentations improve deep learning for regulatory genomics" by Nicholas Keone Lee, Ziqi (Amber) Tang, Shushan Toneyan, and Peter K Koo.


This PyTorch package makes it easy to pretrain sequence-based deep learning models for regulatory genomics data with evolution-inspired data augmentations followed by a finetuning on the original, unperturbed sequence data. 

<img src="fig/augmentations.png" alt="fig" width="500"/>

<img src="fig/overview.png" alt="overview" width="500"/>



Install:

```
pip install git+https://github.com/p-koo/evoaug.git
```

Dependencies:
```
torch 1.12.1+cu113
pytorch_lightning 1.7.7
sklearn 1.0.2
scipy 1.7.3
h5py 3.1.0
pandas 1.3.5
matplotlib 3.2.2
logomaker 0.8
os
pathlib 
```
