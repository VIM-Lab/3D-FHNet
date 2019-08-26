# 3D-FHNet
This repository contains the source codes for the paper 3D -FHNet: Three-Dimensional Fusion Hierarchical Reconstruction Method for Any Number of Views

# Overview
The research field of reconstructing 3D models from 2D images is becoming more and more important. Existing methods typically perform single-view reconstruction or multi-view reconstruction utilizing the properties of recurrent neural networks. Due to the self-occlusion of the model and the special nature of the recurrent neural network, these methods have some problems. We propose a novel three-dimensional fusion hierarchical reconstruction method that utilizes a multi-view feature combination method and a hierarchical prediction strategy to unify the single view and any number of multiple views 3D reconstructions. Experiments show that our method can effectively combine features between different views and obtain better reconstruction results than the baseline, especially in the thin parts of the object.

# Dataset
We train and validate our model on the ShapeNet dataset. The code of data preprocessing is released and the guideline is coming soon.

# Training
*To train our network, preprocesses the dataset first:
```
python preprocess_dataset.py
```

*To start training , run:
```
python run.py
```
