This is a repository for the code implemented during my MSc thesis : "From 2D Dissection Photography to 3D Neuropathology: A study on Machine Learning-based reconstruction methods". 
Dealing with the task of 2d-to-3D volume reconstructions using Diffusion Bridge Models.

### Installation 
To install all the packages used in this repository run:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
The data preprocessing to generate the synthetic dissection brains from MRI scans is under ``` /datasets/photosynth ```. Introducing illumination artifacts and nonlinear deformations.
Training scripts, and diffusion model functions are under ```/ddbm/```,   

The code is adapted from the repository of the paper "Diffusion Bridge Implicit Models" https://github.com/thu-ml/DiffusionBridge

To install the packages run: 
