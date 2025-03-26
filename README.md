# Torch3DR
`Torch3DR` stands for __Torch 3D Research__. This library is aimed to provide you with a set of classes, functions and visualizers (all compatible with `PyTorch`) to help you with your 3D research projects. 

> [!NOTE]  
> The library is still under development and is not yet ready for production. However, you can already use it for your research projects.


## Installation

### From PyPI
```bash
# First install PyTorch3D
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Then install Torch3DR
pip install torch3dr
```

### From Source
```bash
git clone https://github.com/amirhossein-razlighi/Torch3DR.git
cd Torch3DR
pip install -e .
```

### For Development
```bash
# Use the install_requirements.sh
sudo sh install_requirements.sh
```
This will install all the required dependencies and then, installs PyTorch3D from source.
