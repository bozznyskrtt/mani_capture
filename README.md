# Mani
This is package for Mani.
Design to grab up the object, rotate it and captute depth image to make a 3d reconstriction out of the depth image.

## Requirements List
You can find it by open requirements.txt or heres are the list
* matplotlib==3.10.8
* numpy==1.26.4
* open3d==0.19.0
* opencv-python==4.11.0.86
* Pillow==12.1.1
* pyrender==0.1.45
* PySide6==6.11.0
* PyYAML==5.4.1
* scikit-learn==1.7.2
* scipy==1.8.0
* torch==2.10.0
* trimesh==4.11.3
* tqdm==4.67.3

## 🛠️ Setting up your Workspace
Run the following command to set up and Download the package.
```bash
# Go to the workspace directory
cd ~/hebi_ws/src

#Install the package
git clone https://github.com/bozznyskrtt/Mani_capture.git

#if meshlab is not installed run this
sudo apt install meshlab
```

## Install dependencies
```bash
pip install -r requirements.txt
```
If you're facing the version conflict problem, check your pip version. I'm using python 3.10 and pip 22.0.2

```bash
python3 -m pip install --upgrade pip==22.0.2
```

