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

#build 
colcon build --symlink-install
source install/setup.bash
```

## Install dependencies
```bash
pip install -r requirements.txt
```
If you're facing the version conflict problem, check your pip version. I'm using python 3.10 and pip 22.0.2

```bash
#check your pip version
pip --version

#install pip 22.0.2
python3 -m pip install --upgrade pip==22.0.2
```
## 🙂 Introduction

There're 2 main commands you'll be using.

1) this command makes Mani move and capture depth image.

```bash
ros2 launch snapshot mani_capture.launch.py
``` 
2) This command does all the segmentation, prediction, subtraction,clustering, data cleaning and then reconstruct the 3d shapes using TSDF algorithm.

```bash
ros2 launch snapshot mani_postprocess.launch.py
```

## ⚙️ Configuration

There might be a different between my workspace and yours, I'll show where to edit.

1) /launch/mani_capture.launch.py
![png](/media/capture1.png)
Change out_arg default value to your file savin directory.

![png](/media/capture2.png)
Adjust x y z value if your camera position is different.

![png](/media/capture3.png)
change the xacro_path to your robot .urdf.xacro file path.

2) /launch/mani_postprocess.launch.py
![png](/media/postprocess1.png)
Change the absolute path for your this cloned repository.

![png](/media/postprocess2.png)
Change to your depth image saved path.




