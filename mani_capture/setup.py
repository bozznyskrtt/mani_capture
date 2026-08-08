import os
from glob import glob
from pathlib import Path
from setuptools import find_packages, setup

package_name = 'mani_capture'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ("share/" + package_name, ["../pyproject.toml"]),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bozznyskrtt',
    maintainer_email='bozznyskrtt@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Main capture node used by mani_capture.launch.py
            'j6_trigger_capture_node_rgbd_fk = mani_capture.j6_trigger_rgbd_fk:main',
        ],
    },
)
