from setuptools import find_packages, setup

package_name = 'mani_capture'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/mani_capture.launch.py',
                                               'launch/mani_postprocess.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bozznyskrtt',
    maintainer_email='bozznyskrtt@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # Main capture node used by mani_capture.launch.py
            'j6_trigger_capture_node_rgbd_fk = mani_capture.j6_trigger_rgbd_fk:main',
        ],
    },
)
