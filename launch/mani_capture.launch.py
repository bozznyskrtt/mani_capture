"""
mani_capture.launch.py

Launches the full Mani manipulation capture pipeline:
  1. HEBI arm bringup (real hardware + gripper)
  2. hebi_bringup move_group            (t=3 s)
  3. Orbbec Astra Stereo U3 camera      (t=0 s)
  4. Static TF: base_link -> camera_link (t=0 s)
  5. YOLO object detection + TF node    (t=5 s)
  6. hebi_a-2085-06g_moveit_config move_group  (t=5 s)
  7. hebi_control hebi_movers           (t=15 s, after move_group is initialised)
  8. snapshot RGBD capture node         (t=18 s)

After capturing, stop this launch (Ctrl-C) and run:
  ros2 launch snapshot mani_postprocess.launch.py

Launch args:
  outdir        Where sessions are saved  (default: /home/bozznyskrtt/pcl_ws/teddybear)
  threshold_deg Angle change to trigger a capture (default: 3.0)
  idle_flush_sec Session idle timeout in seconds (default: 10.0)
"""

import os
import glob
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    share_dir = get_package_share_directory("mani_capture")
    subprocess.run(["uv", "sync", "--project", share_dir, "--no-editable"], check=True)
    venv_site_pkgs = glob.glob(
        os.path.join(share_dir, ".venv", "lib", "python*", "site-packages")
    )
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    new_pythonpath = ":".join(venv_site_pkgs + [existing_pythonpath]).strip(":")

    # Launch arguments
    outdir_arg = DeclareLaunchArgument(
        'outdir',
        default_value=os.path.expandvars("$HOME/pcl_ws/teddybear"),
        description='Root directory where session folders are saved',
    )
    threshold_deg_arg = DeclareLaunchArgument(
        'threshold_deg',
        default_value='30.0',
        description='Joint angle change (degrees) that triggers a capture',
    )
    idle_flush_sec_arg = DeclareLaunchArgument(
        'idle_flush_sec',
        default_value='10.0',
        description='Seconds of idle before a session is closed and flushed',
    )

    outdir = LaunchConfiguration('outdir')
    threshold_deg = LaunchConfiguration('threshold_deg')
    idle_flush_sec = LaunchConfiguration('idle_flush_sec')

    # ── Package share directories ───────────────────────────────────────────────
    hebi_bringup_dir = get_package_share_directory('hebi_bringup')
    hebi_moveit_dir = get_package_share_directory('hebi_a-2085-06g_moveit_config')
    hebi_control_dir = get_package_share_directory('hebi_control')
    orbbec_dir = get_package_share_directory('orbbec_camera')

    # ── Step 1: Arm bringup (t=0 s) ────────────────────────────────────────────
    arm_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(hebi_bringup_dir, 'bringup_arm.launch.py')
        ),
        launch_arguments={
            'hebi_arm': 'A-2085-06G',
            'use_mock_hardware': 'false',
            'use_gripper': 'true',
        }.items(),
    )

    # Step 2: hebi_bringup move_group (t=3 s)
    hebi_bringup_move_group = TimerAction(
        period=3.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(hebi_bringup_dir, 'move_group.launch.py')
                ),
                launch_arguments={
                    'hebi_arm': 'A-2085-06G',
                    'use_sim_time': 'false',
                }.items(),
            )
        ],
    )

    # Step 3: Orbbec camera (t=0 s) 
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(orbbec_dir, 'launch', 'astra_stereo_u3.launch.py')
        ),
    )

    # Step 4: Static TF base_link -> camera_link (t=0 s) 
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_camera_tf',
        arguments=['0.05', '0.06', '-0.065', '0', '0', '0', 'base_link', 'camera_link'],
    )

    # Step 5: YOLO object detection + TF (t=5 s) 
    # yolo_node = TimerAction(
    #     period=5.0,
    #     actions=[
    #         Node(
    #             package='yolo_ros2',
    #             executable='object_detection_tf_node',
    #             name='object_detection_tf_node',
    #             parameters=[{
    #                 # 'target_name': 'J6_wrist3',
    #                 # 'model_path': '~/camera_data/datasets_2/runs/detect/train/weights/best.pt',    
    #             }],
    #         )
    #     ],
    # )

    # Step 5: YOLO object detection + TF
    yolo_ros_pkg_dir = get_package_share_directory('yolo_bringup')
    yolo_bringup_launch_path = os.path.join(yolo_ros_pkg_dir, 'launch', 'yolo.launch.py')
    yolo_ros_included_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(yolo_bringup_launch_path),
        launch_arguments={
            'model': "yolo26m.pt",
            'use_3d': "True",
            'device': "cuda:0",
        }.items()
    )

    rqt_color_rqt_node = Node(
            package='rqt_image_view',
            executable='rqt_image_view',
            name='rqt_image_view',
            arguments=[
                "/yolo/dbg_image",
            ]
    )

    rviz_config_dir = os.path.join(
        get_package_share_directory('mani_capture'),
        'rviz', 'mani.rviz'
    )
    rviz_node = Node(
            package='rviz2', 
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_dir],
            output='screen',
    )

    yolo_launch = TimerAction(
    period=5.0,
    actions=[
        yolo_ros_included_launch,
        rqt_color_rqt_node,
        rviz_node,
    ],
)


    # Step 6: hebi_a-2085-06g_moveit_config move_group (t=5 s) 
    hebi_moveit_move_group = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(hebi_moveit_dir, 'launch', 'move_group.launch.py')
                ),
            )
        ],
    )

    # Step 7: hebi_control moverhebi_arm_mover1s (t=15 s)
    # hebi_a-2085-06g move_group starts at t=5 s and takes ~5-8 s to initialise.
    # t=15 s gives it a comfortable margin before hebi_movers tries to connect.
    hebi_movers = TimerAction(
        period=15.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(hebi_control_dir, 'launch', 'hebi_moversK.launch.py')
                ),
            )
        ],
    )

    # Step 8: Snapshot RGBD capture node (t=18 s) 
    # Starts 3 s after hebi_movers so the movers are already sending commands.
    snapshot_node = TimerAction(
        period=18.0,
        actions=[
            Node(
                package='mani_capture',
                executable='j6_trigger_rgbd_fk.py',
                name='mani_capture',
                parameters=[{
                    'joint_name': 'J6_wrist3',
                    'threshold_deg': threshold_deg,
                    'outdir': outdir,
                    'idle_flush_sec': idle_flush_sec,
                    'rgb_topic': '/camera/color/image_raw',
                    'depth_topic': '/camera/depth/image_raw',
                    'rgb_camera_info_topic': '/camera/color/camera_info',
                    'depth_camera_info_topic': '/camera/depth/camera_info',
                    'joint_states_topic': '/joint_states',
                    'camera_frame': 'camera_depth_optical_frame',
                    'base_frame': 'base_link',
                    'ee_frame': 'end_effector_1',
                    'xacro_path': (
                        '/home/robot/hebi_ws/src/hebi_description'
                        '/urdf/kits/A-2085-06G.urdf.xacro'
                    ),
                    'sync_slop_sec': 0.03,
                    'sync_queue_size': 50,
                    'save_rgb': True,
                    'save_depth': True,
                    'save_rgbd_npy': True,
                    'save_camera_info': True,
                    'depth_is_aligned_to_rgb': False,
                }],
            )
        ],
    )

    return LaunchDescription([
        outdir_arg,
        threshold_deg_arg,
        idle_flush_sec_arg,
        arm_bringup,
        # hebi_bringup_move_group,
        camera_launch,
        static_tf,
        yolo_launch,
        hebi_moveit_move_group,
        hebi_movers,
        snapshot_node,
    ])


"""
ros2 launch mani_capture mani_capture.launch.py
ros2 launch mani_capture mani_postprocess.launch.py
"""
