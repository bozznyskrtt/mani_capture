import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    #  Declare Launch Arguments (Command line options)

    normal_max_nn_arg = DeclareLaunchArgument(
        'normal_max_nn', default_value='30',
        description='Max neighbor points used to compute surface normals'
    )
    normal_radius_arg = DeclareLaunchArgument(
        'normal_radius', default_value='0.03',
        description='Search radius (meters) for calculating surface normals'
    )
    nso_nb_neighbors_arg = DeclareLaunchArgument(
        'nso_nb_neighbors', default_value='50',
        description='Number of neighbors to analyze for outlier distance checking'
    )
    nso_std_ratio_arg = DeclareLaunchArgument(
        'nso_std_ratio', default_value='1.0',
        description='Standard deviation multiplier threshold for noise filtering'
    )
    output_basename_arg = DeclareLaunchArgument(
        'output_basename', default_value='scan_result',
        description='Base name for the generated output file'
    )
    output_format_arg = DeclareLaunchArgument(
        'output_format', default_value='binary',
        description='File encoding format (ascii or binary)'
    )
    output_path_arg = DeclareLaunchArgument(
        'output_path',
        default_value=os.path.expanduser('~/desktop/ply_outputs'),
        description='Directory path where the PLY files will be saved'
    )
    pointcloud_topic_arg = DeclareLaunchArgument(
        'pointcloud_topic',
        default_value='/camera/depth/color/points',
        description='The ROS 2 topic name for incoming point cloud data'
    )
    pointcloud_topic_arg = DeclareLaunchArgument(
        'pointcloud_topic', default_value='/camera/depth/color/points',
        description='The ROS 2 topic name for incoming point cloud data'
    )
    poisson_density_quantile_arg = DeclareLaunchArgument(
        'poisson_density_quantile', default_value='0.05',
        description='Density threshold to filter out low-confidence mesh regions'
    )
    poisson_depth_arg = DeclareLaunchArgument(
        'poisson_depth', default_value='8',
        description='Octree depth for Poisson reconstruction (higher = sharper but slower)'
    )
    remove_statistical_outliers_arg = DeclareLaunchArgument(
        'remove_statistical_outliers', default_value='true',
        description='Enable or disable statistical outlier filtering (true/false)'
    )
    shutdown_after_save_arg = DeclareLaunchArgument(
        'shutdown_after_save', default_value='false',
        description='Terminate the node automatically once the PLY file is saved (true/false)'
    )
    start_type_description_service_arg = DeclareLaunchArgument(
        'start_type_description_service', default_value='true',
        description='Flag to enable ROS 2 topic/service type description logging (true/false)'
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Force node to synchronize with simulated clock time (true/false)'
    )
    voxel_size_arg = DeclareLaunchArgument(
        'voxel_size',
        default_value='0.01',
        description='Leaf size in meters for voxel downsampling'
    )

    # Define Node Configuration (Mapping configurations to node parameters)

    pointcloud_to_ply_node = Node(
        package='pointcloud_to_ply',
        executable='pointcloud_to_ply_node',
        name='pointcloud_to_ply_node',
        output='screen',
        parameters=[{
            'normal_max_nn': LaunchConfiguration('normal_max_nn'),
            'normal_radius': LaunchConfiguration('normal_radius'),
            'nso_nb_neighbors': LaunchConfiguration('nso_nb_neighbors'),
            'nso_std_ratio': LaunchConfiguration('nso_std_ratio'),
            'output_basename': LaunchConfiguration('output_basename'),
            'output_format': LaunchConfiguration('output_format'),
            'output_path': LaunchConfiguration('output_path'),
            'pointcloud_topic': LaunchConfiguration('pointcloud_topic'),
            'poisson_density_quantile': LaunchConfiguration('poisson_density_quantile'),
            'poisson_depth': LaunchConfiguration('poisson_depth'),
            'remove_statistical_outliers': LaunchConfiguration('remove_statistical_outliers'),
            'shutdown_after_save': LaunchConfiguration('shutdown_after_save'),
            'start_type_description_service': LaunchConfiguration('start_type_description_service'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'voxel_downsample_size': LaunchConfiguration('voxel_size'),
        }]
    )

    # Assemble and return the launch description
    return LaunchDescription([
        normal_max_nn_arg,
        normal_radius_arg,
        nso_nb_neighbors_arg,
        nso_std_ratio_arg,
        output_basename_arg,
        output_format_arg,
        output_path_arg,
        pointcloud_topic_arg,
        poisson_density_quantile_arg,
        poisson_depth_arg,
        remove_statistical_outliers_arg,
        shutdown_after_save_arg,
        start_type_description_service_arg,
        use_sim_time_arg,
        voxel_size_arg,
        pointcloud_to_ply_node
    ])