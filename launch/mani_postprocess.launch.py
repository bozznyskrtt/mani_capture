"""
mani_postprocess.launch.py

Runs the four-stage post-processing pipeline on a captured session:
  7.  depth_crop.py  --indir <session_path> --outdir <session_path>/crop
  8.  refinement.py  <session_path>
  9.  filter.py      <session_path>   (runs after refinement finishes)
  10. tsdf_fuse_from_npy.py <session_path>  (runs after filter finishes)

The session path is determined automatically as the most-recently-modified
session_* folder inside `outdir`, or you can pass it explicitly.

Launch args:
  outdir        Root capture directory  (default: /home/bozznyskrtt/pcl_ws/teddybear)
  session_path  Explicit session folder (default: '' -> auto-detect latest)
"""

import glob
import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration


# Absolute paths to the post-processing scripts (source tree locations)
_SNAPSHOT_SRC = os.path.expandvars('$HOME/hebi_ws/src/mani_capture/mani_capture')
_SCRIPT_DIR = os.path.join(_SNAPSHOT_SRC, '3d model')
_TSDF_SCRIPT = os.path.join(_SNAPSHOT_SRC, 'tsdf_fuse_from_npy_crop.py')


def _find_latest_session(outdir: str) -> str:
    """Return the most recently modified session_* directory in outdir."""
    pattern = os.path.join(outdir, 'session_*')
    sessions = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    if not sessions:
        raise RuntimeError(
            f"No session_* directories found in '{outdir}'. "
            "Did you run mani_capture.launch.py first?"
        )
    latest = max(sessions, key=os.path.getmtime)
    return latest


def _create_pipeline(context, *args, **kwargs):
    outdir = LaunchConfiguration('outdir').perform(context)
    session_path = LaunchConfiguration('session_path').perform(context)

    if not session_path:
        session_path = _find_latest_session(outdir)

    depth_dir = os.path.join(session_path, 'depth')
    crop_dir = os.path.join(session_path, 'crop', 'depth')
    print(f"[mani_postprocess] Session: {session_path}")
    print(f"[mani_postprocess] Depth dir: {depth_dir}")
    print(f"[mani_postprocess] Crop dir: {crop_dir}")

    depth_crop_proc = ExecuteProcess(
        cmd=[
            'python3', os.path.join(_SNAPSHOT_SRC, 'depth_crop.py'),
            '--indir', depth_dir,
            '--outdir', crop_dir,
            '--min_depth', '0',
            '--max_depth', '500',
        ],
        name='depth_crop',
        output='screen',
    )

    refinement_proc = ExecuteProcess(
        cmd=['python3', os.path.join(_SCRIPT_DIR, 'refinement.py'), session_path],
        name='refinement',
        output='screen',
    )

    filter_proc = ExecuteProcess(
        cmd=['python3', os.path.join(_SCRIPT_DIR, 'filter.py'), session_path],
        name='filter',
        output='screen',
    )

    tsdf_proc = ExecuteProcess(
        cmd=['python3', _TSDF_SCRIPT, session_path],
        name='tsdf_fuse',
        output='screen',
    )

    # Chain: depth_crop -> refinement -> filter -> tsdf_fuse
    return [
        depth_crop_proc,
        RegisterEventHandler(
            OnProcessExit(
                target_action=depth_crop_proc,
                on_exit=[refinement_proc],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=refinement_proc,
                on_exit=[filter_proc],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=filter_proc,
                on_exit=[tsdf_proc],
            )
        ),
    ]


def generate_launch_description():
    outdir_arg = DeclareLaunchArgument(
        'outdir',
        default_value=os.path.expandvars("$HOME/pcl_ws/teddybear"),
        description='Root directory where session_* folders live',
    )
    session_path_arg = DeclareLaunchArgument(
        'session_path',
        default_value='',
        description=(
            'Explicit path to a session folder. '
            'Leave empty to auto-detect the latest session in outdir.'
        ),
    )

    return LaunchDescription([
        outdir_arg,
        session_path_arg,
        OpaqueFunction(function=_create_pipeline),
    ])
