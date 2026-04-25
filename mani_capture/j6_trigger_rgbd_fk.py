#!/usr/bin/env python3
import math
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState, CameraInfo
from tf2_ros import Buffer, TransformException, TransformListener


def shortest_angular_distance(a: float, b: float) -> float:
    return math.atan2(math.sin(b - a), math.cos(b - a))


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def transform_from_xyz_rpy(xyz: List[float], rpy: List[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_to_rot(rpy[0], rpy[1], rpy[2])
    T[:3, 3] = np.array(xyz, dtype=np.float64)
    return T


def rot_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis / n
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.array([
        [x * x * C + c,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c    ],
    ], dtype=np.float64)


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n > 0:
        qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def tf_to_mat44(transform) -> np.ndarray:
    tr = transform.translation
    rq = transform.rotation
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_rot(float(rq.x), float(rq.y), float(rq.z), float(rq.w))
    T[:3, 3] = np.array([float(tr.x), float(tr.y), float(tr.z)], dtype=np.float64)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def parse_vec(text: Optional[str], n: int, default: Optional[List[float]] = None) -> List[float]:
    if text is None:
        if default is not None:
            return list(default)
        return [0.0] * n
    vals = [float(x) for x in text.strip().split()]
    if len(vals) != n:
        raise ValueError(f"Expected {n} values, got {vals}")
    return vals


def pretty_T(T: np.ndarray, precision: int = 6) -> str:
    rows = []
    for r in range(4):
        rows.append("[" + ", ".join(f"{float(v):.{precision}f}" for v in T[r, :]) + "]")
    return "[" + ", ".join(rows) + "]"


class URDFChainFK:
    def __init__(self, urdf_xml: str, base_link: str, ee_link: str):
        self.base_link = base_link
        self.ee_link = ee_link
        self.robot = ET.fromstring(urdf_xml)
        self.joints_by_child: Dict[str, dict] = {}
        self.all_joint_names: List[str] = []
        self._parse_joints()
        self.chain = self._build_chain(base_link, ee_link)

    def _parse_joints(self):
        for j in self.robot.findall("joint"):
            name = j.attrib["name"]
            jtype = j.attrib["type"]
            parent = j.find("parent").attrib["link"]
            child = j.find("child").attrib["link"]
            origin = j.find("origin")
            xyz = parse_vec(origin.attrib.get("xyz") if origin is not None else None, 3, [0.0, 0.0, 0.0])
            rpy = parse_vec(origin.attrib.get("rpy") if origin is not None else None, 3, [0.0, 0.0, 0.0])
            axis_node = j.find("axis")
            axis = parse_vec(axis_node.attrib.get("xyz") if axis_node is not None else None, 3, [1.0, 0.0, 0.0])
            joint_info = {
                "name": name,
                "type": jtype,
                "parent": parent,
                "child": child,
                "xyz": xyz,
                "rpy": rpy,
                "axis": axis,
            }
            self.joints_by_child[child] = joint_info
            self.all_joint_names.append(name)

    def _build_chain(self, base_link: str, ee_link: str) -> List[dict]:
        chain_rev = []
        cur = ee_link
        while cur != base_link:
            if cur not in self.joints_by_child:
                known = ", ".join(sorted(self.joints_by_child.keys())[:20])
                raise RuntimeError(
                    f"Could not trace URDF chain from ee '{ee_link}' to base '{base_link}'. "
                    f"Stuck at link '{cur}'. Known child links start with: {known}"
                )
            j = self.joints_by_child[cur]
            chain_rev.append(j)
            cur = j["parent"]
        chain = list(reversed(chain_rev))
        return chain

    def fk(self, joint_positions: Dict[str, float]) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        for j in self.chain:
            T = T @ transform_from_xyz_rpy(j["xyz"], j["rpy"])
            jtype = j["type"]
            if jtype in ("revolute", "continuous"):
                q = float(joint_positions.get(j["name"], 0.0))
                R = rot_from_axis_angle(np.array(j["axis"], dtype=np.float64), q)
                J = np.eye(4, dtype=np.float64)
                J[:3, :3] = R
                T = T @ J
            elif jtype == "prismatic":
                q = float(joint_positions.get(j["name"], 0.0))
                axis = np.array(j["axis"], dtype=np.float64)
                n = np.linalg.norm(axis)
                if n > 1e-12:
                    axis = axis / n
                J = np.eye(4, dtype=np.float64)
                J[:3, 3] = axis * q
                T = T @ J
            elif jtype == "fixed":
                pass
            else:
                raise RuntimeError(f"Unsupported joint type '{jtype}' in joint '{j['name']}'")
        return T


class JointTriggeredRGBDSessionManualFK(Node):
    def __init__(self):
        super().__init__("joint_triggered_rgbd_session_manual_fk")

        # Parameters
        self.declare_parameter("joint_name", "J6_wrist3")
        self.declare_parameter("threshold_deg", 3.0)
        self.declare_parameter("outdir", str(Path.home() / "captures_rgbd_fk"))
        self.declare_parameter("joint_states_topic", "/joint_states")

        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("rgb_camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("depth_camera_info_topic", "/camera/depth/camera_info")

        self.declare_parameter("idle_flush_sec", 10.0)

        self.declare_parameter("camera_frame", "camera_depth_optical_frame")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("ee_frame", "end_effector_1")

        self.declare_parameter("xacro_path", "")
        self.declare_parameter("xacro_args", "")
        self.declare_parameter("save_pose_npy", True)
        self.declare_parameter("save_base_to_ee_npy", True)

        self.declare_parameter("save_rgb", True)
        self.declare_parameter("save_depth", True)
        self.declare_parameter("save_rgbd_npy", True)
        self.declare_parameter("save_camera_info", True)

        self.declare_parameter("depth_is_aligned_to_rgb", False)

        self.declare_parameter("sync_slop_sec", 0.03)
        self.declare_parameter("sync_queue_size", 50)

        self.joint_name = self.get_parameter("joint_name").value
        self.threshold_deg = float(self.get_parameter("threshold_deg").value)
        self.outdir = Path(self.get_parameter("outdir").value)
        self.joint_states_topic = self.get_parameter("joint_states_topic").value

        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.rgb_camera_info_topic = self.get_parameter("rgb_camera_info_topic").value
        self.depth_camera_info_topic = self.get_parameter("depth_camera_info_topic").value

        self.idle_flush_sec = float(self.get_parameter("idle_flush_sec").value)

        self.camera_frame = self.get_parameter("camera_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.ee_frame = self.get_parameter("ee_frame").value

        self.xacro_path = self.get_parameter("xacro_path").value
        self.xacro_args = self.get_parameter("xacro_args").value
        self.save_pose_npy = bool(self.get_parameter("save_pose_npy").value)
        self.save_base_to_ee_npy = bool(self.get_parameter("save_base_to_ee_npy").value)

        self.save_rgb = bool(self.get_parameter("save_rgb").value)
        self.save_depth_flag = bool(self.get_parameter("save_depth").value)
        self.save_rgbd_npy = bool(self.get_parameter("save_rgbd_npy").value)
        self.save_camera_info = bool(self.get_parameter("save_camera_info").value)

        self.depth_is_aligned_to_rgb = bool(self.get_parameter("depth_is_aligned_to_rgb").value)

        self.sync_slop_sec = float(self.get_parameter("sync_slop_sec").value)
        self.sync_queue_size = int(self.get_parameter("sync_queue_size").value)

        if not self.xacro_path:
            raise RuntimeError("Parameter 'xacro_path' must point to your robot .urdf.xacro file")

        ensure_dir(self.outdir)
        self.bridge = CvBridge()

        self.tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.fk_solver = URDFChainFK(
            urdf_xml=self._expand_xacro(self.xacro_path, self.xacro_args),
            base_link=self.base_frame,
            ee_link=self.ee_frame,
        )
        self.get_logger().info(
            f"FK chain ready: {self.base_frame} -> {self.ee_frame}, "
            f"{len(self.fk_solver.chain)} joints in chain"
        )

        # Latest camera info cache
        self.latest_rgb_camera_info: Optional[CameraInfo] = None
        self.latest_depth_camera_info: Optional[CameraInfo] = None

        self.create_subscription(
            CameraInfo, self.rgb_camera_info_topic, self.on_rgb_camera_info, qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo, self.depth_camera_info_topic, self.on_depth_camera_info, qos_profile_sensor_data
        )

        # Message filter sync: RGB + Depth + JointState
        self.rgb_sub = message_filters.Subscriber(
            self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data
        )
        self.joint_sub = message_filters.Subscriber(
            self, JointState, self.joint_states_topic, qos_profile=qos_profile_sensor_data
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.joint_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop_sec,
            allow_headerless=False,
        )
        self.sync.registerCallback(self.on_synced)

        # Capture tracking
        self.last_saved_joint_angle_rad: Optional[float] = None
        self.last_capture_walltime: Optional[float] = None

        # Session tracking
        self.session_folder: Optional[Path] = None
        self.session_id: Optional[int] = None
        self.session_meta: List[str] = []
        self.session_seq = 0

        self.create_timer(0.5, self.check_idle)

        self.get_logger().info("Manual-FK RGBD capture session started")
        self.get_logger().info(f"Trigger joint: {self.joint_name}, threshold={self.threshold_deg} deg")
        self.get_logger().info(f"RGB topic: {self.rgb_topic}")
        self.get_logger().info(f"Depth topic: {self.depth_topic}")
        self.get_logger().info(f"Joint topic: {self.joint_states_topic}")
        self.get_logger().info(f"Frames: camera={self.camera_frame}, base={self.base_frame}, ee={self.ee_frame}")
        self.get_logger().info(f"Outdir: {self.outdir}")
        self.get_logger().info(f"Sync slop={self.sync_slop_sec}s, queue={self.sync_queue_size}")
        self.get_logger().info(f"depth_is_aligned_to_rgb={self.depth_is_aligned_to_rgb}")

    def on_rgb_camera_info(self, msg: CameraInfo):
        self.latest_rgb_camera_info = msg

    def on_depth_camera_info(self, msg: CameraInfo):
        self.latest_depth_camera_info = msg

    def _expand_xacro(self, xacro_path: str, extra_args: str) -> str:
        xacro_path = os.path.expanduser(xacro_path)
        if not os.path.exists(xacro_path):
            raise FileNotFoundError(f"xacro_path does not exist: {xacro_path}")
        cmd = ["xacro", xacro_path]
        if extra_args.strip():
            cmd.extend(extra_args.strip().split())
        self.get_logger().info(f"Expanding xacro: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"xacro failed with code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        return proc.stdout

    def start_new_session(self):
        self.session_id = self.get_clock().now().nanoseconds
        self.session_folder = self.outdir / f"session_{self.session_id}"
        ensure_dir(self.session_folder)
        ensure_dir(self.session_folder / "poses")
        ensure_dir(self.session_folder / "rgb")
        ensure_dir(self.session_folder / "depth")
        ensure_dir(self.session_folder / "rgbd")

        self.session_meta = []
        self.session_seq = 0
        self.session_meta.append(f"session_id: {self.session_id}\n")
        self.session_meta.append(f"joint_name: {self.joint_name}\n")
        self.session_meta.append(f"threshold_deg: {self.threshold_deg}\n")
        self.session_meta.append(f"rgb_topic: {self.rgb_topic}\n")
        self.session_meta.append(f"depth_topic: {self.depth_topic}\n")
        self.session_meta.append(f"joint_states_topic: {self.joint_states_topic}\n")
        self.session_meta.append(f"rgb_camera_info_topic: {self.rgb_camera_info_topic}\n")
        self.session_meta.append(f"depth_camera_info_topic: {self.depth_camera_info_topic}\n")
        self.session_meta.append(f"camera_frame: {self.camera_frame}\n")
        self.session_meta.append(f"base_frame: {self.base_frame}\n")
        self.session_meta.append(f"ee_frame: {self.ee_frame}\n")
        self.session_meta.append(f"xacro_path: {self.xacro_path}\n")
        self.session_meta.append(f"sync_slop_sec: {self.sync_slop_sec}\n")
        self.session_meta.append(f"depth_is_aligned_to_rgb: {self.depth_is_aligned_to_rgb}\n")
        self.session_meta.append("captures:\n")

        self.get_logger().info(f"New session folder: {self.session_folder}")

    def save_depth_image(self, msg: Image, path: Path):
        enc = (msg.encoding or "").lower()

        if "16uc1" in enc or enc == "mono16":
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if img.dtype != np.uint16:
                img = img.astype(np.uint16)
            cv2.imwrite(str(path), img)
            return img

        if "32fc1" in enc:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            mm = np.clip(img * 1000.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(str(path), mm)
            return mm

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if img.dtype != np.uint16:
            img = img.astype(np.uint16)
        cv2.imwrite(str(path), img)
        return img

    def image_msg_to_bgr(self, msg: Image) -> np.ndarray:
        enc = (msg.encoding or "").lower()
        if enc in ("bgr8",):
            return self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if enc in ("rgb8",):
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # fallback
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def lookup_static_cam_to_base(self) -> Tuple[np.ndarray, str]:
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                self.camera_frame,
                self.base_frame,
                Time(),
                timeout=Duration(seconds=0.5)
            )
            T = tf_to_mat44(tf_stamped.transform)
            stamp = tf_stamped.header.stamp
            stamp_str = f"{int(stamp.sec)}.{int(stamp.nanosec):09d}"
            return T, stamp_str
        except TransformException as e:
            raise RuntimeError(
                f"Could not get static transform {self.camera_frame} -> {self.base_frame}: {e}"
            )

    def camera_info_to_dict(self, msg: CameraInfo) -> dict:
        return {
            "width": int(msg.width),
            "height": int(msg.height),
            "distortion_model": str(msg.distortion_model),
            "d": np.array(msg.d, dtype=np.float64),
            "k": np.array(msg.k, dtype=np.float64).reshape(3, 3),
            "r": np.array(msg.r, dtype=np.float64).reshape(3, 3),
            "p": np.array(msg.p, dtype=np.float64).reshape(3, 4),
            "frame_id": str(msg.header.frame_id),
            "stamp_sec": int(msg.header.stamp.sec),
            "stamp_nanosec": int(msg.header.stamp.nanosec),
        }

    def on_synced(self, rgb_msg: Image, depth_msg: Image, joint_msg: JointState):
        try:
            idx_trigger = joint_msg.name.index(self.joint_name)
        except ValueError:
            self.get_logger().warn(f"Trigger joint '{self.joint_name}' not found in joint_states")
            return

        if idx_trigger >= len(joint_msg.position):
            return

        current = float(joint_msg.position[idx_trigger])

        if self.last_saved_joint_angle_rad is None:
            self.last_saved_joint_angle_rad = current
            return

        d_deg = abs(math.degrees(shortest_angular_distance(self.last_saved_joint_angle_rad, current)))
        if d_deg >= self.threshold_deg:
            self.capture(rgb_msg, depth_msg, joint_msg, current, force_first=False)

    def capture(self, rgb_msg: Image, depth_msg: Image, joint_msg: JointState,
                trigger_joint_angle_rad: float, force_first: bool):
        if self.session_folder is None:
            self.start_new_session()

        seq = self.session_seq
        base_name = f"frame_{seq:04d}"

        rgb_path = self.session_folder / "rgb" / f"{base_name}.png"
        depth_path = self.session_folder / "depth" / f"{base_name}.png"
        rgbd_path = self.session_folder / "rgbd" / f"{base_name}.npy"
        poses_dir = self.session_folder / "poses"

        # Convert and save images
        rgb_bgr = self.image_msg_to_bgr(rgb_msg)
        depth_img = self.save_depth_image(depth_msg, depth_path) if self.save_depth_flag else self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        if self.save_rgb:
            cv2.imwrite(str(rgb_path), rgb_bgr)

        # FK and transforms
        joint_positions = {name: float(pos) for name, pos in zip(joint_msg.name, joint_msg.position)}
        T_base_ee = self.fk_solver.fk(joint_positions)
        T_cam_base, static_stamp_used = self.lookup_static_cam_to_base()
        T_cam_ee = T_cam_base @ T_base_ee
        T_ee_cam = invert_T(T_cam_ee)

        if self.save_base_to_ee_npy:
            np.save(poses_dir / f"{base_name}_T_base_to_ee.npy", T_base_ee)
            np.save(poses_dir / f"{base_name}_T_cam_to_base.npy", T_cam_base)
        if self.save_pose_npy:
            np.save(poses_dir / f"{base_name}_T_cam_to_ee.npy", T_cam_ee)
            np.save(poses_dir / f"{base_name}_T_ee_to_cam.npy", T_ee_cam)

        # Camera info save
        rgb_cam_info_dict = None
        depth_cam_info_dict = None
        if self.save_camera_info:
            if self.latest_rgb_camera_info is not None:
                rgb_cam_info_dict = self.camera_info_to_dict(self.latest_rgb_camera_info)
                np.save(poses_dir / f"{base_name}_rgb_camera_info.npy", rgb_cam_info_dict, allow_pickle=True)
            if self.latest_depth_camera_info is not None:
                depth_cam_info_dict = self.camera_info_to_dict(self.latest_depth_camera_info)
                np.save(poses_dir / f"{base_name}_depth_camera_info.npy", depth_cam_info_dict, allow_pickle=True)

        # RGBD bundle save
        if self.save_rgbd_npy:
            rgbd_bundle = {
                "rgb_bgr": rgb_bgr,
                "depth": depth_img,
                "rgb_encoding": str(rgb_msg.encoding),
                "depth_encoding": str(depth_msg.encoding),
                "rgb_stamp_sec": int(rgb_msg.header.stamp.sec),
                "rgb_stamp_nanosec": int(rgb_msg.header.stamp.nanosec),
                "depth_stamp_sec": int(depth_msg.header.stamp.sec),
                "depth_stamp_nanosec": int(depth_msg.header.stamp.nanosec),
                "joint_stamp_sec": int(joint_msg.header.stamp.sec),
                "joint_stamp_nanosec": int(joint_msg.header.stamp.nanosec),
                "rgb_frame_id": str(rgb_msg.header.frame_id),
                "depth_frame_id": str(depth_msg.header.frame_id),
                "joint_names": list(joint_msg.name),
                "joint_positions": np.array(joint_msg.position, dtype=np.float64),
                "T_base_to_ee": T_base_ee,
                "T_cam_to_base": T_cam_base,
                "T_cam_to_ee": T_cam_ee,
                "T_ee_to_cam": T_ee_cam,
                "depth_is_aligned_to_rgb": self.depth_is_aligned_to_rgb,
                "rgb_camera_info": rgb_cam_info_dict,
                "depth_camera_info": depth_cam_info_dict,
            }
            np.save(rgbd_path, rgbd_bundle, allow_pickle=True)

        rgb_ts = f"{int(rgb_msg.header.stamp.sec)}.{int(rgb_msg.header.stamp.nanosec):09d}"
        depth_ts = f"{int(depth_msg.header.stamp.sec)}.{int(depth_msg.header.stamp.nanosec):09d}"
        js_ts = f"{int(joint_msg.header.stamp.sec)}.{int(joint_msg.header.stamp.nanosec):09d}"

        rgb_depth_dt_ms = (
            (float(rgb_msg.header.stamp.sec) + float(rgb_msg.header.stamp.nanosec) * 1e-9)
            - (float(depth_msg.header.stamp.sec) + float(depth_msg.header.stamp.nanosec) * 1e-9)
        ) * 1000.0

        depth_joint_dt_ms = (
            (float(depth_msg.header.stamp.sec) + float(depth_msg.header.stamp.nanosec) * 1e-9)
            - (float(joint_msg.header.stamp.sec) + float(joint_msg.header.stamp.nanosec) * 1e-9)
        ) * 1000.0

        rgb_joint_dt_ms = (
            (float(rgb_msg.header.stamp.sec) + float(rgb_msg.header.stamp.nanosec) * 1e-9)
            - (float(joint_msg.header.stamp.sec) + float(joint_msg.header.stamp.nanosec) * 1e-9)
        ) * 1000.0

        self.session_meta.append(f"  - seq: {seq}\n")
        self.session_meta.append(f"    rgb_png: rgb/{base_name}.png\n")
        self.session_meta.append(f"    depth_png: depth/{base_name}.png\n")
        if self.save_rgbd_npy:
            self.session_meta.append(f"    rgbd_npy: rgbd/{base_name}.npy\n")
        self.session_meta.append(f"    trigger_mode: {'first_frame' if force_first else 'threshold'}\n")
        self.session_meta.append(f"    trigger_joint_angle_rad: {trigger_joint_angle_rad:.9f}\n")
        self.session_meta.append(f"    rgb_stamp: {rgb_ts}\n")
        self.session_meta.append(f"    depth_stamp: {depth_ts}\n")
        self.session_meta.append(f"    joint_stamp: {js_ts}\n")
        self.session_meta.append(f"    rgb_depth_sync_delta_ms: {rgb_depth_dt_ms:.3f}\n")
        self.session_meta.append(f"    depth_joint_sync_delta_ms: {depth_joint_dt_ms:.3f}\n")
        self.session_meta.append(f"    rgb_joint_sync_delta_ms: {rgb_joint_dt_ms:.3f}\n")
        self.session_meta.append(f"    static_cam_to_base_stamp_used: {static_stamp_used}\n")
        self.session_meta.append(f"    depth_is_aligned_to_rgb: {self.depth_is_aligned_to_rgb}\n")
        self.session_meta.append(f"    rgb_frame_id: {rgb_msg.header.frame_id}\n")
        self.session_meta.append(f"    depth_frame_id: {depth_msg.header.frame_id}\n")
        self.session_meta.append(f"    joint_names: {list(joint_msg.name)}\n")
        self.session_meta.append(f"    joint_positions: {[float(x) for x in joint_msg.position]}\n")
        if joint_msg.velocity:
            self.session_meta.append(f"    joint_velocities: {[float(x) for x in joint_msg.velocity]}\n")
        if joint_msg.effort:
            self.session_meta.append(f"    joint_efforts: {[float(x) for x in joint_msg.effort]}\n")
        self.session_meta.append(f"    T_base_to_ee: {pretty_T(T_base_ee)}\n")
        self.session_meta.append(f"    T_cam_to_base: {pretty_T(T_cam_base)}\n")
        self.session_meta.append(f"    T_cam_to_ee: {pretty_T(T_cam_ee)}\n")
        self.session_meta.append(f"    T_ee_to_cam: {pretty_T(T_ee_cam)}\n")

        self.last_saved_joint_angle_rad = trigger_joint_angle_rad
        self.session_seq += 1
        self.last_capture_walltime = time.time()

        self.get_logger().info(
            f"[capture {seq:04d}] "
            f"rgb_depth_dt={rgb_depth_dt_ms:.2f} ms | "
            f"depth_joint_dt={depth_joint_dt_ms:.2f} ms | "
            f"angle={math.degrees(trigger_joint_angle_rad):.2f} deg"
        )

    def flush_meta(self):
        if self.session_folder is None:
            return
        meta_path = self.session_folder / "meta.txt"
        meta_path.write_text("".join(self.session_meta), encoding="utf-8")
        self.get_logger().info(f"Flushed meta: {meta_path}")

    def check_idle(self):
        if self.session_folder is None or self.last_capture_walltime is None:
            return
        if time.time() - self.last_capture_walltime >= self.idle_flush_sec:
            self.flush_meta()
            self.session_folder = None
            self.session_id = None
            self.session_meta = []
            self.session_seq = 0
            self.last_capture_walltime = None
            # keep last_saved_joint_angle_rad so it does not recapture immediately after idle
            self.get_logger().info("Session closed due to idle timeout")

    def destroy_node(self):
        try:
            self.flush_meta()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = None
    try:
        node = JointTriggeredRGBDSessionManualFK()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.flush_meta()
            node.destroy_node()     
        rclpy.shutdown()


if __name__ == "__main__":
    main()


"""
Example:

ros2 run snapshot j6_trigger_capture_node_rgbd_fk --ros-args \
  -p joint_name:=J6_wrist3 \
  -p threshold_deg:=3.0 \
  -p outdir:=/home/bozznyskrtt/pcl_ws/teddybear\
  -p idle_flush_sec:=10.0 \
  -p rgb_topic:=/camera/color/image_raw \
  -p depth_topic:=/camera/depth/image_raw \
  -p rgb_camera_info_topic:=/camera/color/camera_info \
  -p depth_camera_info_topic:=/camera/depth/camera_info \
  -p joint_states_topic:=/joint_states \
  -p camera_frame:=camera_depth_optical_frame \
  -p base_frame:=base_link \
  -p ee_frame:=end_effector_1 \
  -p xacro_path:=/home/bozznyskrtt/hebi_ws/src/hebi_description/urdf/kits/A-2085-06G.urdf.xacro \
  -p sync_slop_sec:=0.03 \
  -p sync_queue_size:=50 \
  -p save_rgb:=true \
  -p save_depth:=true \
  -p save_rgbd_npy:=true \
  -p save_camera_info:=true \
  -p depth_is_aligned_to_rgb:=false

If your driver later provides aligned depth, use for example:
  -p depth_topic:=/camera/aligned_depth_to_color/image_raw
  -p depth_is_aligned_to_rgb:=true
"""