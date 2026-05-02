#!/usr/bin/env python3

import os
import sys
import cv2
import yaml
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

# キーボード入力用
import threading
from pynput import keyboard


class ColorImageSaver(Node):
    def __init__(self):
        super().__init__('color_image_saver')
        
        # パラメータの宣言
        self.declare_parameter('save_directory', os.path.expanduser('~/camera_data'))
        self.save_directory = self.get_parameter('save_directory').value
        
        # ディレクトリ作成
        self.image_dir = Path(self.save_directory) / 'images'
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(f'Save directory: {self.save_directory}')
        
        # CvBridge
        self.bridge = CvBridge()
        
        # 最新の画像とカメラ情報を保持
        self.latest_image = None
        self.latest_image_info = None
        self.camera_info = None
        self.lock = threading.Lock()
        
        # サブスクライバー
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )
        
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.camera_info_callback,
            10
        )
        
        self.get_logger().info('Color Image Saver Node Started')
        self.get_logger().info('Press SPACE key to save image')
        
        # キーボードリスニング開始
        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()

    def image_callback(self, msg):
        """画像トピックのコールバック"""
        with self.lock:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.latest_image = cv_image
                self.latest_image_info = msg
            except Exception as e:
                self.get_logger().error(f'Failed to convert image: {e}')

    def camera_info_callback(self, msg):
        """カメラ情報トピックのコールバック"""
        with self.lock:
            self.camera_info = msg
            # カメラ情報を保存（最初の1回のみ）
            if not hasattr(self, 'camera_info_saved'):
                self.save_camera_info(msg)
                self.camera_info_saved = True

    def on_key_press(self, key):
        """キーボード入力のコールバック"""
        try:
            if key == keyboard.Key.space:
                self.capture_image()
        except AttributeError:
            pass

    def capture_image(self):
        """画像を保存"""
        with self.lock:
            if self.latest_image is None:
                self.get_logger().warn('No image received yet')
                return
            
            # タイムスタンプ取得
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            
            # ファイル名
            image_filename = f'image_{timestamp}.png'
            image_path = self.image_dir / image_filename
            
            # 画像保存
            try:
                cv2.imwrite(str(image_path), self.latest_image)
                self.get_logger().info(f'Image saved: {image_path}')
                
                # メタデータ保存
                metadata = {
                    'timestamp': timestamp,
                    'frame_id': self.latest_image_info.header.frame_id if self.latest_image_info else 'unknown',
                    'seq': self.latest_image_info.header.seq if self.latest_image_info else 0,
                    'height': int(self.latest_image.shape[0]),
                    'width': int(self.latest_image.shape[1]),
                    'channels': int(self.latest_image.shape[2]) if len(self.latest_image.shape) > 2 else 1,
                }
                
                # カメラ内部パラメータを含める
                if self.camera_info:
                    metadata['camera_matrix'] = {
                        'fx': float(self.camera_info.k[0]),
                        'fy': float(self.camera_info.k[4]),
                        'cx': float(self.camera_info.k[2]),
                        'cy': float(self.camera_info.k[5]),
                    }
                    metadata['distortion_coefficients'] = {
                        'd': [float(d) for d in self.camera_info.d],
                    }
                
                metadata_filename = f'image_{timestamp}_metadata.yaml'
                metadata_path = self.image_dir / metadata_filename
                
                with open(metadata_path, 'w') as f:
                    yaml.dump(metadata, f, default_flow_style=False)
                
            except Exception as e:
                self.get_logger().error(f'Failed to save image: {e}')

    def save_camera_info(self, camera_info):
        """カメラ情報を保存"""
        try:
            camera_info_data = {
                'camera_name': camera_info.camera_name,
                'resolution': {
                    'width': int(camera_info.width),
                    'height': int(camera_info.height),
                },
                'camera_matrix': {
                    'fx': float(camera_info.k[0]),
                    'fy': float(camera_info.k[4]),
                    'cx': float(camera_info.k[2]),
                    'cy': float(camera_info.k[5]),
                },
                'distortion_coefficients': {
                    'd': [float(d) for d in camera_info.d],
                },
                'distortion_model': camera_info.distortion_model,
            }
            
            info_path = Path(self.save_directory) / 'camera_info.yaml'
            with open(info_path, 'w') as f:
                yaml.dump(camera_info_data, f, default_flow_style=False)
            
            self.get_logger().info(f'Camera info saved: {info_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to save camera info: {e}')

    def destroy_node(self):
        """ノード終了処理"""
        self.listener.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ColorImageSaver()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()