#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray, Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np
import threading

rospy.init_node('turtlebot_movement', anonymous=True)
cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)

global_image_depth = None
center_x = 0
center_y = 0
image_lock = threading.Lock()
emergency_stop = threading.Event()

class VelocitySmoother:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.linear_x = 0
        self.angular_z = 0

    def smooth(self, linear_x, angular_z):
        self.linear_x = self.alpha * linear_x + (1 - self.alpha) * self.linear_x
        self.angular_z = self.alpha * angular_z + (1 - self.alpha) * self.angular_z
        return self.linear_x, self.angular_z

velocity_smoother = VelocitySmoother(alpha=0.1)

integral_error = 0
previous_error = 0
integral_gain = 0.01
proportional_gain = 0.5
derivative_gain = 0.1
dead_zone_threshold = 0.05
damping_factor = 1.0
angular_proportional_gain = 1.0
angular_speed_limit = 0.5
deceleration_base_rate = 0.2

obstacle_distance_threshold = 1.0

previous_center = None
movement_threshold = 0.01  # Threshold to determine if person is moving
movement_list = []
frame_count = 30  # Number of frames to average over

def depth_image_callback(msg):
    global global_image_depth
    try:
        depth_image = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        with image_lock:
            global_image_depth = depth_image
    except Exception as e:
        rospy.logerr(f"Error processing depth image: {e}")

def calculate_distance(depth_image, x, y, min_valid_distance=0.1):
    if x < 0 or x >= depth_image.shape[1] or y < 0 or y >= depth_image.shape[0]:
        return None
    distance = depth_image[y, x] * 0.001
    if distance == 0 or np.isnan(distance) or distance < min_valid_distance:
        return None
    return distance

def detect_obstacle(depth_array):
    center_section = depth_array[:, depth_array.shape[1]//3: 2*depth_array.shape[1]//3]
    avg_center_dist = np.nanmean(center_section)
    return avg_center_dist < obstacle_distance_threshold

def avoid_obstacle(depth_array):
    left_section = depth_array[:, :depth_array.shape[1]//3]
    center_section = depth_array[:, depth_array.shape[1]//3: 2*depth_array.shape[1]//3]
    right_section = depth_array[:, 2*depth_array.shape[1]//3:]
    
    avg_left_dist = np.nan

