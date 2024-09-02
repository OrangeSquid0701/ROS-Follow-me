#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray, Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np
import threading

# Initialize the ROS node
rospy.init_node('turtlebot_movement', anonymous=True)

# Publisher for cmd_vel
cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)

# Global variables for image data and pose landmarks
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

# Initialize VelocitySmoother
velocity_smoother = VelocitySmoother(alpha=0.1)

# PID Controller Parameters
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

# Obstacle Avoidance Parameters
obstacle_distance_threshold = 1.0  # Adjust as needed

def depth_image_callback(msg):
    global global_image_depth
    try:
        # Convert ROS Image message to a numpy array directly using the message's width and height
        depth_image = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        with image_lock:
            global_image_depth = depth_image  # Keep the original depth data for processing
    except Exception as e:
        rospy.logerr(f"Error processing depth image: {e}")

def calculate_distance(depth_image, x, y, min_valid_distance=0.1):
    if x < 0 or x >= depth_image.shape[1] or y < 0 or y >= depth_image.shape[0]:
        return None
    distance = depth_image[y, x] * 0.001  # Convert from millimeters to meters (if depth is in mm)
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
    
    avg_left_dist = np.nanmean(left_section)
    avg_center_dist = np.nanmean(center_section)
    avg_right_dist = np.nanmean(right_section)
    
    scale_factor = min(avg_center_dist / obstacle_distance_threshold, 1.0)
    scale_factor = max(scale_factor, 0.3)  # Ensure a minimum speed
    
    twist = Twist()
    if avg_center_dist < obstacle_distance_threshold:
        if avg_left_dist > avg_right_dist:
            twist.linear.x = 0.2 * scale_factor
            twist.angular.z = 0.5 * (1.0 - scale_factor)
        else:
            twist.linear.x = 0.2 * scale_factor
            twist.angular.z = -0.5 * (1.0 - scale_factor)
    else:
        twist.linear.x = 0.3 * scale_factor
        twist.angular.z = 0.0
    
    current_linear_x, current_angular_z = velocity_smoother.smooth(twist.linear.x, twist.angular.z)
    twist.linear.x = current_linear_x
    twist.angular.z = current_angular_z
    cmd_vel_pub.publish(twist)

def process_and_control_movement():
    global global_image_depth, center_x, center_y, integral_error, previous_error  # Add 'integral_error' and 'previous_error' here
    image_width = 640  # Adjust according to your camera's resolution
    image_height = 480  # Adjust according to your camera's resolution
    current_linear_x = 0
    current_angular_z = 0
    detected_once = False

    while not rospy.is_shutdown():
        if emergency_stop.is_set():
            break
        with image_lock:
            twist = Twist()
            if global_image_depth is not None:
                if detect_obstacle(global_image_depth):
                    avoid_obstacle(global_image_depth)
                else:
                    # Follow-me logic
                    distance_to_person = calculate_distance(global_image_depth, center_x, center_y)
                    if distance_to_person is not None:
                        desired_distance = 1.2
                        rospy.loginfo(f"Current distance to person: {distance_to_person:.2f} meters")
                        error = distance_to_person - desired_distance
                        integral_error += error
                        derivative_error = error - previous_error
                        previous_error = error
                        if abs(error) < dead_zone_threshold:
                            twist.linear.x = 0
                        else:
                            twist.linear.x = max(-0.2, min(0.2, proportional_gain * error + integral_gain * integral_error - derivative_gain * derivative_error))
                        if distance_to_person < desired_distance + 0.5:
                            twist.linear.x *= 0.5
                        else:
                            twist.linear.x *= damping_factor

                        image_center_x = global_image_depth.shape[1] // 2
                        if abs(center_x - image_center_x) > image_center_x * 0.1:
                            twist.angular.z = angular_proportional_gain * (image_center_x - center_x) / image_center_x

                        current_linear_x, current_angular_z = velocity_smoother.smooth(twist.linear.x, twist.angular.z)
                        twist.linear.x = current_linear_x
                        twist.angular.z = current_angular_z
                        cmd_vel_pub.publish(twist)
                    else:
                        rospy.logwarn("Distance to person could not be determined. Skipping distance adjustment.")
            else:
                rospy.logwarn("No depth image available to calculate distance.")

        rospy.sleep(0.05)

def follow_me_callback(msg):
    if msg.data:
        emergency_stop.set()
    else:
        emergency_stop.clear()

def pose_callback(msg):
    global center_x, center_y
    left_hip_x, left_hip_y, right_hip_x, right_hip_y = msg.data
    center_x = int((left_hip_x + right_hip_x) / 2 * 640)
    center_y = int((left_hip_y + right_hip_y) / 2 * 480)

if __name__ == '__main__':
    try:
        rospy.Subscriber('/camera/depth/image_rect_raw', Image, depth_image_callback)
        rospy.Subscriber('/pose_landmarks', Float32MultiArray, pose_callback)
        rospy.Subscriber('/follow_me_stop', Bool, follow_me_callback)
        process_and_control_movement()
    except rospy.ROSInterruptException:
        pass
