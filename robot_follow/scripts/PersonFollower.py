#!/usr/bin/env python3

import cv2
import mediapipe as mp
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np
import threading

# Initialize mediapipe pose detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize the ROS node
rospy.init_node('mediapipe_body_follower', anonymous=True)

# Publisher for cmd_vel
cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity', Twist, queue_size=10)

# Global variables for image data
global_image_bgr = None
global_image_depth = None
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

# Deceleration parameters
deceleration_rate = 0.05  # How fast to decelerate

def depth_image_callback(msg):
    global global_image_depth

    try:
        # Convert ROS Image message to a numpy array
        depth_image = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)

        # Convert to 8-bit grayscale for visualization
        depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_image_8bit = np.uint8(depth_image_normalized)

        # Convert to a 3-channel BGR image for visualization
        with image_lock:
            global_image_depth = cv2.cvtColor(depth_image_8bit, cv2.COLOR_GRAY2BGR)

    except Exception as e:
        rospy.logerr(f"Error processing depth image: {e}")

def rgb_image_callback(msg):
    global global_image_bgr

    try:
        # Check the image encoding and handle accordingly
        if msg.encoding == 'mono8':  # Single-channel grayscale image
            rgb_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR
        elif msg.encoding == 'rgb8':  # RGB image
            rgb_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        elif msg.encoding == 'bayer_grbg8':  # Bayer GRBG pattern image
            bayer_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            rgb_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_GR2BGR)  # Convert Bayer to BGR
        else:
            rospy.logerr(f"Unsupported image encoding: {msg.encoding}")
            return

        with image_lock:
            global_image_bgr = rgb_image

    except Exception as e:
        rospy.logerr(f"Error processing RGB image: {e}")


def process_and_follow_body():
    global global_image_bgr, global_image_depth
    image_width = 640  # Adjust according to your camera's resolution
    image_height = 480  # Adjust according to your camera's resolution

    # Initialize velocities to zero
    current_linear_x = 0
    current_angular_z = 0

    # Flag to track if landmarks were detected at least once
    detected_once = False

    # Corrected typo from Ppose to Pose
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while not rospy.is_shutdown():
            if emergency_stop.is_set():
                break

            with image_lock:
                if global_image_bgr is not None:
                    # Convert BGR to RGB for MediaPipe processing
                    image_rgb = cv2.cvtColor(global_image_bgr, cv2.COLOR_BGR2RGB)
                    image_rgb.flags.writeable = False
                    results = pose.process(image_rgb)
                    image_rgb.flags.writeable = True

                    # Convert back to BGR for display
                    global_image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                    twist = Twist()  # Initialize Twist message

                    if results.pose_landmarks:
                        detected_once = True  # Set flag to true when landmarks are detected

                        mp_drawing.draw_landmarks(
                            global_image_bgr, 
                            results.pose_landmarks, 
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(144, 238, 144), thickness=2, circle_radius=2)
                        )

                        # Updated landmarks for detection
                        left_shoulder = results.pose_landmarks.landmark[11]  # LEFT_SHOULDER
                        right_shoulder = results.pose_landmarks.landmark[12]  # RIGHT_SHOULDER
                        left_hip = results.pose_landmarks.landmark[23]  # LEFT_HIP
                        right_hip = results.pose_landmarks.landmark[24]  # RIGHT_HIP
                        left_knee = results.pose_landmarks.landmark[25]  # LEFT_KNEE
                        right_knee = results.pose_landmarks.landmark[26]  # RIGHT_KNEE

                        detected_landmarks = 0
                        landmarks = [left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee]
                        for landmark in landmarks:
                            if landmark.visibility > 0.5:
                                detected_landmarks += 1

                        # Check if at least half of the landmarks are detected
                        if detected_landmarks >= 3:
                            avg_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4
                            mid_x = avg_x * image_width
                            offset_x = mid_x - (image_width / 2)

                            turn_sensitivity = 0.5  # Increased turning sensitivity
                            max_turn_rate = 0.5  # Increased maximum turning rate

                            twist.angular.z = -turn_sensitivity * offset_x / (image_width / 2)  # Adjust turning rate
                            twist.angular.z = max(min(twist.angular.z, max_turn_rate), -max_turn_rate)

                            avg_z = (left_hip.z + right_hip.z + left_knee.z + right_knee.z) / 4
                            if avg_z < -1.5:
                                twist.linear.x = -0.2  # Increased forward speed
                            elif avg_z > -1.2:
                                twist.linear.x = 0.2  # Increased backward speed
                            else:
                                twist.linear.x = 0  # Maintain distance

                            current_linear_x = twist.linear.x
                            current_angular_z = twist.angular.z

                        else:
                            if detected_once:
                                # Gradual stopping if landmarks were previously detected
                                if abs(current_linear_x) > 0.01 or abs(current_angular_z) > 0.01:
                                    current_linear_x -= np.sign(current_linear_x) * deceleration_rate
                                    current_angular_z -= np.sign(current_angular_z) * deceleration_rate
                                    if abs(current_linear_x) < 0.01:
                                        current_linear_x = 0
                                    if abs(current_angular_z) < 0.01:
                                        current_angular_z = 0
                                twist.linear.x = current_linear_x
                                twist.angular.z = current_angular_z
                            else:
                                # No landmarks detected at start or after detection
                                twist.linear.x = 0
                                twist.angular.z = 0

                        # Smooth the velocity commands
                        smooth_linear_x, smooth_angular_z = velocity_smoother.smooth(twist.linear.x, twist.angular.z)
                        twist.linear.x = smooth_linear_x
                        twist.angular.z = smooth_angular_z

                        # Publish the movement command
                        cmd_vel_pub.publish(twist)

                    else:
                        if detected_once:
                            # Gradual stopping if landmarks were previously detected
                            if abs(current_linear_x) > 0.01 or abs(current_angular_z) > 0.01:
                                current_linear_x -= np.sign(current_linear_x) * deceleration_rate
                                current_angular_z -= np.sign(current_angular_z) * deceleration_rate
                                if abs(current_linear_x) < 0.01:
                                    current_linear_x = 0
                                if abs(current_angular_z) < 0.01:
                                    current_angular_z = 0
                            twist.linear.x = current_linear_x
                            twist.angular.z = current_angular_z
                        else:
                            # No landmarks detected at start or after detection
                            twist.linear.x = 0
                            twist.angular.z = 0

                        smooth_linear_x, smooth_angular_z = velocity_smoother.smooth(twist.linear.x, twist.angular.z)
                        twist.linear.x = smooth_linear_x
                        twist.angular.z = smooth_angular_z

                        cmd_vel_pub.publish(twist)

                    if global_image_depth is not None:
                        # Show the image with landmarks and speed info
                        combined_image = np.hstack((global_image_depth, global_image_bgr))
                        cv2.putText(combined_image, f'Linear Speed: {twist.linear.x:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(combined_image, f'Angular Speed: {twist.angular.z:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.imshow('Depth Image and RGB Image with Landmarks and Speed', combined_image)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        rospy.signal_shutdown("Emergency stop initiated by user.")
                        emergency_stop.set()
                        break

# Subscribe to both depth and RGB image topics
rospy.Subscriber('/camera/depth/image_rect_raw', Image, depth_image_callback)
rospy.Subscriber('/camera/rgb/image_raw', Image, rgb_image_callback)

# Start the processing and follow thread
follow_thread = threading.Thread(target=process_and_follow_body)
follow_thread.start()

# Spin to keep the script alive
rospy.spin()

# Wait for the follow thread to finish
follow_thread.join()

cv2.destroyAllWindows()
