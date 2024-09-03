#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import cv2
import mediapipe as mp
import numpy as np
import threading

# Initialize mediapipe pose detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

global_image_bgr = None
image_lock = threading.Lock()

def rgb_image_callback(msg):
    global global_image_bgr
    try:
        if msg.encoding == 'mono8':
            rgb_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2BGR)
        elif msg.encoding == 'rgb8':
            rgb_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        elif msg.encoding == 'bayer_grbg8':
            bayer_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            rgb_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_GR2BGR)
        else:
            rospy.logerr(f"Unsupported image encoding: {msg.encoding}")
            return

        with image_lock:
            global_image_bgr = rgb_image
    except Exception as e:
        rospy.logerr(f"Error processing RGB image: {e}")

def detect_pose():
    global global_image_bgr
    rospy.init_node('pose_detector', anonymous=True)
    pose_pub = rospy.Publisher('/pose_landmarks', Float32MultiArray, queue_size=10)

    with mp_pose.Ppose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while not rospy.is_shutdown():
            with image_lock:
                if global_image_bgr is not None:
                    image_rgb = cv2.cvtColor(global_image_bgr, cv2.COLOR_BGR2RGB)
                    image_rgb.flags.writeable = False
                    results = pose.process(image_rgb)
                    image_rgb.flags.writeable = True

                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            global_image_bgr, 
                            results.pose_landmarks, 
                            mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(144, 238, 144), thickness=2, circle_radius=2)
                        )
                        
                        left_hip = results.pose_landmarks.landmark[23]
                        right_hip = results.pose_landmarks.landmark[24]

                        pose_msg = Float32MultiArray()
                        pose_msg.data = [left_hip.x, left_hip.y, right_hip.x, right_hip.y]
                        pose_pub.publish(pose_msg)

                    cv2.imshow("RGB Image with Pose", global_image_bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        rospy.signal_shutdown("Shutdown initiated by user.")
                        break

            rospy.sleep(0.05)

if __name__ == '__main__':
    try:
        rospy.Subscriber('/camera/rgb/image_raw', Image, rgb_image_callback)
        detect_pose()
    except rospy.ROSInterruptException:
        pass
