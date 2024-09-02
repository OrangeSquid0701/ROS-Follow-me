#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
import wave
import pyaudio
import time
import vosk
import json
import numpy as np
from pydub import AudioSegment

# Global variable to track if the robot velocity is 0m/s
velocity_is_zero = False

def play_audio(file_name):
    wave_file = wave.open(file_name, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wave_file.getsampwidth()),
                    channels=wave_file.getnchannels(),
                    rate=wave_file.getframerate(),
                    output=True)
    chunk = 1024
    data = wave_file.readframes(chunk)
    while data:
        stream.write(data)
        data = wave_file.readframes(chunk)
    stream.stop_stream()
    stream.close()
    p.terminate()
    wave_file.close()

def filter_noise(audio_data, rate):
    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    freq_cutoff = 300  # 300 Hz
    fft_audio = np.fft.rfft(audio_np)
    frequencies = np.fft.rfftfreq(len(audio_np), d=1/rate)
    fft_audio[frequencies < freq_cutoff] = 0
    filtered_audio_np = np.fft.irfft(fft_audio)
    filtered_audio_data = filtered_audio_np.astype(np.int16).tobytes()
    return filtered_audio_data

def listen_for_stop_command(model, listen_duration=5):
    print("Listening for 'yes' to stop...")
    recognizer = vosk.KaldiRecognizer(model, 16000)
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
    stream.start_stream()

    start_time = time.time()
    while time.time() - start_time < listen_duration:
        data = stream.read(4000, exception_on_overflow=False)
        filtered_data = filter_noise(data, 16000)
        if recognizer.AcceptWaveform(filtered_data):
            result = recognizer.Result()
            result_dict = json.loads(result)
            text = result_dict.get("text", "")
            print(f"Recognized text: {text}")
            if "yes" in text.lower():
                print("Detected 'yes', stopping the loop.")
                stream.stop_stream()
                stream.close()
                p.terminate()
                return True

    stream.stop_stream()
    stream.close()
    p.terminate()
    return False

def repeat_audio(file_name, model, repeat_interval=5):
    while True:
        play_audio(file_name)
        print(f"Waiting {repeat_interval} seconds to listen for the stop command...")
        if listen_for_stop_command(model, listen_duration=repeat_interval):
            break

def velocity_callback(msg):
    global velocity_is_zero
    if msg.linear.x == 0 and msg.angular.z == 0:
        velocity_is_zero = True
    else:
        velocity_is_zero = False

def main():
    global velocity_is_zero
    
    rospy.init_node('audio_listener', anonymous=True)
    rospy.Subscriber("/mobile_base/commands/velocity", Twist, velocity_callback)
    
    model_path = "/home/charmander/catkin_ws/src/follow_me/scripts/vosk-model-small-en-us-0.15"
    model = vosk.Model(model_path)
    
    audio_file = '/home/charmander/catkin_ws/src/follow_me/scripts/spongebobstop.wav'
    
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        if velocity_is_zero:
            print("Velocity is 0 m/s, starting audio playback and listener...")
            repeat_audio(audio_file, model, repeat_interval=5)
            velocity_is_zero = False  # Reset flag after starting the process
        rate.sleep()

if __name__ == "__main__":
    main()
