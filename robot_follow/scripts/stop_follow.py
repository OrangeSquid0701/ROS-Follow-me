#!/usr/bin/env python3

import wave
import pyaudio
import time
import vosk
import json
import numpy as np
from pydub import AudioSegment

def play_audio(file_name):
    # Open the .wav file
    wave_file = wave.open(file_name, 'rb')

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open a stream with the correct settings
    stream = p.open(format=p.get_format_from_width(wave_file.getsampwidth()),
                    channels=wave_file.getnchannels(),
                    rate=wave_file.getframerate(),
                    output=True)

    # Read data in chunks
    chunk = 1024

    # Play the audio
    data = wave_file.readframes(chunk)
    while data:
        stream.write(data)
        data = wave_file.readframes(chunk)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Close PyAudio
    p.terminate()

    # Close the .wav file
    wave_file.close()

def filter_noise(audio_data, rate):
    # Convert raw audio data to an array
    audio_np = np.frombuffer(audio_data, dtype=np.int16)

    # Apply a simple high-pass filter to remove low-frequency noise
    freq_cutoff = 300  # 300 Hz
    fft_audio = np.fft.rfft(audio_np)
    frequencies = np.fft.rfftfreq(len(audio_np), d=1/rate)
    fft_audio[frequencies < freq_cutoff] = 0

    # Convert the filtered audio back to the time domain
    filtered_audio_np = np.fft.irfft(fft_audio)
    
    # Convert the array back to bytes
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
        
        # Apply noise filtering to the audio data
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

        # Listen for the stop command for a duration equal to the repeat interval
        if listen_for_stop_command(model, listen_duration=repeat_interval):
            break

# Load Vosk model
model_path = "/home/charmander/catkin_ws/src/follow_me/scripts/vosk-model-small-en-us-0.15"  # Update with the path to your model
model = vosk.Model(model_path)

# Specify your .wav file name
audio_file = '/home/charmander/catkin_ws/src/follow_me/scripts/spongebobstop.wav'

# Repeat the audio indefinitely every 5 seconds until "yes" is detected
repeat_audio(audio_file, model, repeat_interval=5)
