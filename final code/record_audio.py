import numpy as np 
import csv 
import pandas as pd
#from scipy.linalg import dft
from scipy import signal
from scipy import fft
from scipy.fft import ifft
import matplotlib.pyplot as plt 
from scipy.io.wavfile import write # For writing to a .wav file
from scipy.io.wavfile import read
import pyaudio
import wave

chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paFloat32  # 16 bits per sample
channels = 1
fs = 44100  # Record at 44100 samples per second
seconds = 3
filename = "output.wav"

p = pyaudio.PyAudio()  # Create an interface to PortAudio

print('Recording')

stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)

frames = np.array([])  # Initialize array to store frames

# Store data in chunks for 3 seconds
for i in range(0, int(fs / chunk * seconds)):
    data = stream.read(chunk)
    data2 = np.frombuffer(data, dtype='float32')
    frames = np.append(frames, data2)

# Stop and close the stream 
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print('Finished recording')

def convert_array_to_wav(array, sample_rate):
  # Convert the normalized array to a wav file.
  write("output.wav", sample_rate, array)


convert_array_to_wav(frames, 44100)
