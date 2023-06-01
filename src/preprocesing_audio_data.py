"""
Preprocessing audio data for Deep Learning
"""
import librosa
import matplotlib.pyplot as plt
import numpy as np

AUDIO_FILE = "audios/fde2dee7_nohash_1.wav"
signal, sample_rate = librosa.load(
    AUDIO_FILE, sr=22050)  # len(signal) = sr * time

# 1. Waveform
librosa.display.waveshow(signal, sr=sample_rate)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# 2. fft -> spectrum
fft = np.fft.fft(signal)

magnitude = np.abs(fft)
frequency = np.linspace(0, sample_rate, len(magnitude))

# one of the parts is taken since the other is repeated
left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(frequency)/2)]

plt.plot(left_frequency, left_magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()

# stft -> spectrogram
N_FFT = 2048
HOP_LENGTH = 512

stft = librosa.core.stft(signal, hop_length=N_FFT, n_fft=HOP_LENGTH)
spectrogram = np.abs(stft)

log_spectogram = librosa.amplitude_to_db(spectrogram)

librosa.display.specshow(log_spectogram, sr=sample_rate, hop_length=HOP_LENGTH)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()

# MFCCs
N_MFCC = 20
mfccs = librosa.feature.mfcc(
    y=signal, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
librosa.display.specshow(mfccs, sr=sample_rate, hop_length=HOP_LENGTH)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()
