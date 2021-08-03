import librosa   #for audio processing
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile #for audio processing
import warnings
warnings.filterwarnings("ignore")

audio_path='../data/alldata/'
samples, sample_rate = librosa.load(audio_path+'SWH-05-20101106_16k-emission_swahili_05h30_-_06h00_tu_20101106_part2.wav')
print(samples)