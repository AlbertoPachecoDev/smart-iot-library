# Smart IoT Library

import requests
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display

def version():
  ''' Shows Smart IoT library version '''
  print('Smart IoT Library ver. 0.4')
  print('torchaudio ver.', torchaudio.__version__)

def wave_size(wave):
  ''' Returns the wave object memory size '''
  if wave.ndim == 1:
    channels, frames = 1, wave.shape[0]
  else:
    channels, frames = wave.shape
  return wave.element_size() * channels * frames  

def load_audio(url, fname):
  '''
    Retrieves audio file from URL and saves with a filename 
    Returns audio-wave, sample-rate, metada & bytes-size
  '''
  r = requests.get(url)
  with open(fname, 'wb') as f:
    f.write(r.content)
  sz = len(r.content)
  meta = torchaudio.info(fname)
  wave, sr = torchaudio.load(fname)
  return (wave, sr, meta, sz)

def print_info(info, fname=None):
  ''' Shows audio metadata information using load_audio() return tuple value ''' 
  if fname:
    print('-' * 30)
    print('   Filename:', fname)
    print('-' * 30)
  wave, sr, meta, sz = info
  channels, frames = wave.shape
  print(f'     Frames: {frames}')
  print(f'   Channels: {channels}')
  print(f'  File size: {sz} bytes')
  print(f'Tensor size: {wave_size(wave)} bytes')
  print(f'      Dtype: {wave.dtype}')
  print(f'        Max: {wave.max().item():6.3f}')
  print(f'        Min: {wave.min().item():6.3f}')
  print(f'       Mean: {wave.mean().item():6.3f}')
  print(f'    Std Dev: {wave.std().item():6.3f}')
  print(wave)

def plot_wave(wave, torch=True):
  ''' Plots PyTorch or NumPy waves '''
  plt.figure()
  plt.plot(wave[0].numpy() if torch else wave)
 
def plot_fft(wave, max_freq=None):
  ''' Displays fft signal from 0 - max_freq range'''
  wave2 = wave[:max_freq] if max_freq else wave
  wave3 = np.abs(wave2.real)
  plt.figure()
  plt.plot(wave3, lw=1, color='green')

def play_audio(wave, sample_rate, torch=True):
  ''' Plays PyTorch/NumPy audio signals '''
  channels = (1 if wave.ndim == 1 else wave.shape[0])
  if torch:
    wave = wave.numpy()
    if channels == 1:
      display(Audio(wave, rate=sample_rate))
    elif channels == 2:
      display(Audio((wave[0], wave[1]), rate=sample_rate))
    else:
      raise ValueError("Waveform with more than 2 channels are not supported.")
  else: # numpy array
    if channels == 1:
      display(Audio(wave, rate=sample_rate))
    else:
      display(Audio((wave[0], wave[1]), rate=sample_rate))
