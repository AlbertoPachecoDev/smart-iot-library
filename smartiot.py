# Smart IoT Library

import requests
import torchaudio

def version():
  ''' Shows Smar IoT library version '''
  print('Smart IoT Library ver. 0.2')

def load_audio(url, fname):
  '''
    Retrieve audio file from URL and saves with a filename 
    Returns audio-wave, sample-rate, metada, bytes-size
  '''
  r = requests.get(url)
  with open(fname, 'wb') as f:
    f.write(r.content)
  sz = len(r.content)
  meta = torchaudio.info(fname)
  wave, sr = torchaudio.load(fname)
  return (wave, sr, meta, sz)

def print_info(info, fname=None):
  ''' Muestra los metadatos de un archivo de audio con la tupla que regresa load_audio'''
  if fname:
    print('-' * 30)
    print('   Filename:', fname)
    print('-' * 30)
  wave, sr, meta, sz = info
  channels, frames = wave.shape
  print(f'     Frames: {frames}')
  print(f'   Channels: {channels}')
  print(f'  File size: {sz} bytes')
  print(f'Tensor size: {wave.element_size() * channels * frames} bytes')
  print(f'      Dtype: {wave.dtype}')
  print(f'        Max: {wave.max().item():6.3f}')
  print(f'        Min: {wave.min().item():6.3f}')
  print(f'       Mean: {wave.mean().item():6.3f}')
  print(f'    Std Dev: {wave.std().item():6.3f}')
  print(wave)
