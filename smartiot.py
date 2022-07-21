# Smart IoT Library

import requests
import torchaudio

def version():
  ''' Shows Smar IoT library version '''
  print('Smart IoT Library ver. 0.1')

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
