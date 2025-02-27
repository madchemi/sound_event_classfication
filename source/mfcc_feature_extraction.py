import os
import tqdm
import librosa
from pickle import TRUE
from tqdm import tqdm
from PIL import Image
import numpy as np
"""
pre_progressing python code

"""
size = 40
pad_size = 40 
repeat_size = 5
"""
긴 파일의 오디오 정보 잘리지 않도록, pad_size 설정 
pad_size는 sample rate에 따라서 달라지 .
mfcc의 return이  정사각형이 되도록 40 설정 
return이 커질 수록, 값이 많아짐. 
"""
#padding = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))




def fine_max_pad_len(sound_dir):
    mfcc_shape = []
    for i in range(10):
        for filename in tqdm(os.listdir(sound_dir+str(i+1)+'/')):
   
          if '.wav' not in filename:
            continue
          wav, sr = librosa.load(sound_dir+str(i+1)+'/'+filename, sr = 16000)
          mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=40, norm = 'ortho',n_fft=2048, hop_length=512)
          mfcc_shape.append(mfcc.shape[1])
    max_pad_len = max(mfcc_shape)
    return max_pad_len


def preprocessing(sound_dir, image_dir, max_pad_len):
        for filename in tqdm(os.listdir(sound_dir)):
   
          if '.wav' not in filename:
            continue
          wav, sr = librosa.load(sound_dir+filename, sr = 16000)
          mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=40, norm = 'ortho',n_fft=2048, hop_length=512)
          pad_width = max_pad_len - mfcc.shape[1]
          pad_mfcc = np.pad(mfcc, pad_width = ((0,0), (0,pad_width)), mode = 'constant')
          print(pad_mfcc.shape)
          save_path = image_dir + filename.split('-')[1] + '/' + filename + '.npy'
          np.save(save_path,pad_mfcc)
