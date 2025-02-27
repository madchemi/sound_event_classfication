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
padding = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))

skip_class=['0', '2', '3', '4', '6', '7', '9']



def pre_progressing(sound_dir, image_dir):
  # 마지막에 /로 끝나게
  if image_dir[-1] != '/':
    image_dir.join('/')
  if sound_dir[-1] != '/':
    sound_dir.join('/')
  for filename in tqdm(os.listdir(sound_dir)):
    # print(filename)
    # 유니코드 정규화?
    # filename = normalize('NFC', filename)
    # wav 포맷 데이터만 사용

    if '.wav' not in filename:
      continue
    # 필요없는 클래스 스
    if filename.split('-')[1] in skip_class:
       src= os.path.join(sound_dir, filename)

       new_name = filename.split('-')
       new_name[1] = '0'
       new_name = '-'.join(new_name)
       
       dst = os.path.join(sound_dir,new_name)
       os.rename(src, dst)
       filename = new_name
    # 5번 클래스 -> 2로
    if filename.split('-')[1] == '5':
       src= os.path.join(sound_dir, filename)

       new_name = filename.split('-')
       new_name[1] = '2'
       new_name = '-'.join(new_name)
       
       dst = os.path.join(sound_dir,new_name)
       os.rename(src, dst)
       filename = new_name  

    # 8번 -> 3번으로 
    if filename.split('-')[1] =='8':
       src= os.path.join(sound_dir, filename)

       new_name = filename.split('-')
       new_name[1] = '3'
       new_name = '-'.join(new_name)
       
       dst = os.path.join(sound_dir,new_name)
       os.rename(src, dst)
       filename = new_name
    wav, sr = librosa.load(sound_dir+filename, sr=16000)
    # 이미지 ndarray
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=100, n_fft=400, hop_length=160)
    # 패딩된 ndarray
    # 리사이즈 할거면 패딩 필요없음?
  
    mfcc = librosa.util.normalize(mfcc)
    padded_mfcc = padding(mfcc, 500)
    # 해당 이미지 새로 저장
    # img = Image.fromarray(padded_mfcc)
    img = Image.fromarray(mfcc)
    img = img.convert('L')
    # resize크기 적당히 조절
    img = img.resize([32,32])
    img.save(image_dir+filename.split('-')[1]+'/'+filename+'.jpg','JPEG')
