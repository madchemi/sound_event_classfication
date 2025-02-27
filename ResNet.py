#pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#데이터셋 로더
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split
#음성파일 처리
import librosa
#기타
import numpy as np
from pickle import TRUE
from tqdm import tqdm
import os 
import pandas as pd
import shutil
# 전처리
from PIL import Image


padding = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))



# 0 에어컨
# 2 children_playing
# 3 dog_bark
# 4 drill
# 6 gunshot
# 7 jack hammer
# 9 street music
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
      continue
    wav, sr = librosa.load(sound_dir+filename, sr=16000)
    # 이미지 ndarray
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=100, n_fft=400, hop_length=160)
    # 패딩된 ndarray
    # 리사이즈 할거면 패딩 필요없음?
    padded_mfcc = padding(mfcc, 500)

    # 해당 이미지 새로 저장
    # img = Image.fromarray(padded_mfcc)
    img = Image.fromarray(mfcc)
    img = img.convert('L')
    # resize크기 적당히 조절
    img = img.resize([32,32])
    img.save(image_dir+filename.split('-')[1]+'/'+filename+'.jpg','JPEG')

# 돌리기 전에 폴더 비워둘것
# 클래스 빈 폴더 생성
for i in range(10):
  os.makedirs('C:/Users/seoji/Desktop/gist/gist_makerton/archive-2/'+str(i), exist_ok=True)
# 전처리 과정 수행
for i in range(10):
  pre_progressing('C:/Users/seoji/Desktop/gist/gist_makerton/archive-2/fold'+str(i+1)+'/', 'C:/Users/seoji/Desktop/gist/gist_makerton/archive-2/image_set/')

for class_folder in os.listdir('C:/Users/seoji/Desktop/gist/gist_makerton/archive-2/image_set/'):
  if len(os.listdir('C:/Users/seoji/Desktop/gist/gist_makerton/archive-2/image_set/'+class_folder)) == 0:
    os.rmdir('C:/Users/seoji/Desktop/gist/gist_makerton/archive-2/image_set/'+class_folder)
dataset = ImageFolder(root='C:/Users/seoji/Desktop/gist/gist_makerton/archive-2/image_set/',transform=transforms.Compose([transforms.ToTensor(),]))

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 각각의 데이터셋에 대해 DataLoader 생성
train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# 만약 MobileNetV3를 사용하려면: models.mobilenet_v3_large(pretrained=True)

# 2. 모델의 마지막 레이어를 학습할 데이터에 맞게 수정 (예: 10개 클래스라면 10으로 수정)
num_classes = len(os.listdir('C:/Users/seoji/Desktop/gist/gist_makerton/archive-2/image_set/')  # 예시: 클래스가 10개인 경우
model.classifier[1] = nn.Linear(model.last_channel, num_classes)  # 마지막 레이어 수정

# 3. 모델을 GPU로 보내기 (가능한 경우)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 4. 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 학습 함수 정의
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()  # 모델을 학습 모드로 설정
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 옵티마이저 초기화
        optimizer.zero_grad()

        # 순전파 (Forward pass)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 역전파 (Backward pass) 및 최적화
        loss.backward()
        optimizer.step()

        # 손실 계산
        running_loss += loss.item()

        # 정확도 계산
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels).item()
        total_predictions += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct_predictions / total_predictions

    return epoch_loss, epoch_acc

# 6. 검증 함수 정의
class ResidualBlock(nn.Module):
    def __init__(self, filters_in, filters_out):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters_in, filters_in, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(filters_in)
        self.conv2 = nn.Conv2d(filters_in, filters_in, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters_in)
        self.conv3 = nn.Conv2d(filters_in, filters_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(filters_out)

        # Shortcut layer
        if filters_in != filters_out:
            self.shortcut = nn.Conv2d(filters_in, filters_out, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.shortcut is not None:
            shortcut = self.shortcut(x)

        out += shortcut
        out = F.relu(out)
        return out

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Initial Conv Layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Residual Blocks
        self.res_block1 = ResidualBlock(16, 32)
        self.res_block2 = ResidualBlock(32, 32)
        self.res_block3 = ResidualBlock(32, 64)
        self.res_block4 = ResidualBlock(64, 64)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Initial Conv Layer
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # Residual Blocks with Pooling
        x = self.res_block1(x)
        x = self.pool(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.pool(x)
        x = self.res_block4(x)

        # Global Average Pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Model instantiation
model = CNNModel()

# Loss and Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# To check the model summary and architecture, we can print it.
print(model)


def evaluate(model, data_loader, criterion):
    model.eval()  # 모델을 평가 모드로 전환
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 평가 시에는 역전파 불필요
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)

    print(f'Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

evaluate(model, val_data_loader, criterion)# 성능 평가 함수 (검증 또는 테스트 데이터로 성능 확인)
def evaluate(model, data_loader, criterion):
    model.eval()  # 모델을 평가 모드로 전환
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # 평가 시에는 역전파 불필요
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)

    print(f'Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

evaluate(model, val_data_loader, criterion)