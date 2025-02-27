import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


from model import CNNModel
from mfcc_feature_extraction import preprocessing, fine_max_pad_len
#from train_validation import train_model, validate_model

from numpy import random
from torchvision import datasets
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader , random_split
from torchvision import models
"""
#for i in range(10):
#  os.makedirs('C:/Users/seoji/Desktop/gist/gist_makerton/urban_numpy_set/'+str(i), exist_ok=True)
max_pad_len = fine_max_pad_len('C:/Users/seoji/Desktop/gist/gist_makerton/archive/fold')

for i in range(10):
   preprocessing('C:/Users/seoji/Desktop/gist/gist_makerton/archive/fold'+str(i+1)+'/', 'C:/Users/seoji/Desktop/gist/gist_makerton/urban_numpy_set/',max_pad_len)
"""
dataset = ImageFolder(root = 'C:/Users/seoji/Desktop/gist/gist_makerton/image_set/',transform=T.Compose([ T.Resize((64,64)),T.ToTensor()]))

train_size = int(0.8*len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset,[train_size, val_size])

train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle = True)
val_data_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False)
    


num_classes = len(os.listdir('C:/Users/seoji/Desktop/gist/gist_makerton/urban_numpy_set/'))  # 예시: 클래스가 10개인 경우

# 3. 모델을 GPU로 보내기 (가능한 경우)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ft = models.resnet34(weights=models.ResNet34_Weights.DEFAULT) 
num_ftrs = model_ft.fc.in_features 
model_ft.fc = nn.Linear(num_ftrs, 10) 

model= model_ft.to(device)
# 4. 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.003)

num_epochs = 500
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []


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




def validate_model(model, val_loader, criterion, device):
    model.eval()  # 모델을 평가 모드로 설정
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # 검증 시에는 그래디언트를 계산하지 않음
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 순전파
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 손실 계산
            running_loss += loss.item()

            # 정확도 계산
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels).item()
            total_predictions += labels.size(0)

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct_predictions / total_predictions
    return epoch_loss, epoch_acc


for epoch in range(num_epochs):
    train_loss, train_acc = train_model(model,train_data_loader,criterion, optimizer, device)
    val_loss, val_acc = validate_model(model, val_data_loader ,criterion, device)

    # 각 epoch의 손실과 정확도를 저장
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}')

print("Training complete.")

model_dir = 'C:/Users/seoji/Desktop/gist/gist_makerton/'

# 모델 저장 파일 경로 (확장자 포함)
model_path = os.path.join(model_dir, 'model_parameter_Pre_ResNet34_{epoch}.pth')

# 폴더가 존재하지 않으면 생성
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 모델 저장
try:
    torch.save(model.state_dict(), model_path)
    print(f"Model saved successfully to {model_path}")
except Exception as e:
    print(f"Error saving model: {e}")





epochs = range(1, num_epochs + 1)

# Loss 그래프
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'bo-', label='Train Loss')
plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
plt.title('Train and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy 그래프
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'bo-', label='Train Accuracy')
plt.plot(epochs, val_accuracies, 'ro-', label='Validation Accuracy')
plt.title('Train and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
