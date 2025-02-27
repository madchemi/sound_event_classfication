import torch


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

