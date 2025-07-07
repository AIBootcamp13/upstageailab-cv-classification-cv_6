import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast

from datasets.shack import cutmix_data, mixup_criterion


def training(model, dataloader, train_dataset, criterion, optimizer, device, epoch, num_epochs):
  model.train()  # 모델을 학습 모드로 설정
  train_loss = 0.0
  train_accuracy = 0
  all_preds_list = []
  all_labels_list = []
  use_cutmix = np.random.rand() < 0.5

  tbar = tqdm(dataloader)
  for images, labels, _ in tbar:
      images = images.to(device)
      labels = labels.to(device)
      
    # 순전파
    # 50% 확률로 CutMix 적용
      if use_cutmix:
          mixed_inputs, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=1.0)
          outputs = model(mixed_inputs, labels)
          loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
      else:
          outputs = model(images, labels)
          loss = criterion(outputs, labels)

      # 역전파 및 가중치 업데이트
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # 손실과 정확도 계산
      train_loss += loss.item()
      # torch.max에서 dim 인자에 값을 추가할 경우, 해당 dimension에서 최댓값과 최댓값에 해당하는 인덱스를 반환
      _, predicted = torch.max(outputs, 1)
      
      # CutMix가 적용되었을 때와 아닐 때를 구분하여 정확도 계산
      if use_cutmix: # CutMix가 적용된 경우
          correct_predictions = lam * (predicted == targets_a.data).sum().item() + (1 - lam) * (predicted == targets_b.data).sum().item()
          train_accuracy += correct_predictions
      else: # 일반적인 경우
          train_accuracy += (predicted == labels).sum().item()

      all_preds_list.extend(predicted.detach().cpu().numpy())
      all_labels_list.extend(labels.detach().cpu().numpy())

      # tqdm의 진행바에 표시될 설명 텍스트를 설정
      tbar.set_description(f"Epoch [{epoch}/{num_epochs}], Train Loss: {loss.item():.4f}")

  # 에폭별 학습 결과 출력
  train_loss = train_loss / len(dataloader)
  train_accuracy = train_accuracy / len(train_dataset)
  train_f1 = f1_score(all_labels_list, all_preds_list, average='macro')


  ret = {
      "train_loss": train_loss,
      "train_accuracy": train_accuracy,
      "train_f1": train_f1,
    }

  return model, ret


def training_use_amp(model, dataloader, train_dataset, criterion, optimizer, device, epoch, num_epochs, scaler):
  model.train()  # 모델을 학습 모드로 설정
  train_loss = 0.0
  train_accuracy = 0
  all_preds_list = []
  all_labels_list = []
  use_cutmix = np.random.rand() < 0.5

  tbar = tqdm(dataloader)
  for images, labels, _ in tbar:
      images = images.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()
      
      # 🔥 autocast로 float16 사용
      with autocast():
          # 50% 확률로 CutMix 적용
          if use_cutmix:
              mixed_inputs, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=1.0)
              outputs = model(mixed_inputs, labels)
              loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
          else:
              outputs = model(images, labels)
              loss = criterion(outputs, labels)
      
      
      # ⚙️ AMP-aware backward + step
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()


      # 손실과 정확도 계산
      train_loss += loss.item()
      # torch.max에서 dim 인자에 값을 추가할 경우, 해당 dimension에서 최댓값과 최댓값에 해당하는 인덱스를 반환
      _, predicted = torch.max(outputs, 1)

      # CutMix가 적용되었을 때와 아닐 때를 구분하여 정확도 계산
      if use_cutmix: # CutMix가 적용된 경우
          correct_predictions = lam * (predicted == targets_a.data).sum().item() + (1 - lam) * (predicted == targets_b.data).sum().item()
          train_accuracy += correct_predictions
      else: # 일반적인 경우
          train_accuracy += (predicted == labels).sum().item()

      all_preds_list.extend(predicted.detach().cpu().numpy())
      all_labels_list.extend(labels.detach().cpu().numpy())

      # tqdm의 진행바에 표시될 설명 텍스트를 설정
      tbar.set_description(f"Epoch [{epoch}/{num_epochs}], Train Loss: {loss.item():.4f}")

  # 에폭별 학습 결과 출력
  train_loss = train_loss / len(dataloader)
  train_accuracy = train_accuracy / len(train_dataset)
  train_f1 = f1_score(all_labels_list, all_preds_list, average='macro')
  
  
  ret = {
      "train_loss": train_loss,
      "train_accuracy": train_accuracy,
      "train_f1": train_f1,
    }

  return model, ret