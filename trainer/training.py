from tqdm import tqdm
import torch
from sklearn.metrics import f1_score

def training(model, dataloader, train_dataset, criterion, optimizer, device, epoch, num_epochs):
  model.train()  # 모델을 학습 모드로 설정
  train_loss = 0.0
  train_accuracy = 0
  preds_list = []
  targets_list = []

  tbar = tqdm(dataloader)
  for images, labels, _ in tbar:
      images = images.to(device)
      labels = labels.to(device)

      # 순전파
      outputs = model(images)
      loss = criterion(outputs, labels)

      # 역전파 및 가중치 업데이트
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # 손실과 정확도 계산
      train_loss += loss.item()
      # torch.max에서 dim 인자에 값을 추가할 경우, 해당 dimension에서 최댓값과 최댓값에 해당하는 인덱스를 반환
      _, predicted = torch.max(outputs, 1)
      train_accuracy += (predicted == labels).sum().item()
      preds_list.extend(outputs.argmax(dim=1).detach().cpu().numpy())
      targets_list.extend(labels.detach().cpu().numpy())

      # tqdm의 진행바에 표시될 설명 텍스트를 설정
      tbar.set_description(f"Epoch [{epoch}/{num_epochs}], Train Loss: {loss.item():.4f}")

  # 에폭별 학습 결과 출력
  train_loss = train_loss / len(dataloader)
  train_accuracy = train_accuracy / len(train_dataset)
  train_f1 = f1_score(targets_list, preds_list, average='macro')
  
  
  ret = {
      "train_loss": train_loss,
      "train_accuracy": train_accuracy,
      "train_f1": train_f1,
    }

  return model, ret