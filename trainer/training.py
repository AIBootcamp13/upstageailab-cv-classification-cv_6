import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast

from datasets.shack import cutmix_data, mixup_criterion


def training(model, dataloader, train_dataset, optimizer, device, epoch, num_epochs, scaler, criterions, miner, triplet_loss_weight):
    model.train()  # 모델을 학습 모드로 설정
    train_loss = 0.0
    train_accuracy = 0
    all_preds_list = []
    all_labels_list = []
    use_cutmix = np.random.rand() < 0.5
    
    # 손실 함수들을 받아옵니다.
    criterion_focal = criterions['focal']
    criterion_triplet = criterions['triplet']

    tbar = tqdm(dataloader)
    for images, labels, _ in tbar:
        images = images.to(device)
        labels = labels.to(device)
        
    # 순전파
    # 50% 확률로 CutMix 적용
        if use_cutmix:
            mixed_inputs, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=1.0)
            embeddings, outputs = model(mixed_inputs, labels)
        else:
            embeddings, outputs = model(images, labels)

        # Focal Loss 계산
        if use_cutmix:
            loss_focal = mixup_criterion(criterion_focal, outputs, targets_a, targets_b, lam)
        else:
            loss_focal = criterion_focal(outputs, labels)
            
        # Triplet Loss 계산 (CutMix가 적용되지 않은 임베딩을 사용하는 것이 이상적이나, 단순화를 위해 현재 배치 임베딩으로 계산)
        hard_pairs = miner(embeddings, labels)
        loss_triplet = criterion_triplet(embeddings, labels, hard_pairs)
        
        # 최종 손실 계산
        loss = loss_focal + triplet_loss_weight * loss_triplet

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


def training_use_amp(model, dataloader, train_dataset, optimizer, device, epoch, num_epochs, scaler, criterions, miner, triplet_loss_weight):
    model.train()  # 모델을 학습 모드로 설정
    train_loss = 0.0
    train_accuracy = 0
    all_preds_list = []
    all_labels_list = []
    use_cutmix = np.random.rand() < 0.5
  
    # 손실 함수들을 받아옵니다.
    criterion_focal = criterions['focal']
    criterion_triplet = criterions['triplet']

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
                embeddings, outputs = model(mixed_inputs, labels)
            else:
                embeddings, outputs = model(images, labels)
        
            # Focal Loss 계산
            if use_cutmix:
                loss_focal = mixup_criterion(criterion_focal, outputs, targets_a, targets_b, lam)
            else:
                loss_focal = criterion_focal(outputs, labels)
                
            # Triplet Loss 계산 (CutMix가 적용되지 않은 임베딩을 사용하는 것이 이상적이나, 단순화를 위해 현재 배치 임베딩으로 계산)
            hard_pairs = miner(embeddings, labels)
            loss_triplet = criterion_triplet(embeddings, labels, hard_pairs)
            
            # 최종 손실 계산
            loss = loss_focal + triplet_loss_weight * loss_triplet
        
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