import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score
import torch.nn.functional as F
from torch.cuda.amp import autocast  # ✅ 추가

from trainer.wandb_logger import WandbLogger


def evaluation(model, dataloader, val_dataset, criterion, device, epoch, num_epochs, class_names, logger: WandbLogger):
    model.eval()  # 모델을 평가 모드로 설정
    valid_loss = 0.0
    valid_accuracy = 0
    total = 0
    all_probs = []
    all_labels = []

    with torch.no_grad(): # model의 업데이트 막기
        tbar = tqdm(dataloader)
        for images, labels, img_names in tbar:
            images = images.to(device)
            labels = labels.to(device)

            # 순전파
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 손실과 정확도 계산
            valid_loss += loss.item()
            # torch.max에서 dim 인자에 값을 추가할 경우, 해당 dimension에서 최댓값과 최댓값에 해당하는 인덱스를 반환
            _, predicted = torch.max(outputs, 1)
            valid_accuracy += (predicted == labels).sum().item()
            total += labels.size(0)
            
            
            probs = F.softmax(outputs, dim=1) # 각 클래스에 대한 확률
            preds = torch.argmax(probs, dim=1) # 예측된 클래스 인덱스
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            

            # tqdm의 진행바에 표시될 설명 텍스트를 설정
            tbar.set_description(f"Epoch [{epoch}/{num_epochs}], Valid Loss: {loss.item():.4f}")

    valid_loss = valid_loss / len(dataloader)
    valid_accuracy = valid_accuracy / len(val_dataset)
    all_preds = np.argmax(all_probs, axis=1)
    valid_f1 = f1_score(all_preds, all_labels, average='macro')
    if logger:
        logger.log_confusion_matrix(all_preds, all_labels, class_names=class_names, step=epoch)
        logger.log_failed_predictions(images, preds, labels, image_names=img_names, class_names=class_names, step=epoch) # preds 이거 말고 predicted 넣어도 됨.
        logger.log_predictions(images, preds, labels, class_names, step=epoch)
    
    ret = {
        "valid_loss": valid_loss, 
        "valid_accuracy": valid_accuracy, 
        "valid_f1": valid_f1,
    }

    return model, ret




def evaluation_use_amp(model, dataloader, val_dataset, criterion, device, epoch, num_epochs, class_names, logger: WandbLogger):
    model.eval()
    valid_loss = 0.0
    valid_accuracy = 0
    all_preds_list = []
    all_labels_list = []
    
    # ✅ 로깅을 위한 첫 배치 데이터 저장용 변수
    first_batch_logged = False
    log_images, log_preds, log_labels, log_img_names = None, None, None, None

    with torch.no_grad():
        tbar = tqdm(dataloader, desc=f"Epoch [{epoch}/{num_epochs}] (Eval)")
        for images, labels, img_names in tbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            valid_loss += loss.item()
            
            # ✅ 예측 계산을 한 번만 수행
            predicted = torch.argmax(outputs, 1)
            valid_accuracy += (predicted == labels).sum().item()

            all_preds_list.extend(predicted.cpu().numpy())
            all_labels_list.extend(labels.cpu().numpy())

            # ✅ 로깅을 위해 첫 배치의 데이터만 저장
            if not first_batch_logged:
                log_images = images.cpu()
                log_preds = predicted.cpu()
                log_labels = labels.cpu()
                log_img_names = img_names
                first_batch_logged = True

            tbar.set_postfix(loss=f"{loss.item():.4f}")

    valid_loss /= len(dataloader)
    valid_accuracy /= len(val_dataset)
    valid_f1 = f1_score(all_labels_list, all_preds_list, average='macro')
    
    # ✅ 루프 종료 후 저장해둔 첫 배치 데이터로 로깅
    if logger and first_batch_logged:
        logger.log_confusion_matrix(all_preds_list, all_labels_list, class_names=class_names, step=epoch)
        # log_failed_predictions, log_predictions 등도 저장된 log_ 변수들을 사용
        logger.log_failed_predictions(log_images, log_preds, log_labels, image_names=log_img_names, class_names=class_names, step=epoch)
        logger.log_predictions(log_images, log_preds, log_labels, class_names, step=epoch)
    
    ret = {
        "valid_loss": valid_loss, 
        "valid_accuracy": valid_accuracy, 
        "valid_f1": valid_f1,
    }

    return model, ret