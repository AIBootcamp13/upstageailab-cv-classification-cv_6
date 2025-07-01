import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score
import torch.nn.functional as F

from trainer.wandb_logger import WandbLogger


def evaluation(model, dataloader, val_dataset, criterion, device, epoch, num_epochs, class_names, logger: WandbLogger):
    model.eval()  # 모델을 평가 모드로 설정
    valid_loss = 0.0
    valid_accuracy = 0
    total = 0
    all_probs = []
    all_labels = []
    failed_imgs, failed_preds, failed_labels, failed_names = [], [], [], []

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