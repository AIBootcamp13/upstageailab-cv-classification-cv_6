import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T


def rand_bbox(size, lam):
    """ 랜덤한 바운딩 박스를 생성합니다. """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    '''실제 학습 루프를 위한 올바른 CutMix 함수'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    # ❗️ 시각화용 코드와 달리, 전체 배치 크기에 맞게 랜덤 인덱스를 생성합니다.
    index = torch.randperm(batch_size).to(x.device)

    # 레이블도 섞어줍니다.
    y_a, y_b = y, y[index]
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    
    mixed_x = x.clone()
    # ❗️ 이제 mixed_x와 x[index, ...]의 배치 크기가 동일하여 오류가 발생하지 않습니다.
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return mixed_x, y_a, y_b, lam

def mixup_data_visualize(x, alpha=1.0):
    # Mixup도 두 이미지만 있으면 됩니다.
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    index = torch.tensor([1, 0])
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        # 람다 값을 베타 분포에서 샘플링
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    # 랜덤하게 섞을 인덱스를 생성
    index = torch.randperm(batch_size).to(device)

    # 이미지와 레이블을 섞음
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''Computes the loss for mixed inputs'''
    # 섞인 두 레이블에 대해 각각 손실을 계산한 후 람다 비율로 합침
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def visualize(original_images, result_image, augmentation_name):
    """ 원본 이미지 2개와 결과 이미지를 나란히 보여주는 함수 """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Matplotlib는 (H, W, C) 형태의 이미지를 기대하지만,
    # PyTorch 텐서는 (C, H, W) 형태이므로 .permute()로 차원 순서를 변경합니다.
    img1_display = original_images[0].permute(1, 2, 0)
    img2_display = original_images[1].permute(1, 2, 0)
    result_display = result_image.permute(1, 2, 0)
    
    axes[0].imshow(img1_display)
    axes[0].set_title("Original Image 1")
    axes[0].axis('off')
    
    axes[1].imshow(img2_display)
    axes[1].set_title("Original Image 2")
    axes[1].axis('off')
    
    axes[2].imshow(result_display)
    axes[2].set_title(f"Result ({augmentation_name})")
    axes[2].axis('off')
    
    plt.show()


if __name__ == "__main__":
    # ❗️ 여기에 준비한 이미지 파일 경로를 입력하세요.
    data_path = './data'
    IMAGE_PATH_1 = f'{data_path}/train/0a4adccbb7fe73e0.jpg'
    IMAGE_PATH_2 = f'{data_path}/train/0a26fc63987e05eb.jpg'

    try:
        # 1. 이미지 불러오기 및 전처리
        # 시각화를 위해 Normalize는 제외하고 텐서로만 변환합니다.
        transform = T.Compose([
            T.Resize((600, 600)),
            T.ToTensor(),
        ])
    
        img1 = Image.open(IMAGE_PATH_1).convert('RGB')
        img2 = Image.open(IMAGE_PATH_2).convert('RGB')
    
        img1_tensor = transform(img1)
        img2_tensor = transform(img2)
    
        # 2. 두 이미지를 하나의 배치(batch)로 묶기
        # CutMix/Mixup 함수는 배치 입력을 가정하므로 [2, 3, 600, 600] 형태로 만듭니다.
        batch_images = torch.stack([img1_tensor, img2_tensor])
    
        # 3. CutMix 적용 및 시각화
        cutmix_result_batch = cutmix_data(batch_images, alpha=1.0)
        visualize(batch_images, cutmix_result_batch[0], "CutMix")
        
        # 4. Mixup 적용 및 시각화
        mixup_result_batch = mixup_data_visualize(batch_images, alpha=0.4) # Mixup은 alpha값을 조금 낮춰야 보기 좋습니다.
        visualize(batch_images, mixup_result_batch[0], "Mixup")

    except FileNotFoundError:
        print("이미지 경로를 올바르게 설정해주세요.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")