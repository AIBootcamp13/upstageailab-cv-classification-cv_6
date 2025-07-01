__all__ = [
    "plot_sample_images",
    "analyze_dataset",
    "generate_quality_report",
    ]

import os
from collections import defaultdict
import pandas as pd
import numpy as np
from PIL import Image
import cv2

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import torch
from tqdm import tqdm



def plot_sample_images(df, image_dir, label_col='target', filename_col='ID', class_names=None, samples_per_class=5):
    """
        간단하게 셈플 이미지 출력하는 코드
    """
    unique_labels = df[label_col].unique()
    n_classes = len(unique_labels)
    plt.figure(figsize=(samples_per_class * 2, n_classes * 2))

    for i, label in enumerate(sorted(unique_labels)):
        samples = df[df[label_col] == label].sample(samples_per_class, random_state=42)
        for j, img_name in enumerate(samples[filename_col]):
            img_path = os.path.join(image_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ax = plt.subplot(n_classes, samples_per_class, i * samples_per_class + j + 1)
            ax.imshow(img)
            ax.axis("off")
            if j == 0 and class_names:
                ax.set_title(class_names[label], fontsize=10)
    plt.tight_layout()
    plt.show()
    

## 자동 분석

def compute_sharpness_score(gray_img):
    return cv2.Laplacian(gray_img, cv2.CV_64F).var()

def estimate_noise(gray_img):
    return gray_img.std()

def extract_feature(pil_img, model, transform):
    tensor = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        feat = model(tensor)
    return feat.view(-1).numpy()

# 🔸 2. 자동 분석 함수
def analyze_dataset(csv_path, img_dir, model, transform, class_map=None, samples_per_class=100):
    df = pd.read_csv(csv_path, header=0)
    stats = defaultdict(list)

    for label in sorted(df["target"].unique()):
        subset = df[df["target"] == label].sample(
            min(samples_per_class, len(df[df["target"] == label])), random_state=42
        )

        features = []

        for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"Class {label}"):
            img_path = os.path.join(img_dir, row["ID"])
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            sharpness = compute_sharpness_score(gray)
            noise = estimate_noise(gray)
            feat = extract_feature(pil, model, transform)

            stats["label"].append(label)
            stats["class_name"].append(class_map[label] if class_map else f"class_{label}")
            stats["sharpness"].append(sharpness)
            stats["noise"].append(noise)
            features.append(feat)

        # 클래스 내 다양성 계산 (벡터 분산)
        features = np.stack(features)
        intra_var = np.mean(np.var(features, axis=0))
        stats["intra_variance"].extend([intra_var] * len(subset))

    return pd.DataFrame(stats)




def generate_quality_report(df, save_path="quality_report.pdf", topk=5):
    """
    df: ['class_name', 'sharpness', 'noise', 'intra_variance'] 포함한 DataFrame
    """
    class_stats = df.groupby("class_name")[["sharpness", "noise", "intra_variance"]].mean().reset_index()
    
    class_stats["risk_score"] = (
        class_stats["sharpness"].rank(ascending=True) +
        class_stats["noise"].rank(ascending=False) +
        class_stats["intra_variance"].rank(ascending=False)
    )
    class_stats["risk_score"] = class_stats["risk_score"].round(1)
    top_risky = class_stats.sort_values("risk_score", ascending=True).head(topk)


    with PdfPages(save_path) as pdf:
        
        # 🔹 1. 샤프니스 그래프
        plt.figure(figsize=(10, 4))
        sns.barplot(data=class_stats.sort_values("sharpness"), x="sharpness", y="class_name", palette="Blues_r")
        plt.title("평균 선명도 (Sharpness) - 낮을수록 흐림 가능성 ↑")
        plt.xlabel("Variance of Laplacian")
        plt.ylabel("Class")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # 🔹 2. 노이즈 수준 그래프
        plt.figure(figsize=(10, 4))
        sns.barplot(data=class_stats.sort_values("noise"), x="noise", y="class_name", palette="Reds_r")
        plt.title("평균 노이즈 수준 (Pixel StdDev)")
        plt.xlabel("Std Dev")
        plt.ylabel("Class")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # 🔹 3. 클래스 다양성 (Intra-class variance)
        plt.figure(figsize=(10, 4))
        sns.barplot(data=class_stats.sort_values("intra_variance"), x="intra_variance", y="class_name", palette="Purples_r")
        plt.title("클래스 내부 다양성 (Intra-class Feature Variance)")
        plt.xlabel("Feature Variance")
        plt.ylabel("Class")
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # 위험 클래스 topk 테이블
        fig, ax = plt.subplots(figsize=(10, 1 + len(top_risky) * 0.5))
        ax.axis("off")
        table = ax.table(cellText=top_risky.values, colLabels=top_risky.columns, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        plt.title(f"위험 클래스 Top {topk} (종합 Risk Score 기준)")
        pdf.savefig()
        plt.close()


        # 🔹 4. 테이블 요약
        fig, ax = plt.subplots(figsize=(12, 0.5 + len(class_stats) * 0.4))
        ax.axis("off")
        tbl = ax.table(cellText=class_stats.values,
                       colLabels=class_stats.columns,
                       loc="center",
                       cellLoc="center")
        tbl.auto_set_font_size(False)      # 자동 조정 끄기
        tbl.set_fontsize(11)               # 원하는 폰트 크기 설정
        tbl.scale(1, 1.5)
        plt.title("클래스별 통계 요약")
        pdf.savefig()
        plt.close()

    print(f"✅ 리포트 저장 완료: {save_path}")
