# 한국 문서 분류 프로젝트 - 향상된 탐색적 데이터 분석 (Enhanced EDA)
# 이 코드는 17가지 종류의 한국 문서(신분증, 의료서류 등)를 분류하는 AI 모델을 만들기 위한 데이터 분석입니다.

# 📚 필요한 라이브러리들 불러오기
import os
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from PIL import Image, ImageStat
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import torch
import torchvision.models as models
import torchvision.transforms as T
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 🎨 그래프 스타일 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'NanumGothic', 'Malgun Gothic']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

print("✅ 라이브러리 로딩 완료")
print("✅ 한글 폰트 설정 완료")

# 📁 데이터 경로 설정
class Config:
    """설정 클래스"""
    data_path = './data'
    train_csv_path = f'{data_path}/labels_updated.csv'
    meta_path = f'{data_path}/meta.csv'
    train_img_path = f'{data_path}/train'
    test_data_path = f'{data_path}/test'
    output_path = f'{data_path}/train_valid_set'
    
    # EDA 설정
    n_sample_images = 3  # 클래스별 샘플 이미지 개수
    aspect_ratio_bins = 5  # 가로세로 비율 구간 개수
    random_state = 42

config = Config()

# 📊 데이터 로더 클래스
class DocumentDataAnalyzer:
    """문서 데이터 분석을 위한 클래스"""
    
    def __init__(self, config):
        self.config = config
        self.data_df = None
        self.meta_df = None
        self.class_map = None
        self.size_analysis_df = None
        
    def load_data(self):
        """데이터 파일들을 로드합니다."""
        print("📊 데이터 파일 로딩 중...")
        
        # CSV 파일 읽기
        self.data_df = pd.read_csv(self.config.train_csv_path, header=0)
        self.meta_df = pd.read_csv(self.config.meta_path)
        
        # 타입 변환
        self.data_df["target"] = self.data_df["target"].astype(int)
        self.meta_df["target"] = self.meta_df["target"].astype(int)
        
        # 클래스 매핑 생성
        self.class_map = dict(zip(self.meta_df["target"], self.meta_df["class_name"]))
        
        print(f"✅ 훈련 데이터: {len(self.data_df)}개")
        print(f"✅ 클래스 개수: {len(self.meta_df)}개")
        
        return self
    
    def analyze_class_distribution(self):
        """클래스별 데이터 분포를 분석합니다."""
        print("\n📊 클래스 분포 분석 중...")
        
        # 클래스별 개수 계산
        label_counts = self.data_df["target"].value_counts().sort_index()
        label_dist = pd.DataFrame({
            'target': label_counts.index,
            'count': label_counts.values
        })
        
        # 클래스 이름 추가
        label_dist = pd.merge(label_dist, self.meta_df, on="target")
        
        # 통계 출력
        print(f"\n📈 클래스별 데이터 개수:")
        print(label_dist[["target", "class_name", "count"]].to_string(index=False))
        
        print(f"\n⚠️ 데이터 불균형 분석:")
        print(f"   최대: {label_dist['count'].max()}개")
        print(f"   최소: {label_dist['count'].min()}개")
        print(f"   평균: {label_dist['count'].mean():.1f}개")
        print(f"   표준편차: {label_dist['count'].std():.1f}")
        print(f"   불균형 비율: {label_dist['count'].max() / label_dist['count'].min():.2f}배")
        
        return label_dist
    
    def plot_class_distribution(self, label_dist):
        """클래스 분포를 시각화합니다."""
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        # 1. 막대 그래프
        ax1 = axes[0]
        bars = ax1.bar(range(len(label_dist)), label_dist['count'], 
                      color=sns.color_palette("husl", len(label_dist)))
        ax1.set_xlabel('문서 종류')
        ax1.set_ylabel('데이터 개수')
        ax1.set_title('문서 종류별 데이터 분포', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(label_dist)))
        ax1.set_xticklabels(label_dist['class_name'], rotation=45, ha='right')
        
        # 막대 위에 숫자 표시
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. 파이 차트
        ax2 = axes[1]
        colors = sns.color_palette("husl", len(label_dist))
        wedges, texts, autotexts = ax2.pie(label_dist['count'], 
                                          labels=label_dist['class_name'],
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          textprops={'fontsize': 8})
        ax2.set_title('문서 종류별 비율', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_image_dimensions(self):
        """이미지 크기와 품질을 분석합니다."""
        print("\n🔍 이미지 크기 및 품질 분석 중...")
        
        results = []
        failed_images = []
        
        for _, row in tqdm(self.data_df.iterrows(), total=len(self.data_df), 
                          desc="이미지 분석"):
            img_path = os.path.join(self.config.train_img_path, row["ID"])
            
            try:
                # OpenCV로 이미지 읽기
                img = cv2.imread(img_path)
                if img is None:
                    failed_images.append(row["ID"])
                    continue
                
                h, w, c = img.shape
                
                # PIL로 추가 정보 얻기
                pil_img = Image.open(img_path)
                file_size = os.path.getsize(img_path) / (1024 * 1024)  # MB
                
                # 이미지 통계
                stat = ImageStat.Stat(pil_img)
                brightness = sum(stat.mean) / len(stat.mean)  # 평균 밝기
                
                results.append({
                    'ID': row["ID"],
                    'target': row["target"],
                    'class_name': self.class_map[row["target"]],
                    'width': w,
                    'height': h,
                    'channels': c,
                    'aspect_ratio': round(w / h, 3),
                    'total_pixels': w * h,
                    'file_size_mb': round(file_size, 3),
                    'brightness': round(brightness, 1)
                })
                
            except Exception as e:
                failed_images.append(row["ID"])
                print(f"❌ 오류 발생: {row['ID']} - {str(e)}")
        
        self.size_analysis_df = pd.DataFrame(results)
        
        if failed_images:
            print(f"⚠️ 읽기 실패한 이미지: {len(failed_images)}개")
        
        print(f"✅ 분석 완료: {len(results)}개 이미지")
        return self.size_analysis_df
    
    def plot_image_analysis(self):
        """이미지 분석 결과를 시각화합니다."""
        if self.size_analysis_df is None:
            print("❌ 이미지 분석을 먼저 실행하세요.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 1. 이미지 높이 분포
        sns.histplot(data=self.size_analysis_df, x='height', bins=30, ax=axes[0])
        axes[0].set_title('이미지 높이 분포')
        axes[0].set_xlabel('높이 (픽셀)')
        
        # 2. 이미지 너비 분포
        sns.histplot(data=self.size_analysis_df, x='width', bins=30, ax=axes[1])
        axes[1].set_title('이미지 너비 분포')
        axes[1].set_xlabel('너비 (픽셀)')
        
        # 3. 가로세로 비율 분포
        sns.histplot(data=self.size_analysis_df, x='aspect_ratio', bins=30, ax=axes[2])
        axes[2].set_title('가로세로 비율 분포')
        axes[2].set_xlabel('가로세로 비율 (W/H)')
        
        # 4. 파일 크기 분포
        sns.histplot(data=self.size_analysis_df, x='file_size_mb', bins=30, ax=axes[3])
        axes[3].set_title('파일 크기 분포')
        axes[3].set_xlabel('파일 크기 (MB)')
        
        # 5. 클래스별 평균 가로세로 비율
        class_aspect = self.size_analysis_df.groupby('class_name')['aspect_ratio'].mean().sort_values()
        sns.barplot(x=class_aspect.values, y=class_aspect.index, ax=axes[4])
        axes[4].set_title('클래스별 평균 가로세로 비율')
        axes[4].set_xlabel('평균 가로세로 비율')
        
        # 6. 총 픽셀 수 vs 파일 크기
        axes[5].scatter(self.size_analysis_df['total_pixels'], 
                       self.size_analysis_df['file_size_mb'], 
                       alpha=0.6, c=self.size_analysis_df['target'], cmap='tab20')
        axes[5].set_title('총 픽셀 수 vs 파일 크기')
        axes[5].set_xlabel('총 픽셀 수')
        axes[5].set_ylabel('파일 크기 (MB)')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_class_characteristics(self):
        """클래스별 특성을 상세 분석합니다."""
        if self.size_analysis_df is None:
            print("❌ 이미지 분석을 먼저 실행하세요.")
            return
        
        print("\n📊 클래스별 이미지 특성 분석:")
        
        # 클래스별 통계
        class_stats = self.size_analysis_df.groupby('class_name').agg({
            'width': ['mean', 'std', 'min', 'max'],
            'height': ['mean', 'std', 'min', 'max'],
            'aspect_ratio': ['mean', 'std', 'min', 'max'],
            'file_size_mb': ['mean', 'std'],
            'brightness': ['mean', 'std']
        }).round(2)
        
        # 컬럼명 정리
        class_stats.columns = [f'{col[0]}_{col[1]}' for col in class_stats.columns]
        
        print(class_stats.to_string())
        
        # 박스플롯으로 시각화
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 가로세로 비율
        sns.boxplot(data=self.size_analysis_df, x='class_name', y='aspect_ratio', ax=axes[0,0])
        axes[0,0].set_title('클래스별 가로세로 비율 분포')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 파일 크기
        sns.boxplot(data=self.size_analysis_df, x='class_name', y='file_size_mb', ax=axes[0,1])
        axes[0,1].set_title('클래스별 파일 크기 분포')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 밝기
        sns.boxplot(data=self.size_analysis_df, x='class_name', y='brightness', ax=axes[1,0])
        axes[1,0].set_title('클래스별 밝기 분포')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 총 픽셀 수
        sns.boxplot(data=self.size_analysis_df, x='class_name', y='total_pixels', ax=axes[1,1])
        axes[1,1].set_title('클래스별 총 픽셀 수 분포')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return class_stats
    
    def stratified_split(self, test_size=0.2):
        """계층적 분할을 수행합니다."""
        if self.size_analysis_df is None:
            print("❌ 이미지 분석을 먼저 실행하세요.")
            return None, None
        
        print(f"\n🔄 계층적 데이터 분할 중... (검증 비율: {test_size*100}%)")
        
        # 가로세로 비율 구간 생성
        self.size_analysis_df['aspect_bin'] = pd.qcut(
            self.size_analysis_df['aspect_ratio'], 
            q=self.config.aspect_ratio_bins, 
            labels=False, 
            duplicates='drop'
        )
        
        # 계층 생성 (클래스 + 비율구간)
        self.size_analysis_df['strata'] = (
            self.size_analysis_df['target'].astype(str) + '_' + 
            self.size_analysis_df['aspect_bin'].astype(str)
        )
        
        # 계층별 최소 샘플 수 확인
        strata_counts = self.size_analysis_df['strata'].value_counts()
        min_samples = 2
        
        # 충분한 샘플이 있는 계층만 사용
        valid_strata = strata_counts[strata_counts >= min_samples].index
        df_valid = self.size_analysis_df[
            self.size_analysis_df['strata'].isin(valid_strata)
        ].copy()
        
        # 계층적 분할
        sss = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=test_size, 
            random_state=self.config.random_state
        )
        
        train_idx, val_idx = next(sss.split(df_valid, df_valid['strata']))
        
        train_df = df_valid.iloc[train_idx].reset_index(drop=True)
        val_df = df_valid.iloc[val_idx].reset_index(drop=True)
        
        # 불충분한 샘플의 계층은 검증 데이터에 추가
        df_insufficient = self.size_analysis_df[
            ~self.size_analysis_df['strata'].isin(valid_strata)
        ].copy()
        
        if len(df_insufficient) > 0:
            val_df = pd.concat([val_df, df_insufficient]).reset_index(drop=True)
        
        print(f"✅ 분할 완료!")
        print(f"   훈련 데이터: {len(train_df)}개")
        print(f"   검증 데이터: {len(val_df)}개")
        print(f"   전체 데이터: {len(train_df) + len(val_df)}개")
        
        # 분할 품질 확인
        self._validate_split_quality(train_df, val_df)
        
        return train_df, val_df
    
    def _validate_split_quality(self, train_df, val_df):
        """분할 품질을 검증합니다."""
        print("\n🔍 분할 품질 검증:")
        
        # 클래스별 분포 비교
        train_dist = train_df['target'].value_counts(normalize=True).sort_index()
        val_dist = val_df['target'].value_counts(normalize=True).sort_index()
        
        # 분포 차이 계산
        common_classes = train_dist.index.intersection(val_dist.index)
        if len(common_classes) > 0:
            dist_diff = abs(train_dist[common_classes] - val_dist[common_classes]).mean()
            print(f"   평균 클래스 분포 차이: {dist_diff:.3f}")
        
        # 가로세로 비율 분포 비교
        from scipy import stats
        aspect_stat, aspect_p = stats.ks_2samp(train_df['aspect_ratio'], val_df['aspect_ratio'])
        print(f"   가로세로 비율 분포 유사도 (p-value): {aspect_p:.3f}")
        
        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 클래스 분포 비교
        comparison_df = pd.DataFrame({
            'train': train_dist,
            'val': val_dist
        }).fillna(0)
        
        comparison_df.plot(kind='bar', ax=axes[0])
        axes[0].set_title('훈련/검증 데이터 클래스 분포 비교')
        axes[0].set_xlabel('클래스')
        axes[0].set_ylabel('비율')
        axes[0].legend()
        axes[0].tick_params(axis='x', rotation=45)
        
        # 가로세로 비율 분포 비교
        sns.kdeplot(data=train_df, x='aspect_ratio', label='훈련', ax=axes[1])
        sns.kdeplot(data=val_df, x='aspect_ratio', label='검증', ax=axes[1])
        axes[1].set_title('가로세로 비율 분포 비교')
        axes[1].set_xlabel('가로세로 비율')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, train_df, val_df):
        """분석 결과를 저장합니다."""
        print("\n💾 결과 저장 중...")
        
        # 출력 디렉토리 생성
        os.makedirs(self.config.output_path, exist_ok=True)
        
        # 분할된 데이터 저장
        train_df.to_csv(f"{self.config.output_path}/train_enhanced.csv", index=False)
        val_df.to_csv(f"{self.config.output_path}/val_enhanced.csv", index=False)
        
        # 전체 분석 결과 저장
        if self.size_analysis_df is not None:
            self.size_analysis_df.to_csv(f"{self.config.output_path}/image_analysis.csv", index=False)
        
        print("✅ 저장 완료!")
        print(f"   저장 위치: {self.config.output_path}")
    
    def generate_summary_report(self):
        """종합 분석 리포트를 생성합니다."""
        print("\n📋 종합 분석 리포트")
        print("=" * 50)
        
        if self.data_df is not None:
            print(f"📊 기본 정보:")
            print(f"   총 이미지 수: {len(self.data_df):,}개")
            print(f"   클래스 수: {len(self.meta_df)}개")
            
        if self.size_analysis_df is not None:
            print(f"\n🖼️ 이미지 특성:")
            print(f"   평균 해상도: {self.size_analysis_df['width'].mean():.0f} x {self.size_analysis_df['height'].mean():.0f}")
            print(f"   평균 파일 크기: {self.size_analysis_df['file_size_mb'].mean():.2f} MB")
            print(f"   평균 가로세로 비율: {self.size_analysis_df['aspect_ratio'].mean():.2f}")
            print(f"   평균 밝기: {self.size_analysis_df['brightness'].mean():.1f}")
            
            # 클래스별 특징
            print(f"\n📈 클래스별 특징:")
            for class_id, class_name in self.class_map.items():
                class_data = self.size_analysis_df[self.size_analysis_df['target'] == class_id]
                if len(class_data) > 0:
                    print(f"   {class_name}: {len(class_data)}개, "
                          f"평균 비율 {class_data['aspect_ratio'].mean():.2f}")
        
        print(f"\n🎯 권장사항:")
        print(f"   - 데이터 증강 기법 적용 (회전, 크기 조절, 밝기 조절)")
        print(f"   - 클래스 불균형 해결 (가중치 조정 또는 샘플링)")
        print(f"   - 이미지 전처리 표준화 (크기 정규화, 밝기 보정)")
        print(f"   - 교차 검증을 통한 모델 성능 검증")

# 🚀 메인 실행 코드
def main():
    """메인 실행 함수"""
    print("🚀 한국 문서 분류 프로젝트 EDA 시작!")
    print("=" * 60)
    
    # 분석기 초기화 및 실행
    analyzer = DocumentDataAnalyzer(config)
    
    # 1. 데이터 로드
    analyzer.load_data()
    
    # 2. 클래스 분포 분석
    label_dist = analyzer.analyze_class_distribution()
    analyzer.plot_class_distribution(label_dist)
    
    # 3. 이미지 분석
    analyzer.analyze_image_dimensions()
    analyzer.plot_image_analysis()
    
    # 4. 클래스별 특성 분석
    class_stats = analyzer.analyze_class_characteristics()
    
    # 5. 데이터 분할
    train_df, val_df = analyzer.stratified_split(test_size=0.2)
    
    # 6. 결과 저장
    if train_df is not None and val_df is not None:
        analyzer.save_results(train_df, val_df)
    
    # 7. 종합 리포트
    analyzer.generate_summary_report()
    
    print("\n🎉 EDA 완료!")

if __name__ == "__main__":
    main()