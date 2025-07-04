import os

import wandb
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from dotenv import load_dotenv
from utils.korean_matplot_setting import set_korean_font



class WandbLogger:
    def __init__(self, project_name="my-project", run_name: str=None, config: dict=None, group: str=None, save_path=None):
        self.project = project_name
        self.name = run_name
        self.config = config or {}
        self.group = group
        self.save_path = save_path or "checkpoint.pth"
        load_dotenv()
        self.init()

    def init(self):
        self.finish()
        
        if wandb.run is not None:
            print("🧯 wandb.run이 아직 살아있어요. 강제로 종료합니다.")
            wandb.finish()

        
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        self.run = wandb.init(
            project=self.project,
            name=self.name,
            config=self.config,
            id=None, # 이전 run_id 재사용하지 않기
            resume=False,  # 반드시 새로운 run 시작
            reinit=True,  # 커널 재시작 환경에서 안전하게 새로 시작!
            group=self.group
        )
        set_korean_font()

    def log_metrics(self, metrics_dict, step=None):
        if step is not None:
            wandb.log(metrics_dict, step=step)
        else:
            wandb.log(metrics_dict)

    def log_predictions(self, images, labels, preds, class_names=None, max_samples=16, step=None):
        """이미지 예측 결과 시각화"""
        log_imgs = []
        for i in range(min(len(images), max_samples)):
            img = images[i].detach().cpu()
            img = img.permute(1,2,0).numpy()  # CHW → HWC
            img = (img * 255).astype(np.uint8)

            pred_label = preds[i].item() if isinstance(preds[i], (int, np.integer)) else preds[i]
            true_label = labels[i].item() if isinstance(labels[i], (int, np.integer)) else labels[i]

            caption = f"✅ {class_names[true_label] if class_names else true_label} | 🔮 {class_names[pred_label] if class_names else pred_label}"
            log_imgs.append(wandb.Image(img, caption=caption))
        
        wandb.log({"predictions": log_imgs}, step=step)

    def log_confusion_matrix(self, labels, preds, class_names=None, step=None):
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names if class_names else "auto",
                    yticklabels=class_names if class_names else "auto")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        wandb.log({"confusion_matrix": wandb.Image(plt)}, step=step)
        plt.close()
        
    def log_failed_predictions(self, images, labels, preds, image_names=None, class_names=None, max_samples=16, step=None):
        """정답을 틀린 경우에만 이미지 로그"""
        failed_imgs = []
        for i in range(len(images)):
            true_label = labels[i].item() if hasattr(labels[i], "item") else labels[i]
            pred_label = preds[i].item() if hasattr(preds[i], "item") else preds[i]
            if true_label != pred_label:
                img = images[i].detach().cpu()
                img = img.permute(1, 2, 0).numpy()  # CHW → HWC
                img = (img * 255).astype(np.uint8)

                caption = f"{image_names[i]}:  {class_names[true_label] if class_names else true_label} | Pred: {class_names[pred_label] if class_names else pred_label}"
                failed_imgs.append(wandb.Image(img, caption=caption))

                if len(failed_imgs) >= max_samples:
                    break

        if failed_imgs:
            wandb.log({"failed_predictions": failed_imgs}, step=step)

    def save_model(self, save_path=None):
        path = save_path if save_path else self.save_path
        wandb.save(path)

        # Artifact 업로드 (선택)
        artifact = wandb.Artifact("best-model", type="model")
        artifact.add_file(path)
        wandb.log_artifact(artifact)

    def finish(self):
        wandb.finish()