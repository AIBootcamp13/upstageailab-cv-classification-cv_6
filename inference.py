import os

import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from config.config import load_config
from models import get_model
from datasets import get_dataset
from datasets.transforms import build_unified_transforms


cfg = load_config("config/main_config.yaml")
ModelClass = get_model(cfg['MODEL'])
DatasetClass = get_dataset(cfg['DATASET'])
num_classes = 17
num_workers = os.cpu_count() // 2
output_root = './output'
data_path = './data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_transform = build_unified_transforms(cfg["transforms"]["test"])

tst_dataset = DatasetClass(
    f"{data_path}/sample_submission.csv",
    f"{data_path}/test/",
    transform=test_transform
)

tst_loader = DataLoader(
    tst_dataset,
    batch_size=cfg["BATCH_SIZE"],
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True
)


preds_list = []
model = ModelClass(num_classes).to(device)
model.load_state_dict(torch.load(f"{output_root}/checkpoint.pth", map_location="cpu"))

model.eval()
for image, _, _ in tqdm(tst_loader):
    image = image.to(device)

    with torch.no_grad():
        preds = model(image)
    preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
    
    
pred_df = pd.DataFrame(tst_dataset.df, columns=['ID', 'target'])
pred_df['target'] = preds_list


sample_submission_df = pd.read_csv(f"{data_path}/sample_submission.csv")
assert (sample_submission_df['ID'] == pred_df['ID']).all()

pred_df.to_csv("pred.csv", index=False)