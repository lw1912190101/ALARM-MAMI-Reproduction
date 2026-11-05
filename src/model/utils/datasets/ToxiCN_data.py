from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import pandas as pd

from .base_data import Base_Dataset


class ToxiCN_Dataset(Base_Dataset):
    def __init__(self, split: str, **kargs):
        super(ToxiCN_Dataset, self).__init__()
        self.data_path = Path('data/ToxiCN')
        self.img_path = self.data_path / 'img'
        self.data = self.get_data(split)
    
    def get_data(self, split: str) -> pd.DataFrame:
        data = pd.read_json('data/ToxiCN/data.jsonl', orient='records', lines=True)
        data = data[data['split'] == split]
        return data
    
    def get_full_data(self) -> pd.DataFrame:
        data = pd.read_json('data/ToxiCN/data.jsonl', orient='records', lines=True)
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        mid = item['mid']
        img = self.image_path / f'{mid.split('_')[1]}.jpg'
        img = Image.open(img).convert("RGB")
        label = item['label']
        text = item['text']
        return {
            'id': mid,
            'img': img,
            'label': label,
            'text': text
        }


class ToxiCN_Collator:
    def __init__(self):
        pass

    def __call__(self, batch):
        mid = [item['id'] for item in batch]
        img = [item['img'] for item in batch]
        label = [item['label'] for item in batch]
        text = [item['text'] for item in batch]
        return {
            'id': mid,
            'img': img,
            'label': label,
            'text': text
        }
