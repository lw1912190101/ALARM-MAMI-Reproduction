from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import pandas as pd

from .base_data import Base_Dataset


class FHM_Dataset(Base_Dataset):
    def __init__(self, split: str, **kargs):
        super(FHM_Dataset, self).__init__()
        self.data_path = Path('data/FHM')
        self.img_path = self.data_path / 'img'
        self.data = self.get_data(split)
    
    def get_data(self, split: str) -> pd.DataFrame:
        data = pd.read_json('data/FHM/data.jsonl', orient='records', lines=True)
        data = data[data['split'] == split]
        return data
    
    def get_full_data(self) -> pd.DataFrame:
        data = pd.read_json('data/FHM/data.jsonl', orient='records', lines=True)
        return data

    def get_img(self, mid: str) -> Image.Image:
        img_path = self.img_path / f'{mid}.png'
        return Image.open(img_path).convert('RGB')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        mid = item['mid']
        img = self.get_img(mid)
        label = item['label']
        text = item['text']
        return {
            'id': mid,
            'img': img,
            'label': label,
            'text': text
        }

class FHM_Collator:
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