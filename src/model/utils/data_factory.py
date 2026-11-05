from typing import Dict
from torch.utils.data import DataLoader

from .datasets.FHM_data import FHM_Dataset, FHM_Collator
from .datasets.MAMI_data import MAMI_Dataset, MAMI_Collator
from .datasets.ToxiCN_data import ToxiCN_Dataset, ToxiCN_Collator

def DataloaderFactory(dataset: str, **kwargs):
    # return dataloader
    dataset = None
    collator = None
    match dataset:
        case 'FHM':
            dataset = FHM_Dataset(**kwargs)
            collator = FHM_Collator()
        case 'MAMI':
            dataset = MAMI_Dataset(**kwargs)
            collator = MAMI_Collator()
        case 'ToxiCN':
            dataset = ToxiCN_Dataset(**kwargs)
            collator = ToxiCN_Collator()
        case _:
            raise NotImplementedError(f"Dataset {dataset} not supported")
    
    dataloader = DataLoader(dataset, collator=collator, **kwargs)
    return dataloader

def DataDfFactory(dataset: str, **kwargs):
    match dataset:
        case 'FHM':
            dataset = FHM_Dataset(**kwargs)
        case 'MAMI':
            dataset = MAMI_Dataset(**kwargs)
        case 'ToxiCN':
            dataset = ToxiCN_Dataset(**kwargs)
        case _:
            raise NotImplementedError(f"Dataset {dataset} not supported")
    return dataset.get_data(**kwargs)