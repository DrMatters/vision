import albumentations as albu
import pandas as pd
from torch.utils import data


class MarketDataset(data.Dataset):
    def __init__(self, folder: str, index_df: pd.DataFrame = None,
                 transforms=albu.Compose([albu.HorizontalFlip()])):
        self.folder = folder
        self.index_df = index_df
        self.transforms = transforms
        self.img_ids =

    def __getitem__(self, item):
        pass

    def __len__(self):
        return len(self.index_df)
