from typing import Dict, List
from dataclasses import dataclass, field
import torch
import os
import json
from PIL import Image

@dataclass
class FolderResults():
    prompt: List[str] = field(default_factory=list)
    total_num: int = 0
    correct_num: int = 0
    elapsed_time: float = 0
    dist_sum: float = 0
    mean_dist: float = 0
    accuracy: float = 0

@dataclass
class EvaluationResults():
    global_info: FolderResults
    dirs: Dict[str, FolderResults] = field(default_factory=dict)


class InferDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path='./data/squirrels_head/'):
        super(InferDataset, self,).__init__()

        self.pil_imgs = sorted([os.path.join(dir_path, x) for x in list(filter(lambda x: x.endswith('jpg'), os.listdir(dir_path)))])
        self.points = sorted([os.path.join(dir_path, x) for x in list(filter(lambda x: x.endswith('json'), os.listdir(dir_path)))])

    def __len__(self):
        return len(self.pil_imgs)

    def __getitem__(self, idx):
        with open(self.points[idx]) as user_file:
            j_string = user_file.read()    
        json_data = json.loads(j_string)
        image = Image.open(self.pil_imgs[idx])
        return image, json_data
    

def collate_fn(batch):
  images, points = list(zip(*batch))
  return images, [torch.tensor([(i['x'], i['y']) for i in points_single]) for points_single in points]