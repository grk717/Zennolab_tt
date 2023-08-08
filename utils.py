import os
import torch
import json
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

def data_preprocessing(root_dir):
    dataset_dirs = [os.path.join(root_dir, x) for x in os.listdir(root_dir)]
    for dataset in dataset_dirs:
        jsons = sorted([os.path.join(dataset, x) for x in list(filter(lambda x: x.endswith('json'), os.listdir(dataset)))])
        imgs = sorted([os.path.join(dataset, x) for x in list(filter(lambda x: x.endswith('jpg'), os.listdir(dataset)))])
        for f, img_path in zip(jsons, imgs):
            if f.split('.')[0] != img_path.split('.')[0]:
                break
            with open(f) as user_file:
                j_string = user_file.read()    
            try:
                json_data = json.loads(j_string)
                if len(json_data) < 1:
                    raise json.JSONDecodeError('Valid but empty json', j_string, 0)
            except json.JSONDecodeError:
                os.remove(f)
                os.remove(img_path)


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n = masks.shape[0]

    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)
    for index, mask in enumerate(masks):
        # little addition to torchvision.ops.masks_to_boxes
        if not mask.any():
            bounding_boxes[index, 0] = mask.shape[0] // 2
            bounding_boxes[index, 1] = mask.shape[1] // 2
            bounding_boxes[index, 2] = mask.shape[0] // 2
            bounding_boxes[index, 3] = mask.shape[1] // 2
            continue
        
        y, x = torch.where(mask != 0)

        bounding_boxes[index, 0] = torch.min(x)
        bounding_boxes[index, 1] = torch.min(y)
        bounding_boxes[index, 2] = torch.max(x)
        bounding_boxes[index, 3] = torch.max(y)

    return bounding_boxes

def init_model(model_str: str ="CIDAS/clipseg-rd64-refined", device: str ='cuda'):
    processor = CLIPSegProcessor.from_pretrained(model_str)
    model = CLIPSegForImageSegmentation.from_pretrained(model_str).to(device)
    return processor, model