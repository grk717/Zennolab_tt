from tqdm import tqdm
from data_utils import EvaluationResults, FolderResults, InferDataset, collate_fn
import torch
import time
from typing import List
from torchvision.ops import box_convert 
from utils import masks_to_boxes

class FolderEvaluator:
    def __init__(self, processor, model, device, batch_size=16, mask_threshold=0.3) -> None:
        self.img_size = processor.image_processor.size['height']
        self.processor = processor
        self.batch_size = batch_size
        self.model = model
        self.device = device
        self.mask_threshold = mask_threshold
        self.results = EvaluationResults(global_info=FolderResults())


    def compare_points(self, gt_centers, predicted_centers, folder_result):
        for i in range(len(gt_centers)):
            dists = torch.cdist(gt_centers[i], predicted_centers[i][None, ...], p=2)
            folder_result.dist_sum += float(dists.min())
            folder_result.correct_num += int((dists < 0.1).sum())
            folder_result.total_num += len(gt_centers[i])

    def evaluate(self, dir_path: str, prompts: List[str]):
        start = time.time()
        
        folder_results = FolderResults(prompt=prompts)
        dataset = InferDataset(dir_path=dir_path)
        dataloader = torch.utils.data.DataLoader(dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=0,
                                           pin_memory=True, collate_fn=collate_fn)
        
        for images, gt_centers in tqdm(dataloader):
            inputs = self.processor(text=prompts*len(images), images=images, padding="max_length", return_tensors="pt")
            inputs['pixel_values'] = inputs['pixel_values'].to(self.device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self.device)
            inputs['input_ids'] = inputs['input_ids'].to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            preds = torch.sigmoid(outputs.logits).cpu()
            if preds.dim() < 3:
                preds = preds.unsqueeze(0)
            preds_bool = preds >= self.mask_threshold
            boxes = masks_to_boxes(preds_bool)
            predicted_centers = box_convert(boxes, 'xyxy', 'cxcywh')[:,:2] /  torch.tensor([self.img_size, self.img_size])
            self.compare_points(gt_centers, predicted_centers, folder_results)

        folder_results.elapsed_time = time.time() - start
        folder_results.mean_dist = folder_results.dist_sum / folder_results.total_num
        folder_results.accuracy = folder_results.correct_num / folder_results.total_num 
        self.results.global_info.elapsed_time += folder_results.elapsed_time
        self.results.global_info.total_num += folder_results.total_num
        self.results.global_info.correct_num += folder_results.correct_num
        self.results.global_info.dist_sum += folder_results.dist_sum
        self.results.dirs[dir_path] = folder_results