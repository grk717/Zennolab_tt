# Zennolab_tt

## Approach 

Used multi-modal zero-shot model [CLIPSeg](https://huggingface.co/docs/transformers/main/en/model_doc/clipseg) and converted mask to bounding boxes. Tried zero-shot detection models from [Autodistill](https://github.com/autodistill/autodistill), all of them showed worse performance globally, but better in some categories.

## Launch
```
git clone https://github.com/grk717/Zennolab_tt.git
cd Zennolab_tt
docker build -t zennolab .
docker run -t --name zenno --gpus all zennolab 
```
Then wait for results in terminal.

## Results

Inferenced with `batch_size=16` on 3050Ti GPU. Inference took 16.13 minutes.

| Category  | Accuracy | Mean distance |
| ------------- | ------------- |------------- |
| squirrels_head  | 0.871  | 0.061 |
| squirrels_tail  | 0.238  | 0.192 |
| the_center_of_the_gemstone  | 0.884  | 0.047 |
| the_center_of_the_koalas_nose  | 0.281  | 0.134 |
| the_center_of_the_owls_head  | 0.952  | 0.036 |
| the_center_of_the_seahorses_head  | 0.841  | 0.065 |
| the_center_of_the_teddy_bear_nose  | 0.459  | 0.118 |
| **Global**  | **0.648** | **0.089** |

