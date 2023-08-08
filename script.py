import torch 
import numpy as np
import os
from utils import data_preprocessing, init_model
from evaluator import FolderEvaluator
import pprint
from dataclasses import asdict

np.random.seed(42)
torch.manual_seed(42)
os.environ["PYTHONHASHSEED"] = "42"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_preprocessing(r"./tasks/")

FOLDERS = [r'./tasks/squirrels_head', 
            r'./tasks/squirrels_tail',
            r'./tasks/the_center_of_the_gemstone',
            r'./tasks/the_center_of_the_koalas_nose', 
            r'./tasks/the_center_of_the_owls_head', 
            r'./tasks/the_center_of_the_seahorses_head', 
            r'./tasks/the_center_of_the_teddy_bear_nose']
PROMPTS = [['squirrels head'],
            ['squirrels tail'],
            ['gemstone'],
            ['koalas head'],
            ['owls head'],
            ['seahorses head'],
            ['teddy bear head']]      

processor, model = init_model("CIDAS/clipseg-rd64-refined", device)

fe = FolderEvaluator(processor, model, device)

list(map(fe.evaluate, FOLDERS, PROMPTS))
fe.results.global_info.accuracy = fe.results.global_info.correct_num / fe.results.global_info.total_num
pprint.pprint(asdict(fe.results))