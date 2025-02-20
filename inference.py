from main import build_model_main
from util.slconfig import SLConfig

import torch

args = SLConfig.fromfile('config/DINO/DINO_4scale.py')
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = build_model_main(args)

for name, param in model.named_parameters():
    print(name, param.shape)
