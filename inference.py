from main import build_model_main
from util.slconfig import SLConfig

args = SLConfig.fromfile('config/DINO/DINO_4scale.py')
model = build_model_main(args)

for name, param in model.named_parameters():
    print(name, param.shape)
