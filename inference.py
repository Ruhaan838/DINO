from main import build_model_main
from util.slconfig import SLConfig

import torch
from PIL import Image
from torchvision import transforms as T
import gdown
import os

#load the sample data
image_path = "./figs/idea.jpg"
image = Image.open(image_path)

transforms = T.Compose([
    T.Resize((800, 1333)),
    T.ToTensor(),
])

image = transforms(image).unsqueeze(0)

#model preparation
args = SLConfig.fromfile('config/DINO/DINO_4scale.py')
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, _, _  = build_model_main(args)

file_id = "1eeAHgu-fzp28PGdIjeLe-pzGPMG2r2G_"
url = f"https://drive.google.com/uc?id={file_id}"

weigths = "dino_4scale.pth"
if not os.path.exists(weigths):
    gdown.download(url, weigths, quiet=False)

model.load_state_dict(torch.load(weigths, map_location='cpu',weights_only=False)['model'])
output = model.forward(image, None)
print("Output:",output)