from main import build_model_main
from util.slconfig import SLConfig

import torch
from PIL import Image
from torchvision import transforms as T
import gdown

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

# url = "https://drive.google.com/file/d/1eeAHgu-fzp28PGdIjeLe-pzGPMG2r2G_/view?usp=drive_link"
# weigths = "dino_4scale.pth"
# gdown.download(url, weigths, quiet=False)


output = model(image, target=None)
print("Output:",output)

