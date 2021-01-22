import torch
from torchvision import transforms

import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from glob import glob 
import random

# クラスの定義を書いておかないとモデルをロードするところでコケる
class SimpleNet(nn.Module):
    
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(64 * 64 * 3, 84) # 64 pixel * 64 * RGB 
        self.fc2 = nn.Linear(84, 50)
        self.fc3 = nn.Linear(50, 2)

    def forward(self, x):
        x = x.view(-1, 12288) #画像を１次元のベクトルに変換
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 学習済みのモデルをロード
#simplenet = torch.load("/tmp/simplenet")
simplenet = torch.load("./tmp/simplenet_cat_fish")

img_transforms = transforms.Compose([
    transforms.Resize((64, 64)),  # 64 pixel x 64 pixel
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# GPUの有無を確認
if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda") 
else:
    print("Using CPU")
    device = torch.device("cpu")

labels = ['cat','fish']

filepaths = glob("./data/val/*/*.jpg")

# ランダムにターゲットを選ぶ
for _ in range(10):
    try:
        filepath = random.choice(filepaths)
        print(filepath)

        img = Image.open(filepath) 
        img = img_transforms(img).to(device)
        img = torch.unsqueeze(img, 0)   # 画像が一個だけのバッチをつくる

        simplenet.eval()
        prediction = F.softmax(simplenet(img), dim=1)
        prediction = prediction.argmax()
        print(labels[prediction]) 
    except:
        print("Error:", filepath)