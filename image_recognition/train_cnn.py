#%%
import torchvision
from torchvision import transforms

import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from glob import glob 

train_data_path = "./data/train"

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 64 pixel x 64 pixel
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def check_image(path):
    try:
        im = Image.open(path)
        return True
    except:
        return False

train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=img_transforms,is_valid_file=check_image)
print(len(train_data))

val_data_path = "./data/val/"
val_data = torchvision.datasets.ImageFolder(root=val_data_path, transform=img_transforms, is_valid_file=check_image)

test_data_path = "./data/test/"
test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=img_transforms, is_valid_file=check_image)

val_data[0]

#%%

batch_size = 32
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

#%%


class CNNNet(nn.Module):
    
    def __init__(self, num_classes = 2):
        super(CNNNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=4, padding=2), # input channel, output channel(# of filters)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6)) # 入力のサイズに関係なく(6, 6)で出力してくれる
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
#        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

cnnnet = CNNNet()

#%%

from torchinfo import summary

summary(cnnnet,(1,3,224,224))


#%%

# Optimizer
optimizer = optim.Adam(cnnnet.parameters(), lr=0.001)

# GPUの有無を確認
if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda") 
else:
    print("Using CPU")
    device = torch.device("cpu")
cnnnet.to(device) # 昔のバージョンだと　cuda()
print(cnnnet)

epochs = 100

def train(model, optimizer,loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0

        model.train() # 学習モードにセット　DropoutLayerなどが有効に

        for batch in train_loader:
            optimizer.zero_grad() # 一旦リセット

            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs) 

            loss = loss_fn(output, labels)
            loss.backward() # back propagation - gradientの計算

            optimizer.step()
            training_loss += loss.data.item()
        training_loss /= len(train_loader) # average

        model.eval()# 学習モードをオフ　DropoutLayerなどが無効に モデルのパラメータはアップデートされない
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(inputs)

            loss = loss_fn(output, labels)
            valid_loss += loss.data.item()

            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], labels).view(-1)

            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader)

        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, Accuracy = {:.2f}'
            .format(epoch, training_loss, valid_loss, num_correct/num_examples))

# training
train(cnnnet, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, 
    val_data_loader, epochs=30, device=device)
print("finished training")

# save
import os
os.makedirs("./tmp", exist_ok=True)
torch.save(cnnnet, "./tmp/cnnnet_cat_fish")  # まるごとセーブ

# 
# torch.save(simplenet.state_dict(), "./tmp/simplenet_cat_fish_dict") # layerの名前と重みをdict形式で保存



# %%


# %%
