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
    transforms.Resize((64, 64)),  # 64 pixel x 64 pixel
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

batch_size = 32
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

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
        x = self.fc3(x) # cross entropyロスを使うときは内部で
                        # softmax()をかけているのでここではそのまま返す
        return x

simplenet = SimpleNet()

# Optimizer
optimizer = optim.Adam(simplenet.parameters(), lr=0.001)

# GPUの有無を確認
if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda") 
else:
    print("Using CPU")
    device = torch.device("cpu")
simplenet.to(device) # 昔のバージョンだと　cuda()
print(simplenet)

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
train(simplenet, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, 
    val_data_loader, epochs=10, device=device)
print("finished training")

# save
import os
os.makedirs("./tmp", exist_ok=True)
torch.save(simplenet, "./tmp/simplenet_cat_fish")  # まるごとセーブ

# 
# torch.save(simplenet.state_dict(), "./tmp/simplenet_cat_fish_dict") # layerの名前と重みをdict形式で保存

