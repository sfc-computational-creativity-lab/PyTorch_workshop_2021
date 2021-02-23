#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'
#%%
from glob import glob
from collections import Counter

esc50_list = [f.split("-")[-1].replace(".wav","") for f in glob("./ESC-50/audio/*.wav")]
print(esc50_list)
print(Counter(esc50_list))

# %%
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path

class ESC50(Dataset):
    def __init__(self, path):
        files = Path(path).glob("*.wav")
        self.items = [(f, int(f.name.split("-")[-1].replace(".wav", ""))) for f in files]
        self.length = len(self.items)

    def __getitem__(self, index):
        filename, label = self.items[index]
        audio_tensor, sample_rate = torchaudio.load(filename)
        return audio_tensor, label

    def __len__(self):
        return self.length

# %%
test_esc50 = ESC50("./data/train/")
tensor, label = list(test_esc50)[0]
print(tensor.shape, label) # [1, 220500] tensor: audio signal

# %%
from torch.utils.data import DataLoader

batch_size = 16

train_esc50 = ESC50("./data/train")
val_esc50 = ESC50("./data/val")
test_esc50 = ESC50("./data/test")

train_loader = DataLoader(train_esc50, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_esc50, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_esc50, batch_size = batch_size, shuffle = True)

print(len(train_loader), len(val_loader), len(test_loader)) # len -> number of batches 
# %%

# GPUの有無を確認
import torch
if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda") 
else:
    print("Using CPU")
    device = torch.device("cpu")

# %%

from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# original model in the book
class AudioNet(Module):
    def __init__(self):
        super(AudioNet, self).__init__()

        self.conv1 = nn.Conv1d(1, 128, 80, 4)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, 10, 4)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, 10)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(256, 512, 10)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(4)
        self.avgpool = nn.AvgPool1d(30)
        self.fc1 = nn.Linear(512, 50)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = self.avgpool(x)
        x = x.squeeze(-1)
        x = self.fc1(x)
        return x

audionet = AudioNet()
audionet.to(device) # 昔のバージョンだと　cuda()
summary(audionet, input_size = (1, 220500))
# %% 
# updated version
# class AudioNet(nn.Module):
#     def __init__(self):
#         super(AudioNet, self).__init__()

# #        self.conv1 = nn.Conv1d(1, 128, 80, 4)
#         self.conv1 = nn.Conv1d(100, 128, kernel_size=5, stride=4)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.pool1 = nn.MaxPool1d(4)
#         self.conv2 = nn.Conv1d(128, 128, 3)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.pool2 = nn.MaxPool1d(4)
#         self.conv3 = nn.Conv1d(128, 256, 3)
#         self.bn3 = nn.BatchNorm1d(256)
#         self.pool3 = nn.MaxPool1d(4)
#         self.conv4 = nn.Conv1d(256, 512, 3)
#         self.bn4 = nn.BatchNorm1d(512)
#         self.pool4 = nn.MaxPool1d(4)
#         self.fc1 = nn.Linear(512, 50)

#     def forward(self, x):
#         x = x.unsqueeze(-1).view(-1, 100, 2205)
#         x = self.conv1(x)
#         x = F.relu(self.bn1(x))
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = F.relu(self.bn2(x))
#         x = self.pool2(x)
#         x = self.conv3(x)
#         x = F.relu(self.bn3(x))
#         x = self.pool3(x)
#         x = self.conv4(x)
#         x = F.relu(self.bn4(x))
#         x = self.pool4(x)
#         x = x.squeeze(-1)
#         x = self.fc1(x)
#         return x

# audionet = AudioNet()
# audionet.to(device) # 昔のバージョンだと　cuda()
# summary(audionet, input_size = (1, 220500))

# %%
import torch.optim as optim
optimizer = optim.Adam(audionet.parameters(),lr = 1e-5)
# %%
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

# %%
train(audionet, optimizer, nn.CrossEntropyLoss(), train_loader, val_loader, epochs=40, device=device)


# %%
