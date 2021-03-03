#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"]='2'
#%%
from glob import glob
from collections import Counter

esc50_list = [f.split("-")[-1].replace(".wav","") for f in glob("./ESC-50/audio/*.wav")]
print(Counter(esc50_list))

# %%
import torchaudio
import torchvision
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np 
import functools

SAMPLE_RATE = 16000

class ESC50(Dataset):
    def __init__(self, path):
        files = Path(path).glob("*.wav")
        self.items = [(f, int(f.name.split("-")[-1].replace(".wav", ""))) for f in files]
        self.length = len(self.items)
        self.transforms = torchvision.transforms.Compose([torchaudio.transforms.Spectrogram()])#,torchaudio.transforms.AmplitudeToDB()])

    @functools.lru_cache(maxsize=500) # least recently used cache - 一番古く使われたものから消していく
    def __getitem__(self, index):
        filename, label = self.items[index]
        audio_tensor, sample_rate = torchaudio.load(filename)
        audio_tensor = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)(audio_tensor)
        return self.transforms(audio_tensor), label

    def __len__(self):
        return self.length

# %%
train_esc50 = ESC50("./data/train/")
tensor, label = list(train_esc50)[0]
print(tensor.shape, label) 

#%%

waveform, sr = torchaudio.load("./data/train/1-100210-B-36.wav")

print(waveform.shape)

#waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
  

specgram = torchaudio.transforms.Spectrogram()(waveform)
print(specgram)

      
#print(specgram)
print("Shape of spectrogram: {}".format(specgram.size()))

#%%

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(specgram.log2()[0,:,:].numpy(), cmap='gray')

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
#%%


class ACNNNet(nn.Module):
    
    def __init__(self, num_classes = 2):
        super(ACNNNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=4, padding=2), # input channel, output channel(# of filters)
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

acnnnet = ACNNNet(num_classes=50)
acnnnet.to(device)

from torchinfo import summary

print(list(specgram.shape))

summary(acnnnet, [1,1,201,401])
# %%
from torchinfo import summary 
from torchvision import models

#spec_resnet = models.resnet50(pretrained=True)
spec_vgg16 = models.vgg16(pretrained=True)

# %%
from torch import nn as nn
for param in spec_vgg16.parameters():
    param.requires_grad = False

print(spec_vgg16)

# %%

spec_vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
spec_vgg16.classifier[6] = nn.Linear(spec_vgg16.classifier[6].in_features, 50)

for param in spec_vgg16.features[0].parameters():
    param.requires_grad = True

for param in spec_vgg16.classifier.parameters():
    param.requires_grad = True

print(spec_vgg16)
spec_vgg16.to(device)


# %%

from torchinfo import summary

print(list(specgram.shape))

summary(spec_vgg16, [1,1,201,401])

# %%
import torch.optim as optim
optimizer = optim.Adam(acnnnet.parameters(), lr=1e-3)
# %%
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F

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
train(acnnnet, optimizer, nn.CrossEntropyLoss(), train_loader, val_loader, epochs=40, device=device)


# %%
