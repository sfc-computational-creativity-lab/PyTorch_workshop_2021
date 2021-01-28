#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"]='2,1'
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
test_esc50 = ESC50("./data/train/")
tensor, label = list(test_esc50)[0]
print(tensor.shape, label) 

#%%

waveform, sr = torchaudio.load("./data/train/1-37226-A-29.wav")
specgram = torchaudio.transforms.Spectrogram()(waveform)
specgram = test_esc50.transforms(waveform)
print(specgram)
#print(specgram)
print("Shape of spectrogram: {}".format(specgram.size()))

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

# %%
from torchsummary import summary 
from torchvision import models

spec_resnet = models.resnet50(pretrained=True)


# %%
from torch import nn as nn
for param in spec_resnet.parameters():
    param.requires_grad = False

print(spec_resnet)

spec_resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
spec_resnet.fc = nn.Sequential(nn.Linear(spec_resnet.fc.in_features, 500), 
    nn.ReLU(),
    nn.Dropout(), nn.Linear(500, 50)
)
print(spec_resnet)
spec_resnet.to(device)


# %%
import torch.optim as optim
optimizer = optim.Adam([ {'params': spec_resnet.layer1.parameters(), 'lr': 1e-6},
                        {'params': spec_resnet.layer2.parameters(), 'lr': 1e-6},
                        {'params': spec_resnet.layer3.parameters(), 'lr': 1e-6},
                        {'params': spec_resnet.layer4.parameters(), 'lr': 1e-6},
                        {'params': spec_resnet.fc.parameters(), 'lr': 1e-2}], lr=1e-7)
# %%
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

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
train(spec_resnet, optimizer, nn.CrossEntropyLoss(), train_loader, val_loader, epochs=40, device=device)


# %%
