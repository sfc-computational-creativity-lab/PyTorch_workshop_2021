# %%
# MIDIファイルの扱い方 - ここでは pretty_midiを使う

!pip install pretty_midi

import pretty_midi

# %%

path = './data/midi/Nocturnes/train/chno0901.mid'

pm = pretty_midi.PrettyMIDI(path)

print(pm)
# %%
# トラックを確認
print(pm.instruments)

# %%

# MIDIのNote-on Note-offの確認
for inst in pm.instruments:
    if inst.is_drum: # ドラムはトラックは無視
        continue

    for note in inst.notes[:5]: # 最初の5ノート
        print(note)

# %%

# MIDIのNote-on Note-offの確認

MIN_NOTE = 36
MAX_NOTE = 92
PITCH_NUM = 92 - 36 + 1

def get_pitch_array(filepath):
    print(filepath)
    pm = pretty_midi.PrettyMIDI(filepath)
    
    # トラックごとにピッチの配列を作る
    results = []
    for inst in pm.instruments:
        if inst.is_drum: # ドラムはトラックは無視
            continue

        # noteをスタートのタイミングでソートする
        notes = sorted(inst.notes, key=lambda note: note.start)

        # ピッチの配列
        pitches = [ min(PITCH_NUM - 1, max(0, note.pitch - MIN_NOTE)) for note in inst.notes ]
        
        results.append(pitches)
    
    return results

def pitch_array_to_midi(pitches, bpm=120):
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano)

    quarter_note_length = 60/bpm

    for index, pitch in enumerate(pitches):
        start = index * quarter_note_length
        end = start + quarter_note_length
        p   = pitch + MIN_NOTE
        note = pretty_midi.Note(velocity=100, pitch= p, start=start, end=end)
        piano.notes.append(note)
    pm.instruments.append(piano)
    return pm


#%%

from torch.utils.data import Dataset
from pathlib import Path
import random

class MIDIData(Dataset):
    def __init__(self, path, prime_length = 8, total_num = 1000):
        self.files = Path(path).glob("*.mid")

        # 各トラックごとにピッチの配列だけを取り出した配列を作る
        pitches = []
        for filepath in self.files:
            pitches.extend(get_pitch_array(str(filepath)))

        # ランダムに prime_lengthの長さのピッチ列を作り、次のノートを格納する
        self.primes = []
        self.nexts = []
        for _ in range(total_num):
            ps = random.choice(pitches)

            if (len(ps) < prime_length + 1):
                continue # 短すぎるシーケンスは無視

            start_index = random.randint(0, len(ps) - prime_length -1 -1) # randintの範囲に注意 
            end_index = start_index + prime_length
            next_index = end_index + 1 # 次のピッチのインデックス

            prime = ps[start_index:end_index]
            next_pitch = ps[next_index]

            self.primes.append(prime) # PyTorchのテンソルにして格納
            self.nexts.append(next_pitch)  

        self.length = len(self.primes)

    def __getitem__(self, index):
        return torch.tensor(self.primes[index]),  torch.tensor(self.nexts[index])

    def __len__(self):
        return self.length
# %%
train_data = MIDIData('./data/midi/chopin/train/', total_num=50000)
val_data = MIDIData('./data/midi/chopin/val/')

print(train_data.primes[:3])
print(train_data.nexts[:3])


batch_size = 32
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

# %%

import torch.nn as nn

EMBEDDING_DIM = 16

embed = nn.Embedding(PITCH_NUM, EMBEDDING_DIM)

x = torch.tensor(train_data.primes[0])
x = torch.unsqueeze(x, 1) # RNNの入力は デフォルトで(seq_length, batch, input dimension)のフォーマット
print(x.shape)

emb = embed(x)
print(emb)
print(emb.shape)

# %% 

lstm = nn.RNN(EMBEDDING_DIM, PITCH_NUM)
output, h = lstm(emb) 

print(output)
print(h.shape)

# %%

# ピッチのシーケンスから次のピッチを予測するモデル

class PitcnNet(nn.Module):
    
    def __init__(self):
        super(PitcnNet, self).__init__()
        self.embeds = nn.Embedding(PITCH_NUM, EMBEDDING_DIM)
        self.rnn   = nn.LSTM(EMBEDDING_DIM, PITCH_NUM, batch_first=True)

    def forward(self, x):
        emb = self.embeds(x)
        _, (hidden, _) = self.rnn(emb)
        y = hidden.squeeze(0)
        # print(x.shape, hidden.shape, y.shape)
        return y

pitchnet = PitcnNet()
# %%
# Optimizer
optimizer = optim.Adam(pitchnet.parameters(), lr=0.001)

# GPUの有無を確認
if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda") 
else:
    print("Using CPU")
    device = torch.device("cpu")
pitchnet.to(device) # 昔のバージョンだと　cuda()
print(pitchnet)


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

# training
train(pitchnet, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, 
    val_data_loader, epochs=100, device=device)

print("finished training")

#%%

# save
import os
os.makedirs("./tmp", exist_ok=True)
torch.save(pitchnet, "./tmp/pitchnet_model.pth")  # まるごとセーブ

#%%


pitchnet.eval()

seq = random.choice(val_data.primes)
seq = torch.tensor(seq)

for _ in range(20):
    seq_input = torch.unsqueeze(seq, 0) # バッチを作る
    prediction = F.softmax(pitchnet(seq_input), dim=1)
    next_note = prediction.argmax()
    seq = torch.cat((seq, torch.unsqueeze(next_note,0)), 0)
    print(seq)

# %%

pm = pitch_array_to_midi(seq.tolist())

pm.write("./tmp/piano-output.mid")
# %%
