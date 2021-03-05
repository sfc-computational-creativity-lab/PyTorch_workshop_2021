
#%%

# なんちゃってPerformance RNN

# MIDIファイルの扱い方 - ここでは pretty_midiを使う

import pretty_midi
import math

# %%

path = './data/midi/bach/train/aof/dou1.mid'

pm = pretty_midi.PrettyMIDI(path)

print(pm)

# %%

# PerformanceRNN用の定数

RANGE_NOTE_ON = 128     # Note On: 0-127
RANGE_NOTE_OFF = 128    # Note Off: 0-127 
RANGE_VEL = 32          # Velocity 0-127を 4で割って 0-31 に丸める
RANGE_TIME_SHIFT = 100  # 10ms - 1.0 sec を 100段階で

TOTAL_EVENTS = RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_VEL + RANGE_TIME_SHIFT

MAX_TIME_SHIFT = 1.0   

START_IDX = {
    'note_on': 0,
    'note_off': RANGE_NOTE_ON,
    'time_shift': RANGE_NOTE_ON + RANGE_NOTE_OFF,
    'velocity': RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT
}

print("TOTAL_EVENTS", TOTAL_EVENTS)
print(START_IDX)
# %%
# Performance RNN用のデータの作成

#%%

# dictionary をイベントＩＤの配列に
def get_event_id_from_noteon_pitch(p):   
    event_id = math.floor(p/127. * RANGE_NOTE_ON)
    return event_id

def get_event_id_from_noteoff_pitch(p):   
    event_id = math.floor(p/127. * RANGE_NOTE_ON)
    return event_id + START_IDX['note_off']

def get_noteon_pitch_from_event_id(event_id):
    return event_id

def get_noteoff_pitch_from_event_id(event_id):
    return event_id - START_IDX['note_off']

def get_event_id_from_timeshift(t):
    t = max(0, min(MAX_TIME_SHIFT, t))
    event_id = math.floor(t/MAX_TIME_SHIFT * RANGE_TIME_SHIFT)
    return event_id + START_IDX['time_shift']

def get_timeshift_from_event_id(event_id):
    t = event_id - START_IDX['time_shift']
    t = t/RANGE_TIME_SHIFT * MAX_TIME_SHIFT
    return t

def get_event_id_from_velocity(v):
    event_id = math.floor(v/127. * RANGE_VEL)
    return event_id + START_IDX['velocity']

def get_velocity_from_event_id(event_id):
    v = event_id - START_IDX['velocity']
    v = math.floor(v/RANGE_VEL * 127.)
    return v

def get_event_type_and_value(event_id):
    if event_id < START_IDX['note_off']:
        return 'note_on', get_noteon_pitch_from_event_id(event_id)
    if event_id < START_IDX['time_shift']:
        return 'note_off', get_noteoff_pitch_from_event_id(event_id)
    if event_id < START_IDX['velocity']:
        return 'time_shift', get_timeshift_from_event_id(event_id)
    else:
        return 'velocity', get_velocity_from_event_id(event_id)

#%%

def event_array_to_note_array(events, bpm=120):
    cur_time = 0.0
    cur_vel = 0
    cur_pitch = 0
    data = []

    for index, event_id in enumerate(events):
        event_type, value = get_event_type_and_value(event_id)

        if event_type is 'note_on' or event_type is 'note_off':
            cur_pitch = value
            cur_event = event_type
        elif event_type is 'velocity':
            cur_vel = value 
        elif event_type is 'time_shift':
            cur_time += value

        if event_type is 'note_on' and cur_pitch > 0:
            if cur_vel == 0: # フォーマットがおかしい場合
                cur_vel = 64 # とりあえず中間の値
            note = { 'time': cur_time, 'type': 'note_on', 'pitch': cur_pitch, 'velocity': cur_vel  }
            data.append(note)
            cur_vel = 0
            cur_pitch = 0

        elif event_type is 'note_off' and cur_pitch > 0:
            note = { 'time': cur_time, 'type': 'note_off', 'pitch': cur_pitch, 'velocity': 0  }
            data.append(note)
            cur_pitch = 0
            cur_vel = 0
    return data

def event_array_to_midi(events, bpm=120):
    # まずはnoteon noteoffの列に
    data = event_array_to_note_array(events)

    # midi fileのオブジェクト
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano)

    # noteonの状態のピッチを保持
    cur_noteons = {}

    cur_time = 0.0
    for index, note in enumerate(data):
        pitch = note['pitch']
        velocity = note['velocity']
        event_type = note['type']
        evnt_time = note['time']

        if event_type is 'note_on':
            # すでに同じピッチがオンになってたら終わらせる
            if pitch in cur_noteons.keys():
                prev_noteon = cur_noteons[pitch]
                midi_note = pretty_midi.Note(velocity=prev_noteon['velocity'], pitch=pitch, 
                                        start=prev_noteon['time'], end=evnt_time)
                piano.notes.append(midi_note)
                del cur_noteons[pitch]
            # ノートオン中のピッチに追加
            cur_noteons[pitch] = note

        if event_type is 'note_off':
            if pitch in cur_noteons.keys():
                noteon = cur_noteons[pitch]
                midi_note = pretty_midi.Note(velocity=noteon['velocity'], pitch=pitch, 
                                        start=noteon['time'], end=evnt_time)
                piano.notes.append(midi_note)
                del cur_noteons[pitch]
            else:
                print("error: cannot find a note to note off")
    pm.instruments.append(piano)
    return pm

#%%

def note_data_from_midi_file(filepath):
    # MIDI Fileをロード
    pm = pretty_midi.PrettyMIDI(path)

    # 結果を格納する配列
    data = []

    for inst in pm.instruments:
        if inst.is_drum: # ドラムはトラックは無視
            continue

        for note in inst.notes: 
            # 各ノートは Note onとNote offで分けて考えられる
            noteon = { 'time': note.start, 'type': 'note_on', 'pitch': note.pitch, 'velocity': note.velocity  }
            noteoff = { 'time': note.end, 'type': 'note_off', 'pitch': note.pitch, 'velocity': 0  }
            data.append(noteon)
            data.append(noteoff)

    # noteをスタートのタイミングでソートする
    data = sorted(data, key=lambda event: event['time'])
    return data

def event_list_from_midi_file(filepath):

    # ソート済みのmidiのノート情報
    data = note_data_from_midi_file(filepath)

    # イベントリストに変換
    events = []
    prev_time = 0.0
    for i, d in enumerate(data):
        time = d['time']
        pitch = d['pitch']
        event_type = d['type'] 

        if event_type is 'note_on':
            velocity = d['velocity']

            # timeshift
            if i == 0:
                prev_time = time
            
            time_shift = time - prev_time
            e = get_event_id_from_timeshift(time_shift)
            events.append(e)

            # noteon velocity
            e = get_event_id_from_velocity(velocity)
            events.append(e)

            # noteon pitch
            e = get_event_id_from_noteon_pitch(pitch)
            events.append(e)

        elif event_type is 'note_off':
            # timeshift
            time_shift = time - prev_time
            e = get_event_id_from_timeshift(time_shift)
            events.append(e)

            # noteoff pitch
            e = get_event_id_from_noteoff_pitch(pitch)
            events.append(e)

        # store timing
        prev_time = time
    return events


#%%

# ファイルをいったんeventのリストに変換　ＭＩＤＩにまた戻してちゃんと同じになっているか確認
events_ = event_list_from_midi_file(path)
pm2 = event_array_to_midi(events_)

pm2.write("./tmp/temp.mid")

#%%

import torch
from torch.utils.data import Dataset
from pathlib import Path
import random

class MIDIData(Dataset):
    def __init__(self, path, prime_length = 24, total_num = 1000):
        self.files = Path(path).glob("*/*.mid")

        # 各トラックごとにイベントの配列を取り出した配列を作る
        events = []
        for filepath in self.files:
            events.append(event_list_from_midi_file(str(filepath)))
        print("total # of sequences", len(events))

        # ランダムに prime_lengthの長さのピッチ列を作り、次のノートを格納する
        self.primes = []
        self.nexts = []
        for _ in range(total_num):
            ps = random.choice(events)

            if (len(ps) < prime_length + 1):
                continue # 短すぎるシーケンスは無視

            start_index = random.randint(0, len(ps) - prime_length -1 -1) # randintの範囲に注意 
            end_index = start_index + prime_length
            next_index = end_index + 1 # 次のイベントのインデックス

            prime = ps[start_index:end_index] # input
            next_pitch = ps[next_index]       # output

            self.primes.append(prime) 
            self.nexts.append(next_pitch)  

        self.length = len(self.primes)

    def __getitem__(self, index):
        # PyTorchのテンソルにしてreturn
        return torch.tensor(self.primes[index]),  torch.tensor(self.nexts[index])

    def __len__(self):
        return self.length



# %%

train_data = MIDIData('./data/midi/bach/train/', total_num=100000)
val_data = MIDIData('./data/midi/bach/val/', total_num=8000) 

batch_size = 32
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

# %%

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

EMBEDDING_DIM =128
HIDDEN_DIM = 256


class PerformanceRNN(nn.Module):
    
    def __init__(self):
        super(PerformanceRNN, self).__init__()

        self.embeds = nn.Embedding(TOTAL_EVENTS, EMBEDDING_DIM)
        self.lstm   = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True)
        self.lstm2   = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, batch_first=True)
        self.fc     = nn.Linear(HIDDEN_DIM, TOTAL_EVENTS) 

    def forward(self, x):
        emb = self.embeds(x)
        w, (_, _) = self.lstm(emb) # output, (h, c)
        _, (h, _) = self.lstm2(w) # output, (h, c)
        h = h.squeeze(dim=0)        # バッチのdimentionはsqueezeしないように注意
        y = self.fc(h)
        return y

prnn_model = PerformanceRNN()

#%%

# Optimizer
optimizer = optim.Adam(prnn_model.parameters(), lr=0.001)

# GPUの有無を確認
if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda") 
else:
    print("Using CPU")
    device = torch.device("cpu")
prnn_model.to(device) # 昔のバージョンだと　cuda()
print(prnn_model)



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
train(prnn_model, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, 
    val_data_loader, epochs=40, device=device)

print("finished training")
#%%

train_data[:4]

# %%

# 学習済みモデルのテスト
import numpy as np

prnn_model.eval()

temperature = 1.5

for i in range(10):
    seq = random.choice(val_data.primes)
    seq = torch.tensor(seq)
    seq = seq.to(device)

    for _ in range(1024):
        seq_input = torch.unsqueeze(seq, 0) # バッチを作る
#        seq_input = seq_input[:,-32:]
        output = prnn_model(seq_input)
        prediction = F.softmax(output / temperature, dim=1)
        next_note = torch.multinomial(prediction, 1) #  next_note = prediction.argmax()
        seq = torch.cat((seq, torch.squeeze(next_note, dim=0)), 0)

    print(seq)
    pm = event_array_to_midi(seq.tolist())
    pm.write("./tmp/prnn_output_bach_%d.mid" % i)

# %%


# %%

