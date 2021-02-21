
#%%

# MIDIファイルの扱い方 - ここでは pretty_midiを使う

!pip install pretty_midi

import pretty_midi

# %%

path = './data/midi/Nocturnes/chno0901.mid'

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

# PerformanceRNN用のデータ

RANGE_NOTE_ON = 128     # Note On: 0-127
RANGE_NOTE_OFF = 128    # Note Off: 0-127 
RANGE_VEL = 32          # Velocity 0-127を 4で割って 0-31 に丸める
RANGE_TIME_SHIFT = 100  # 10ms - 1.0 sec を 100段階で

START_IDX = {
    'note_on': 0,
    'note_off': RANGE_NOTE_ON,
    'time_shift': RANGE_NOTE_ON + RANGE_NOTE_OFF,
    'velocity': RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT
}



# %%
# Performance RNN用のデータの作成

for inst in pm.instruments:
    if inst.is_drum: # ドラムはトラックは無視
        continue
    
    # 結果を格納する配列
    data = []

    for note in inst.notes: 
        # 各ノートは Note onとNote offで分けて考えられる
        noteon = { 'time': note.start, 'type': 'note_on', 'pitch': note.pitch, 'velocity': note.velocity  }
        noteoff = { 'time': note.end, 'type': 'note_off', 'pitch': note.pitch, 'velocity': 0  }
        data.append(noteon)
        data.append(noteoff)

    print(data[:10])

        

# %%


for inst in pm.instruments:
    if inst.is_drum: # ドラムはトラックは無視
        continue
    
    # 結果を格納する配列
    data = []

    for note in inst.notes: 
        # 各ノートは Note onとNote offで分けて考えられる
        noteon = { 'time': note.start, 'type': 'note_on', 'pitch': note.pitch, 'velocity': note.velocity  }
        noteoff = { 'time': note.end, 'type': 'note_off', 'pitch': note.pitch, 'velocity': 0  }
        data.append(noteon)
        data.append(noteoff)

    