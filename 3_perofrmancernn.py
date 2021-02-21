
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
