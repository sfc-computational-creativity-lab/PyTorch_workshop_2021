
#%%

from __future__ import print_function
import torch



# %%

# 初期化しないと値はバラバラに
x = torch.empty(5, 3)
print(x)

# %%

# 乱数
x = torch.rand(5, 3)
print(x)

#%%

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

#%%

x = torch.tensor([5.5, 3])
print(x)


#%%

x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)        

#%%

print(x.shape)
print(x.size())

#%%
y = torch.rand(5, 3)
print(x + y)

#%%

result = torch.empty(5, 3)
torch.add(x, y, out=result) # output先を指定
print(result)

#%%

print(x)
print(y)
y.add_(x) # add_  値自体を置き換える
print(y)

#%%

x = torch.randn(4, 4)
y = x.view(16)   # viewでreshape    x自体は変化しないことに注意
z = x.view(-1, 8) # -1はよしなにやってねサイン

print(x.shape, y.shape, z.shape)

#%%

# view()して作ったテンソルも同じメモリ空間を指しているので、yに値を入れるとxも変化する
x = torch.randn(4, 4)
print(x)
y = x.view(16)   
y[0:2] = 0

print(y)
print(x)

#%%
x = torch.randn(1)
print(x)
print(x.item()) # 普通のpythonの値を取り出す   値が一つだけの時しか使えない

#%%
a = torch.ones(5)
print(a, type(a))

b = a.numpy() # numpyのarrayに変換
print(b, type(b))

#%%

a.add_(1)
print(a)
print(b) # メモリを共有してるのでaを変えるとbも変化

# %%

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a) # numpy -> torch
np.add(a, 1, out=a)
print(a)
print(b)

#%%

# GPUがある場合
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

# %%
