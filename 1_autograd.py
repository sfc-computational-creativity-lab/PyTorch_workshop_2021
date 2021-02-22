#!/usr/bin/env python
# coding: utf-8



#%%

import torch

x = torch.tensor(2., requires_grad= True)

#%%

y = x + 2
z = y * y

print(z)

#%%
z.backward()
print(x.grad)

#%%

x =torch.ones(2, 2, requires_grad=True) # 勾配の計算が必要なときはフラグを立てる
print(x)


#%%

y = x + 2
print(y)


#%%

print(y.grad_fn)

#%%



#%%

z = y * y * 3

z.backward()
print(x.grad)


#%%

out = z.mean()

print(z, out)


# %%

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad) # defaultはFalse
a.requires_grad_(True) # requires_grad_()でその変数のrequires_grad設定を変更できる
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)


#%%

out.backward() # backpropagataion


# scalarの場合は直接x.gradでアクセスできる

#%%

print(z.grad)
print(y.grad)
print(x.grad)


#%%

x = torch.randn(3, requires_grad=True)

y = x
print(y)
i = 0
while y.data.norm() < 1000:
    y = y * 2
    i += 1

print(y)
print(i)


# %%

v = torch.tensor([1, .1, .1], dtype=torch.float)
y.backward(v)

print(x.grad)


# %%

2 ** 9


# %%



