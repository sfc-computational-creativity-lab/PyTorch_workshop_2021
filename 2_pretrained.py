# %%
# torch visionに用意されているmodelを確認
import torch
from torchvision import models
dir(models)
# %%
# resnetのpre-trainedモデルをロード

resnet = models.resnet101(pretrained=True)

print(resnet)

# %%
# keras風のsummary()を使うのであれば

!pip install torchinfo

# %%
from torchinfo import summary

summary(resnet)

summary(resnet, depth=1)
# summary() 自体がprintした上に、jupyterは最後のアウトプットをprintするので二重になる

# %%

summary(resnet, (1, 3, 224, 224), depth=1)

summary(resnet, (1, 3, 224, 224), depth=2)

summary(resnet, (1, 3, 224, 224), depth=3)

# %%

# resnetに画像を入力するための変換用の関数も用意されている
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(256),  
    transforms.CenterCrop(224),  #入力が224 x 224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # 学習した画像データセットの平均、標準偏差に合わせる
        std=[0.229, 0.224, 0.225]
    )
])
print(preprocess)
#%%

from PIL import Image
from IPython.display import display

img = Image.open("./data/images/black.jpg")

display(img)  
#img.show() # ポップアップで表示することも可能
print(img)

# %%

# tensorに変換
img_t = preprocess(img)

print(img_t)

print(img_t.shape)
# %%

# 画像認識のモデルへの入力は複数の画像をまとめた
# バッチ(batch)で入力することになっている

img_batch = torch.unsqueeze(img_t, 0) # テンソルの最初に一次元追加！
print(img_batch.shape) 

img_batch = img_t.unsqueeze(0)  # こういう書き方もの
print(img_batch.shape) 

# numpyにあるような expand_dimsは torchにはない

# %%

# resnetにかける

# ポイントはモデルを関数のように使えるということ

# inferenceの前に evaluationモードに変更
# trainモードのままだとdropboxレイヤーなどが生きたままになる
resnet.eval()  

# 実際にresnetで解析!
out = resnet(img_batch)
print(out.shape)
# %%

# トップ5をみると...
top5 = out.topk(5)


print(top5.indices)

# numpyに変換するには 
print(top5.indices.numpy())
print(top5.indices.numpy().shape) # バッチサイズ 1の結果であることに注意

# %%

# ImageNetのクラスの名前を読み込む
with open('./data/others/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

print(labels[:5])

# %% 

for index, class_index in enumerate(top5.indices[0]):
    print(f"{index}: {labels[int(class_index)]}")

# %%

value, index = torch.max(out, 1)  # [1, 1000]の1000の列の中のmaxを探したい!
print(value, index)

# %%
# 実際に outの中身を見てみると... マイナスの値も
# softmax() をかける前！
print(out)

# %%

# softmaxを通す
out_softmax = torch.nn.functional.softmax(out, dim=1)
print(out_softmax)
print(out_softmax.shape)

value, index = torch.max(out_softmax, 1)
print(f"{labels[index]} confidence: {value.item()}") # item()で普通のpythonの数に戻す




# %%
