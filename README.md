# MyTransformer

试着写写，Transformer。锻炼锻炼，手部肌肉。

## 已实现部分

+ 大量的 BUG。（Valid ppl 13+）
+ Data preprocessing
+ DataLoading
+ Transformer 模型
    + LayerNorm [reference](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html?highlight=layer#torch.nn.LayerNorm)
+ Label smooth cross entropy
    + Ignore padding index [reference](https://discuss.pytorch.org/t/ignore-index-in-the-cross-entropy-loss/25006/9)

+ Training 部分
+ Inference 部分（写了一个比较 low 的 beam search）

## Usage

预处理数据。数据在[这里](https://git.io/JPK9N)下载。

```
python preprocess.py --data-path ~/path/to/iwslt14.tokenized.de-en --target-lang en --source-lang de
```

训练

```
python train.py --data data-bin/iwslt14.tokenized.de-en --src-lang de --tgt-lang en --epoch 3
```

推断

```
python inference.py --data data-bin/iwslt14.tokenized.de-en --src-lang de --tgt-lang en --model-path path/to/checkpoint.pt
```

## To do

+ 更好的logging

+ debug
