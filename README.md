# MyTransformer

试着写写，Transformer。锻炼锻炼，手部肌肉。

目前本项目仅支持已经经过 bpe 分词的数据文件训练。同时该模型在给定的iwslt14.de-en数据集上使用post-norm方法并不能够收敛，因此可能还有潜在的bug。

## 已实现部分

+ 少量的 BUG。
+ Data preprocessing
+ DataLoading
    + Memory Pinning [reference](https://pytorch.org/docs/stable/data.html#memory-pinning)
+ Transformer 模型
    + LayerNorm [reference](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html?highlight=layer#torch.nn.LayerNorm)
    + Pre-norm & Post-norm
+ Label smooth cross entropy
    + Ignore padding index [reference](https://discuss.pytorch.org/t/ignore-index-in-the-cross-entropy-loss/25006/9)
+ Training 部分
    + lr_scheduler

+ Inference 部分（写了一个比较 low 的 beam search）

## Known Issue

+ ppl计算有点问题（但问题也不是很大）
+ post norm无法训练收敛

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

+ documentation

