# MyTransformer

试着写写，Transformer。锻炼锻炼，手部肌肉。

目前本项目仅支持已经经过 bpe 分词的数据文件训练。同时相比标准 Transformer 模型，BLEU 差的有点多（标准 Transformer BLEU 34.44），因此可能还有潜在的 bug。

| Model         | Sacrebleu | 1-gram BLEU | 2-gram BLEU | 3-gram BLEU | 4-gram BLEU | BLEU-4 |
| ------------- | --------- | ----------- | ----------- | ----------- | ----------- | ------ |
| MyTransformer | 30.77     | 0.6023      | 0.3713      | 0.2480      | 0.1635      | 0.2543 |

目前实现本项目的心得已总结在个人博客！欢迎大家来看！

[MyTransformer](https://pzzzzz5142.github.io/学习/NLP/MyTransformer)

当然，报告没写完😅。最近考试比较多，然后还有几个想法没有得到验证，有些地方实现的还比较粗糙，所以后面的内容后面再补吧。

同时为了方便大家阅读和讨论（~~我博客的评论区寄了，我也懒的研究怎么修了~~），这篇报告我也按照下方顺序将其拆分为几篇文章挂到知乎上了，当然最新进展还是以上方博客为主（~~最近应该是不会有什么进展~~），欢迎大家点赞投币收藏（bushi。

## 已实现部分

知乎链接🔗：[从 0 开始的 Transformer 复现·前言（一）](https://zhuanlan.zhihu.com/p/437981886)

+ 少量的 BUG。

    ~~（想啥呢，怎么可能专门介绍 Bug 的）~~

+ Data

    知乎链接🔗：[从 0 开始的 Transformer 复现·数据（二）](https://zhuanlan.zhihu.com/p/438123116)

    + Data Preprocessing
    + DataLoading
        + Memory Pinning [reference](https://pytorch.org/docs/stable/data.html#memory-pinning)

+ Transformer 模型
    
    知乎链接🔗：[从 0 开始的 Transformer 复现·模型（三）](https://zhuanlan.zhihu.com/p/438632726)

    + LayerNorm [reference](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html?highlight=layer#torch.nn.LayerNorm)
    + Pre-norm & Post-norm
    
+ Label Smooth Cross Entropy
    
    知乎链接🔗：tbd
    
    + Ignore padding index [reference](https://discuss.pytorch.org/t/ignore-index-in-the-cross-entropy-loss/25006/9)
    
+ Training 部分
    
    知乎链接🔗：tbd
    
    + lr_scheduler
    
+ Inference 部分（写了一个比较 low 的 beam search）

    知乎链接🔗：tbd

## Known Issues

+ inference 性能问题
+ 测试集 sacrebleu 30.77

## Usage

预处理数据。数据在[这里](https://git.io/JPK9N)下载。

```
python preprocess.py --data-path ~/path/to/iwslt14.tokenized.de-en --target-lang en --source-lang de
```

训练

```
python train.py --data data-bin/iwslt14.tokenized.de-en --src-lang de --tgt-lang en --epoch 3 --model-config config/iwslt.de-en.yaml
```

推断

```
python inference.py --data data-bin/iwslt14.tokenized.de-en --src-lang de --tgt-lang en --model-path path/to/checkpoint.pt
```

## 效果展示

取测试集前 3 个句子作为展示。

### 例一

源语句：

wissen sie , eines der großen vernügen beim reisen und eine der freuden bei der ethnographischen forschung ist , gemeinsam mit den menschen zu leben , die sich noch an die alten tage erinnern können . die ihre vergangenheit noch immer im wind spüren , sie auf vom regen geglätteten steinen berühren , sie in den bitteren blättern der pflanzen schmecken .

目标语句：

you know , one of the intense pleasures of travel and one of the delights of ethnographic research is the opportunity to live amongst those who have not forgotten the old ways , who still feel their past in the wind , touch it in stones polished by rain , taste it in the bitter leaves of plants .

模型预测：

you know , one of the great commons with travel , and one of the pleasures with ethnography research is to live together with the people who can remember the old days . their past is still in the wind , they still feel a touch of the plants .

### 例二

源语句：

einfach das wissen , dass jaguar-schamanen noch immer jenseits der milchstraße reisen oder die bedeutung der mythen der ältesten der inuit noch voller bedeutung sind , oder dass im himalaya die buddhisten noch immer den atem des dharma verfolgen , bedeutet , sich die zentrale offenbarung der anthropologie ins gedächtnis zu rufen , das ist der gedanke , dass die welt , in der wir leben , nicht in einem absoluten sinn existiert , sondern nur als ein modell der realität , als eine folge einer gruppe von bestimmten möglichkeiten der anpassung die unsere ahnen , wenngleich erfolgreich , vor vielen generationen wählten .

目标语句：

just to know that jaguar shamans still journey beyond the milky way , or the myths of the inuit elders still resonate with meaning , or that in the himalaya , the buddhists still pursue the breath of the dharma , is to really remember the central revelation of anthropology , and that is the idea that the world in which we live does not exist in some absolute sense , but is just one model of reality , the consequence of one particular set of adaptive choices that our lineage made , albeit successfully , many generations ago .

模型预测：

it &apos;s just knowing that hunters are still traveling beyond the milky way , or the myths of the oldest of the inuit , or that in the himalaya , the buddhists still continue to pursue the breath of the dharma , the central response of the anthropology of anthropology , which is that the world that we live in is not just an absolute sense , but as a model of reality , as a consequence of the most adaptation of our species , is to be successful in many generations .

### 例三

源语句：

und natürlich teilen wir alle dieselben anpassungsnotwendigkeiten .

目标语句：

and of course , we all share the same adaptive imperatives .

模型预测：

and of course , we all share the same adaptation .
