https://kexue.fm/archives/8321 的pytorch版<br>
环境：
```
Python: 3.8
numpy: 1.19.1
transformers: 4.4.2
torch: 1.7.1
scipy: 1.5.2
cuda: 10.2
GPU: RTX 2080Ti
batch size: 1024
```

七种模型：

```
simbert, bert_base, bert_wwm_ext, bert_wwm_ext_large, wobert, roformer, bert_large
```

八个数据集：

```
STS_B
LCQMC
ChineseTextualInference
BQ_corpus
ATEC
ChineseSTS
cnsd_mnli
cnsd_snli
```

五种构造向量的方式：

```
last
last2avg
first_last_avg
cls
pooler
```

四种降维方式：

```
不降维
384
256
128
```

总共约1400种不同情况。结果分析见result文件夹。
