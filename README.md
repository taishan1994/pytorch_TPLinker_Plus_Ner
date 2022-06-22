# pytorch_TPLinker_Plus_Ner
延申
- 一种基于多头选择的命名实体识别：https://github.com/taishan1994/pytorch_Multi_Head_Selection_Ner
- 一种基于bert_bilstm_crf的命名实体识别：https://github.com/taishan1994/pytorch_bert_bilstm_crf_ner
- 一种one vs rest方法进行命名实体识别：https://github.com/taishan1994/pytorch_OneVersusRest_Ner
- 一种级联Bert用于命名实体识别，解决标签过多问题：https://github.com/taishan1994/pytorch_Cascade_Bert_Ner
- 中文命名实体识别最新进展：https://github.com/taishan1994/awesome-chinese-ner
- 信息抽取三剑客：实体抽取、关系抽取、事件抽取：https://github.com/taishan1994/chinese_information_extraction
- 一种基于机器阅读理解的命名实体识别：https://github.com/taishan1994/BERT_MRC_NER_chinese
- W2NER：命名实体识别最新sota：https://github.com/taishan1994/W2NER_predict

****

基于pytorch的TPLinker_plus进行中文命名实体识别。

之前的多头选择不能够解决嵌套的实体识别，因为它对每一个字符对进行的是多分类，也就是只能属于一种实体类型，而这里的可以识别嵌套的实体。还是和之前其它几种实体识别方式相同的代码模板，这里稍微做了一些修改，主要是在数据加载方面。之前都是预先处理好所有需要的数据保存好，由于tplinker需要更多内存，这里使用DataLoader中的collate_fn对每一批的数据分别进行操作，可以大大减少内存的使用。模型主要是来自这里：[tplinker_plus](https://github.com/Tongjilibo/bert4torch/blob/master/examples/sequence_labeling/task_sequence_labeling_ner_tplinker_plus.py)，需要额外了解的知识有：[基于Conditional Layer Normalization的条件文本生成 - 科学空间|Scientific Spaces](https://spaces.ac.cn/archives/7124)和[将“softmax+交叉熵”推广到多标签分类问题 - 科学空间|Scientific Spaces](https://www.spaces.ac.cn/archives/7359)。实现运行步骤如下：

- 1、在raw_data下新建一个process.py将数据处理成mid_data下的格式。
- 2、修改部分参数运行main.py，以进行训练、验证、测试和预测。

模型及数据下载地址：链接：https://pan.baidu.com/s/1B-e-MV1lOMQj2ur5MADRww?pwd=he3e  提取码：he3e

# 运行

在16GB的显存下都只能以batch_size=2进行运行。。。

```python
!python main.py \
--bert_dir="model_hub/chinese-bert-wwm-ext/" \
--data_dir="./data/cner/" \
--log_dir="./logs/" \
--output_dir="./checkpoints/" \
--num_tags=8 \
--seed=123 \
--gpu_ids="0" \
--max_seq_len=150 \
--lr=3e-5 \
--other_lr=3e-4 \
--train_batch_size=2 \
--train_epochs=1 \
--eval_batch_size=8 \
--max_grad_norm=1 \
--warmup_proportion=0.1 \
--adam_epsilon=1e-8 \
--weight_decay=0.01 \
--dropout_prob=0.3 \
```

### 结果

```python
precision:0.8806 recall:0.8999 micro_f1:0.8901
          precision    recall  f1-score   support

   TITLE       0.87      0.88      0.87       767
    RACE       0.88      0.93      0.90        15
    CONT       1.00      1.00      1.00        33
     ORG       0.89      0.90      0.89       543
    NAME       0.99      1.00      1.00       110
     EDU       0.82      0.94      0.88       109
     PRO       0.67      0.95      0.78        19
     LOC       1.00      1.00      1.00         2

micro-f1       0.88      0.90      0.89      1598

虞兔良先生：1963年12月出生，汉族，中国国籍，无境外永久居留权，浙江绍兴人，中共党员，MBA，经济师。
{'TITLE': [['中共党员', 41], ['经济师', 50]], 'RACE': [['汉族', 18]], 'CONT': [['中国国籍', 21]], 'NAME': [['虞兔良', 1]], 'EDU': [['MBA', 46]], 'LOC': [['浙江绍兴人', 35]]}
```



