# 概况
+ 模型：`simbert_L12_H768`，`bert_base_chinese`，`hit_roberta_wwm_ext`，`hit_bert_wwm_ext`，`hit_bert_wwm`
+ 数据集：`LCQMC`，`STS_B`
+ `pooler type`：`cls`, `cls_before_pooler`, `avg`, `avg_first_last`, `avg_top2`
+ `sample`：随机采样10k，不采样
+ 是否接降维层：是，否
+ 降维层大小：512, 256, 128, 64, 32
总共600总情况
# 参数
```json
{
    "num_train_epochs": 1,
    "per_device_train_batch_size": 64,
    "learning_rate": 1e-5,
    "temp": 0.05,
    "hidden_dropout_prob": 0.3
}
```
注：结果中维度为768表示不降维
# simbert_L12_H768
在`LCQMC`数据集上
不管是那种pooler type方式，随机采样10k的结果都好于不采样。对于降维，采样后降维前后效果差别不大；而不采样降维效果大幅好于不降维。
![pic](./simcse_results_with_dimension_reduction_figs/simbert_L12_H768_LCQMC_compare_bettween_sample.png)
无论是否采样，综合表现最好的pooler type是`cls_before_pooler`，且降维在一定程度上有提升。
![pic](./simcse_results_with_dimension_reduction_figs/simbert_L12_H768_LCQMC_compare_bettween_pooler_type.png)
在`STS_B`数据集上
不管是那种pooler type方式，不采样的结果都好于随机采样10k，但他们之间的差别没有`LCQMC`数据集这么大。对于降维，不管采样还是不采样，不降维效果要好于降维，同样，其差别不大
![pic](./simcse_results_with_dimension_reduction_figs/simbert_L12_H768_STS_B_compare_bettween_sample.png)
无论是否采样，综合表现最好的pooler type同样是`cls_before_pooler`，但是降维并没有带来提升，反而是效果变差
![pic](./simcse_results_with_dimension_reduction_figs/simbert_L12_H768_STS_B_compare_bettween_pooler_type.png)

# bert_base_chinese
在`LCQMC`数据集上
不管是那种pooler type方式，随机采样10k的结果都好于不采样。对于降维，采样后降维有一定程度下降；而不采样降维效果大幅好于不降维。
![pic](./simcse_results_with_dimension_reduction_figs/bert_base_chinese_LCQMC_compare_bettween_sample.png)
对于不采样，综合表现最好的pooler type是`avg_top2`，且降维有大幅提升。对于采样，综合表现最好的pooler type是`avg`，降维后效果有下降。
![pic](./simcse_results_with_dimension_reduction_figs/bert_base_chinese_LCQMC_compare_bettween_pooler_type.png)
在`STS_B`数据集上
不管是那种pooler type方式，不采样的结果都好于随机采样10k，但他们之间的差别没有`LCQMC`数据集这么大。对于降维，不管采样还是不采样，不降维效果要好于降维
![pic](./simcse_results_with_dimension_reduction_figs/bert_base_chinese_STS_B_compare_bettween_sample.png)
无论是否采样，综合表现最好的pooler type是`avg`，但是降维并没有带来提升，反而是效果变差
![pic](./simcse_results_with_dimension_reduction_figs/bert_base_chinese_STS_B_compare_bettween_pooler_type.png)

# hit_roberta_wwm_ext
在`LCQMC`数据集上
不管是那种pooler type方式，随机采样10k的结果都好于不采样。对于降维，采样后降维有一定程度下降；而不采样降维效果大幅好于不降维。
![pic](./simcse_results_with_dimension_reduction_figs/hit_roberta_wwm_ext_LCQMC_compare_bettween_sample.png)
对于不采样，综合表现最好的pooler type是`avg_first_last`，且降维有大幅提升。对于采样，综合表现最好的pooler type是`avg`，降维后效果有下降。
![pic](./simcse_results_with_dimension_reduction_figs/hit_roberta_wwm_ext_LCQMC_compare_bettween_pooler_type.png)
在`STS_B`数据集上
不管是那种pooler type方式，不采样的结果都好于随机采样10k，但他们之间的差别没有`LCQMC`数据集这么大。对于降维，不管采样还是不采样，不降维效果要好于降维
![pic](./simcse_results_with_dimension_reduction_figs/hit_roberta_wwm_ext_STS_B_compare_bettween_sample.png)
无论是否采样，综合表现最好的pooler type是`avg`，但是降维并没有带来提升，反而是效果变差
![pic](./simcse_results_with_dimension_reduction_figs/hit_roberta_wwm_ext_STS_B_compare_bettween_pooler_type.png)

# hit_bert_wwm_ext
在`LCQMC`数据集上
不管是那种pooler type方式，随机采样10k的结果都好于不采样。对于降维，采样后降维有一定程度下降；而不采样降维效果大幅好于不降维。
![pic](./simcse_results_with_dimension_reduction_figs/hit_bert_wwm_ext_LCQMC_compare_bettween_sample.png)
对于不采样，综合表现最好的pooler type是`avg_first_last`，且降维有大幅提升。对于采样，综合表现最好的pooler type是`avg`，降维后效果有下降。
![pic](./simcse_results_with_dimension_reduction_figs/hit_bert_wwm_ext_LCQMC_compare_bettween_pooler_type.png)
在`STS_B`数据集上
不管是那种pooler type方式，不采样的结果都好于随机采样10k，但他们之间的差别没有`LCQMC`数据集这么大。对于降维，不管采样还是不采样，不降维效果要好于降维
![pic](./simcse_results_with_dimension_reduction_figs/hit_bert_wwm_ext_STS_B_compare_bettween_sample.png)
无论是否采样，综合表现最好的pooler type是`avg`，但是降维并没有带来提升，反而是效果变差
![pic](./simcse_results_with_dimension_reduction_figs/hit_bert_wwm_ext_STS_B_compare_bettween_pooler_type.png)

# hit_bert_wwm
在`LCQMC`数据集上
不管是那种pooler type方式，随机采样10k的结果都好于不采样。对于降维，采样后降维有一定程度下降；而不采样降维效果大幅好于不降维。
![pic](./simcse_results_with_dimension_reduction_figs/hit_bert_wwm_LCQMC_compare_bettween_sample.png)
对于不采样，综合表现最好的pooler type是`avg_first_last`，且降维有大幅提升。对于采样，综合表现最好的pooler type是`avg`，降维后效果有下降。
![pic](./simcse_results_with_dimension_reduction_figs/hit_bert_wwm_LCQMC_compare_bettween_pooler_type.png)
在`STS_B`数据集上
不管是那种pooler type方式，不采样的结果都好于随机采样10k，但他们之间的差别没有`LCQMC`数据集这么大。对于降维，不管采样还是不采样，不降维效果要好于降维
![pic](./simcse_results_with_dimension_reduction_figs/hit_bert_wwm_STS_B_compare_bettween_sample.png)
无论是否采样，综合表现最好的pooler type是`avg`，但是降维并没有带来提升，反而是效果变差
![pic](./simcse_results_with_dimension_reduction_figs/hit_bert_wwm_STS_B_compare_bettween_pooler_type.png)

# 不同模型之间的比较
## LCQMC数据集
采样情况下：不管哪种pooler type都是simBert效果最好。
![pic](./simcse_results_with_dimension_reduction_figs/LCQMC_random_sample_10k_data_compare_bettween_model.png)
不采样：`avg`和`avg_first_last`情况下`hit_bert_wwm`表现最好；使用`avg_top2`，`bert_base_chinese`和`hit_bert_wwm`不相上下；`cls`比较混乱，在32，64，128维度下`simbert`效果最好，256维度时`bert_base_chinese`效果最好，往后就是`hit_roberta_wwm_ext`效果最好；`cls_before_pooler`情况下`simbert`效果最好。
![pic](./simcse_results_with_dimension_reduction_figs/LCQMC_all_data_compare_bettween_model.png)
## STS_B数据集
采样情况下：不管哪种pooler type都是simBert效果最好。
![pic](./simcse_results_with_dimension_reduction_figs/STS_B_random_sample_10k_data_compare_bettween_model.png)
不采样：不管哪种pooler type都是simBert效果最好。
![pic](./simcse_results_with_dimension_reduction_figs/STS_B_all_data_compare_bettween_model.png)