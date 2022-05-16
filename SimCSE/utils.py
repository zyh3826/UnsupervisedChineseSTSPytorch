# coding: utf-8
DATASET_ROOT_PATH = '/source/c0/NLPSource/Datasets/STS'
MODEL_ROOT_PATH = '/source/c0/NLPSource/embedding/transformer_based'
SAVE_ROOT = '/source/c0/NLPSource/embedding/simcse'

DATASETS = {
    'STS_B': 64,
    'LCQMC': 64,
    # 'ChineseTextualInference': 32,
    # 'BQ_corpus': 32,
    # 'ATEC': 32,
    # 'ChineseSTS': 32,
    # 'cnsd_mnli': 32,
    # 'cnsd_snli': 32,
    }
VEC_TYPE = ['cls', 'cls_before_pooler', 'avg', 'avg_first_last', 'avg_top2']
MODELS = {
    'simbert_L12_H768': 'chinese_simbert_L-12_H-768_A-12',
    'bert_base_chinese': 'bert_base_chinese',
    'hit_roberta_wwm_ext': 'hit_chinese_roberta_wwm_ext',
    # 'hit_roberta_wwm_ext_large': 'hit_chinese_roberta_wwm_large_ext',
    'hit_bert_wwm_ext': 'hit_chinese_wwm_ext',
    'hit_bert_wwm': 'hit_chinese_wwm',
    # 'bert_uer_large': 'tencent_uer_bert_large',
}
DATASETS_PATH = {
    'STS_B': 'STS_B/STS-B.test.data',
    'LCQMC': 'LCQMC/LCQMC.test.data',
    # 'ChineseTextualInference': 'ChineseTextualInference/train.txt',
    # 'BQ_corpus': 'BQ_corpus/test.csv',
    # 'ATEC': 'ATEC/atec_nlp_sim_train.csv',
    # 'ChineseSTS': 'ChineseSTS/simtrain_to05sts.txt',
    # 'cnsd_mnli': 'cnsd_mnli/cnsd_multil_dev_matched.jsonl',
    # 'cnsd_snli': 'cnsd_snli/cnsd_snli_v1.0.test.jsonl',
    }
