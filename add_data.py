# from sentence_transformers import SentenceTransformer, util
from bert4vec import Bert4Vec
import numpy as np
import jsonlines
from datasets import load_dataset, Dataset, DatasetDict
import json
import faiss
from datasets import load_from_disk

def get_dataset():
    # 生成domain句子列表
    max_corpus_size = 20000
    max_query_size = 1000
    cnt = 0
    sentences_path = "./domain.txt"
    with open(sentences_path, 'a') as file:
        with open('./domain.jsonl', 'r', encoding='utf-8') as f:
            for item in jsonlines.Reader(f):
                file.write(item['text'].replace('\n', '', -1)+'\n')
                cnt += 1
                if cnt >= max_corpus_size:
                    
    # 生成queries
    queries_bio = []
    queries_chem = []
    with open('./task_bio.jsonl', 'r', encoding='utf-8') as f:
        for i in jsonlines.Reader(f):
            queries_bio.append(i['text'])
            if len(queries_bio) >= max_query_size:
                break
    with open('./task_chem.jsonl', 'r', encoding='utf-8') as f:
        for i in jsonlines.Reader(f):
            queries_chem.append(i['text'])
            if len(queries_chem) >= max_query_size:
                break
    print('Start encoding sentences...\n')
    model = Bert4Vec(mode='paraphrase-multilingual-minilm')
    model.build_index(sentences_path, ann_search=True, gpu_index=True, n_search=32, batch_size=64)
    print('Start searching for top k...\n')
    results_bio = model.search(queries=queries_bio, threshold=0.6, top_k=5)
    results_chem = model.search(queries=queries_bio, threshold=0.6, top_k=5)

    text_bio = []
    text_chem = []
    for sent_key in results_bio:
        for sent in sent_key:
            text_bio.append(sent[0])
    for sent_key in results_chem:
        for sent in sent_key:
            text_chem.append(sent[0])
    
    dict_bio = {'text': text_bio}
    dict_chem = {'text': text_chem}
    
    ds_bio = DatasetDict({'train': Dataset.from_dict(dict_bio)})
    ds_chem = DatasetDict({'train': Dataset.from_dict(dict_chem)})
    with open('./data_selected_bio.txt', 'a') as f:
    for item in ds_bio['train']:
        f.write(item['text']+'\n')
    with open('./data_selected_chem.txt', 'a') as f:
        for item in ds_bio['train']:
            f.write(item['text']+'\n')
    ds_bio.save_to_disk("./datasets/bio_data")
    ds_chem.save_to_disk("./datasets/chem_data")
    return ds_bio, ds_chem