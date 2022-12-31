import json
import jsonlines
from datasets import load_dataset, Dataset, DatasetDict

def dict2txt(name, type_name):
    f = open("data.txt", 'a')
    if name == 'chemprot':
        file = open('./chemprot/' + type_name + '.jsonl', 'r')
        for line in file.readlines():
            data = json.loads(line)
            f.write(data['text']+'\n')
    elif name == 'bioasq':
        data = json.load(open('./bioasq/' + type_name + '.json', 'r'))
        for d in data:
            tmp_str = d['question']
            for sent in d['text']:
                tmp_str = tmp_str + sent
            f.write(tmp_str+'\n')
    f.close()
    return

def main():
    dict2txt('chemprot', 'train')
    dict2txt('chemprot', 'test')
    dict2txt('chemprot', 'dev')
    dict2txt('bioasq', 'train')
    dict2txt('bioasq', 'test')

    # 先生成task.jsonl
    fj = jsonlines.open('task_chem.jsonl', 'a')
    f = open('data.txt', 'r')
    tmp_dict = {}
    cnt = 1
    for line in f.readlines():
        tmp_dict['index'] = cnt
        tmp_dict['text'] = line[:-1]
        jsonlines.Writer.write(fj, tmp_dict)
        cnt += 1
    f.close()
    fj.close()
    
    # 再生成domain.jsonl
    ds_split = load_dataset('ccdv/pubmed-summarization', split='train')
    print(ds_split)
    cnt = 10066
    tmp_dict = {}
    fj = jsonlines.open('domain.jsonl', 'a')
    for item in ds_split:
        tmp_dict['index'] = cnt
        tmp_dict['text'] = item['abstract']
        jsonlines.Writer.write(fj, tmp_dict)
        cnt += 1
    fj.close()
    
    # 生成world.jsonl
    fj_bio = jsonlines.open('world_bio.jsonl', 'a')
    fj_chem = jsonlines.open('world_chem.jsonl', 'a')
    f = open('domain.jsonl', 'r')
    for line in f.readlines():
        data = json.loads(line)
        jsonlines.Writer.write(fj_bio, data)
        jsonlines.Writer.write(fj_chem, data)
    f.close()
    f = open('task_bio.jsonl', 'r')
    for line in f.readlines():
        data = json.loads(line)
        jsonlines.Writer.write(fj_bio, data)
    f.close()
    f = open('task_chem.jsonl', 'r')
    for line in f.readlines():
        data = json.loads(line)
        jsonlines.Writer.write(fj_chem, data)
    f.close()
    fj_bio.close()
    fj_chem.close()
    
if __name__ == "__main__":
    main()