import json
import os
import random

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.json')]

def read_sentences(file):
    with open(file, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    sentence_data = []
    for d in data['utterance']:
        tmp = {}
        tmp['dialect'] = d['dialect_form']
        tmp['standard'] = d['standard_form']
        tmp['words'] = list(set([k['eojeol'] for k in d['eojeolList'] if k['isDialect']]))
        for w in tmp['words']:
            if w not in tmp['dialect']:
                continue
        if len(tmp['words']) != 0:
            sentence_data.append(tmp)
    return sentence_data

if __name__ == '__main__':
    folder = listdir_fullpath('한국어 방언 발화(제주도)/Validation/[라벨]제주도_학습용데이터_3')
    sentence_data = []
    for file in folder:
        sentence_data.extend(read_sentences(file))
    one_dialect = []
    multiple_dialects = []
    length = 0
    for s in sentence_data:
        length += len(s['dialect'])
        length += len(s['standard'])
        if len(s['words']) > 3:
            multiple_dialects.append(s)
        else:
            one_dialect.append(s)

    dialect_dict = set()
    for s in multiple_dialects:
        dialect_dict.update(s['words'])
    print(len(dialect_dict))
    print(length)
    one_dialect_words = {}

    for s in one_dialect:
        word = s['words'][0]
        if word in one_dialect_words:
            one_dialect_words[word] += 1
        else:
            one_dialect_words[word] = 1

    # fine tune 30, test 10 => generate multiple pairs
    pairs = []
    pair_num = len(sentence_data) // 12

    for i in range(pair_num):
        tmp_train = []
        tmp_words = set()
        for j in range(10):
            s = sentence_data.pop(random.randrange(len(sentence_data)))
            tmp_train.append(s)
            tmp_words.update(s['words'])
        tmp_test = []
        while len(tmp_test) < 1:
            s = sentence_data.pop(random.randrange(len(sentence_data)))
            if len(set(s['words'])-tmp_words) == 0:
                tmp_test.append(s)
            else:
                sentence_data.append(s)
        while len(tmp_test) < 2:
            s = sentence_data.pop(random.randrange(len(sentence_data)))
            if len(set(s['words'])-tmp_words) != 0:
                tmp_test.append(s)
            else:
                sentence_data.append(s)
        pair = {}
        pair['train'] = tmp_train
        pair['test'] = tmp_test
        pairs.append(pair)

    with open('result.json', 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent='\t', ensure_ascii=False)