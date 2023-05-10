import json
import nltk.translate.bleu_score as bleu
import torch
from transformers import AutoModel, AutoTokenizer
import os
import ast
import csv
import numpy as np

class Scoring:
    def __init__(self):
        self.model = AutoModel.from_pretrained('BM-K/KoSimCSE-roberta-multitask')  # or 'BM-K/KoSimCSE-bert-multitask'
        self.tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta-multitask')  # or 'BM-K/KoSimCSE-bert-multitask'

    def cal_score(self, a, b):
        if len(a.shape) == 1: a = a.unsqueeze(0)
        if len(b.shape) == 1: b = b.unsqueeze(0)

        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / b.norm(dim=1)[:, None]
        return torch.mm(a_norm, b_norm.transpose(0, 1)).item()

    def get_sentence_similarity(self, sentences):
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        embeddings, _ = self.model(**inputs, return_dict=False)

        return self.cal_score(embeddings[0][0], embeddings[1][0])

    def get_bleu(self, answer, resp):
        # answer is the target sentence, resp is the response
        return bleu.sentence_bleu(list(map(lambda ref: ref.split(), [answer])), resp.split(), smoothing_function=bleu.SmoothingFunction().method4)

    def get_new_split(self, sentence, dict):
        tmp_words = sentence.split()
        words = []
        for word in tmp_words:
            find_match = False
            if word in [',', '.', '?', '!']:
                continue
            for dialect in dict:
                if dialect in word:
                    if dialect == word:
                        words.append(word)
                    else:
                        words.append(dialect)
                        sub = word.partition(dialect)[-1]
                        words.append(sub)
                    find_match = True
                    break
            if not find_match:
                words.append(word)
        return words

    def get_dialect_correction(self, original, generated, answer, dialect_dict, standard_dict):
        words = self.get_new_split(original, dialect_dict)
        dialects = []
        standard = []
        for word in words:
            if word in dialect_dict:
                dialects.append(word)
            else:
                standard.append(word)
        
        sim_word = []
        ans_words = self.get_new_split(answer, standard_dict)
        for ans_word in ans_words:
            if ans_word not in standard:
                sim_word.append(ans_word)
        
        count = 0
        for s in sim_word:
            # possibly get similar words
            if s in generated:
                count += 1
        if len(dialects) == 0:
            print(original, dialect_dict)
        correction_score = count / len(dialects)
        return correction_score

    def get_stadard_language_preserve(self, original, generated, answer, dialect_dict, standard_dict):
        words = self.get_new_split(original, dialect_dict)
        standard = []
        for word in words:
            if word not in dialect_dict:
                standard.append(word)
        
        dialect_count = 0
        ans_words = self.get_new_split(answer, standard_dict)
        for ans_word in ans_words:
            if ans_word not in standard:
                dialect_count += 1
        
        count = 0
        for word in words:
            if word in generated:
                count += 1
        preserve_score = 2*count / (len(words)+len(self.get_new_split(generated, standard_dict))-dialect_count)

        return preserve_score

def generate_total_info():
    # Currently temporaily change 데 로오는 to 데로오는 due to processing
    with open("naive_info.json", "r", encoding='utf-8-sig') as f:
        naive_info = json.load(f)
    return_list = []
    for idx, l in enumerate(naive_info):
        return_dict = {}
        for k, v in l.items():
            return_dict[k] = v
        dialect = return_dict['dialect'].split()
        dialect = [d for d in dialect if d not in ['.', ',', '?', '!']]
        standard = return_dict['standard'].split()
        standard = [s for s in standard if s not in ['.', ',', '?', '!']]
        if len(dialect) != len (standard):
            print(idx)
            continue
        dialect_words = []
        standard_words = []
        for d, s in zip(dialect, standard):
            if d != s:
                dialect_words.append(d)
                standard_words.append(s)
        return_dict['dialect words'] = dialect_words
        return_dict['standard words'] = standard_words
        return_list.append(return_dict)
    with open("result.json", "w", encoding='utf-8-sig') as f:
        json.dump(return_list, f, indent='\t', ensure_ascii=False)


if __name__ == "__main__":
    if not os.path.isfile("result.json"):
        generate_total_info()
    
    score = Scoring()
    with open("result.json", "r", encoding='utf-8-sig') as f:
        total_info = json.load(f)
    final_result = []
    final_result.append(["dialect", "standard", "response", "BLEU_score", \
                         "ori_sim", "new_sim", "relative_sim", "dialect_correction", "standard_preserve"])
    with open("archive_result_vanilla.csv", "r", encoding='utf-8-sig') as read_file:
        read_csv = csv.reader(read_file)
        next(read_csv, None)
        for l in read_csv:
            original = l[0]
            answer = l[1]
            response = l[2]
            dialect_dict = []
            standard_dict = []
            for d in total_info:
                if d["dialect"] == original:
                    dialect_dict = d["dialect words"]
                    standard_dict = d["standard words"]
                    break
            scores = []
            BLEU_score = round(score.get_bleu(answer, response), 3)
            ori_sim = round(score.get_sentence_similarity([original, answer]), 3)
            new_sim = round(score.get_sentence_similarity([response, answer]), 3)
            relative_sim = round(new_sim/ori_sim, 3)
            dialect_correction = round(score.get_dialect_correction(original, response, answer, dialect_dict, standard_dict), 3)
            standard_preserve = round(score.get_stadard_language_preserve(original, response, answer, dialect_dict, standard_dict), 3)
            final_result.append([original, answer, response, BLEU_score, ori_sim,\
                                new_sim, relative_sim,\
                                dialect_correction, standard_preserve])
    
    
    with open("vanilla_score.csv", "w",  newline='', encoding='utf-8-sig') as write_file:
        result_file = csv.writer(write_file, delimiter=',')
        result_file.writerows(final_result)
    
    tmp_result = np.array(final_result[1:])[:, 3:]
    tmp_result = tmp_result.astype(np.float64)
    print("avg score: ", )
    print(np.mean(tmp_result, axis=0))
    print(np.std(tmp_result, axis=0))
    # [0.03983333 0.71516667 0.5877     0.83036667 0.1133     0.4872    ]
    # [0.1161     0.71516667 0.65956667 0.93083333 0.14866667 0.57996667]
    # [0.09883333 0.71516667 0.7012     1.02256667 0.17846667 0.52643333]