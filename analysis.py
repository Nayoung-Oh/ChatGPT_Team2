import csv
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy import stats

def get_different_performance(naive_result, result):
    our_one_better_BLEU = []
    our_one_better_sim = []
    our_one_better_corr = []
    our_one_better_pres = []
    total_bad = []

    for naive, result in zip(naive_result, result):
        naive[3:] = [float(n) for n in naive[3:]]
        result[3:] = [float(n) for n in result[3:]]
        temp = []
        temp.extend(naive.copy())
        temp.extend(result[2:].copy())
        if naive[3] < result[3] - 0.6:
            our_one_better_BLEU.append(temp)
        if naive[6] < 0.8 and result[6] > 1.2:
            our_one_better_sim.append(temp)
        if naive[7] < result[7] - 0.6:
            our_one_better_corr.append(temp)
        if naive[8] < result[8] - 0.6:
            our_one_better_pres.append(temp)
        if naive[6] < 0.9 and result[6] < 0.9 and naive[7] < 0.6 and result[7] < 0.6 and naive[8] < 0.6 and result[8] < 0.6:
            total_bad.append(temp)

    result = []
    result.append(["dialect", "standard", "naive_response", "BLEU_score", \
                            "ori_sim", "new_sim", "relative_sim", "dialect_correction", "standard_preserve", "new_response"])
    result.append(["BLEU"])
    result.extend(our_one_better_BLEU)
    result.append(["sim"])
    result.extend(our_one_better_sim)
    result.append(["corr"])
    result.extend(our_one_better_corr)
    result.append(["pres"])
    result.extend(our_one_better_pres)
    result.append(["BAD"])
    result.extend(total_bad)
    return result

def check_impact_of_sentence_length(naive_result, result, total_info):
    coeffs = []
    for res in [naive_result, result]:
        length_list = []
        scores = []
        for l in res:
            dialect_count = 0
            for d in total_info:
                if d["dialect"] == l[0]:
                    dialect_count = len(d['dialect words'])
            length_list.append(len(l[0].split()) - dialect_count)
            s = [float(l[i]) for i in [3, 6, 7, 8]]
            scores.append(s)
        x = np.array(length_list)
        ys = np.array(scores)
        tmp_coeff = []
        for i in range(4):
            tmp_coeff.append(np.corrcoef(x, ys[:, i])[0, 1])
        coeffs.append(tmp_coeff)
    return coeffs

def paired_t_test(naive_result, result):
    naive_val = []
    feature_val = []
    for nl, rl in zip(naive_result, result):
        naive_val.append([float(nl[i]) for i in [3, 6, 7, 8]])
        feature_val.append([float(rl[i]) for i in [3, 6, 7, 8]])

    naive_val = np.array(naive_val, dtype=np.float32)
    feature_val = np.array(feature_val, dtype=np.float32)
    print(naive_val.shape, feature_val.shape)
    stat, p_val = stats.ttest_rel(naive_val, feature_val, alternative='two-sided')
    return stat, p_val
    
if __name__ == '__main__':
    naive_result = []
    with open("naive_score.csv", "r", encoding='utf-8-sig') as f:
        read_csv = csv.reader(f)
        next(read_csv, None)
        for l in read_csv:
            naive_result.append(l)

    result = []
    with open("score.csv", "r", encoding='utf-8-sig') as f:
        read_csv = csv.reader(f)
        next(read_csv, None)
        for l in read_csv:
            result.append(l)

    # extract meaningful sentences
    score_filtering = get_different_performance(naive_result, result)
    with open("analysis.csv", "w", newline='', encoding='utf-8-sig') as f:
        result_file = csv.writer(f, delimiter=',')
        result_file.writerows(score_filtering)

    # check impact of sentence length
    with open("result.json", "r", encoding='utf-8-sig') as f:
        total_info = json.load(f)
    
    coeffs = check_impact_of_sentence_length(naive_result, result, total_info)
    print(coeffs)
    # positive means, longer sentence better performance
    # 0.15919314704001628
    # -0.2069618892046019
    # 0.018518710539204376
    # 0.061295888144040515
    # ********
    # 0.08105750674624523
    # -0.39396608050177556
    # -0.12301859627005571
    # 0.5001123965158196
    # ********

    # paired T-test
    stat, p_val = paired_t_test(naive_result, result)
    print('statistic:', stat, '   p-value:', p_val)
    # statistic: [-6.84156306 -5.35348158 -5.16551618 -5.07883041]    p-value: [6.57664308e-10 5.59636709e-07 1.24352512e-06 1.78826114e-06]

    
    