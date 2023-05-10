import csv
import numpy as np

from statsmodels.stats.anova import AnovaRM
import pandas as pd
import pingouin as pg

def get_different_performance(naive_result, result_ex, result_with):
    our_one_better_BLEU = []
    our_one_better_sim = []
    our_one_better_corr = []
    our_one_better_pres = []
    total_bad = []
    # dialect,standard,response,BLEU_score,ori_sim,new_sim,relative_sim,dialect_correction,standard_preserve

    for naive, res_e, res_w in zip(naive_result, result_ex, result_with):
        naive[3:] = [float(n) for n in naive[3:]]
        res_e[3:] = [float(n) for n in res_e[3:]]
        res_w[3:] = [float(n) for n in res_w[3:]]
        temp = []
        temp.extend(naive.copy())
        temp.extend(res_e[2:].copy())
        temp.extend(res_w[2:].copy())
        if naive[3] < res_e[3] - 0.1 and res_e[3] < res_w[3] - 0.1:
            our_one_better_BLEU.append(temp)
        if naive[6] < res_e[6] and res_e[6] < 1 and res_w[6] > 1:
            our_one_better_sim.append(temp)
        if naive[7] < res_e[7] - 0.2 and res_e[7] < res_w[7] - 0.2:
            our_one_better_corr.append(temp)
        if naive[8] < res_e[8] - 0.2 and res_e[8] < res_w[8] - 0.2:
            our_one_better_pres.append(temp)
        if naive[6] < 0.9 and res_e[6] < 0.9 and res_w[6] < 0.9\
            and naive[7] < 0.6 and res_e[7] < 0.6 and res_w[7] < 0.6 \
            and naive[8] < 0.6 and res_e[8] < 0.6 and res_w[8] < 0.6:
            total_bad.append(temp)

    result = []
    result.append(["dialect", "standard", "naive_response", "BLEU_score", \
                             "ori_sim", "new_sim", "relative_sim", "dialect_correction", "standard_preserve", "new_response", "gram_response"])
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

def check_impact_of_sentence_length(naive_result, result_ex, result_with):
    for res in [naive_result, result_ex, result_with]:
        length_list = []
        scores = []
        for l in res:
            length_list.append(len(l[0].split()))
            s = [float(l[i]) for i in [3, 6, 7, 8]]
            scores.append(s)
        x = np.array(length_list)
        ys = np.array(scores)
        for i in range(4):
            print(np.corrcoef(x, ys[:, i])[0, 1])
        print("********")
    # -0.1486633772073761
    # -0.04155230285227387
    # 0.15247367002795428
    # 0.006801566881819235
    # ********
    # 0.26222584837041485
    # -0.016252819399444438
    # 0.040033214758286785
    # 0.4015342728041271
    # ********
    # 0.15034077484282068
    # -0.10979390889501163
    # -0.18853972908433078
    # 0.40156527548077575
    # ********

def ANOVA(naive_result, result_ex, result_with):
    # https://recipesds.tistory.com/entry/RMANOVA-Repeated-Measured-ANOVA-%EB%B0%98%EB%B3%B5%EC%B8%A1%EC%A0%95-ANOVA%EC%9D%98-%EC%9A%B0%EC%95%84%ED%95%A8
    naive_val = []
    feature_val = []
    grammar_val = []
    for nl, rl, gl in zip(naive_result, result_ex, result_with):
        naive_val.append([float(nl[i]) for i in [3, 6, 7, 8]])
        feature_val.append([float(rl[i]) for i in [3, 6, 7, 8]])
        grammar_val.append([float(gl[i]) for i in [3, 6, 7, 8]])

    ids = list(range(len(naive_val)))
    sent_type = ['vanila', 'example', 'all']
    data = []
    for i in ids:
        temp = [i, sent_type[0]]
        temp.extend(naive_val[i])
        data.append(temp)
        temp = [i, sent_type[1]]
        temp.extend(feature_val[i])
        data.append(temp)
        temp = [i, sent_type[2]]
        temp.extend(grammar_val[i])
        data.append(temp)

    data = pd.DataFrame(data, columns=['id', 'sent_type', 'BLEU', 'sim', 'corr', 'pres'])

    for score in ['BLEU', 'sim', 'corr', 'pres']:
        print(AnovaRM(data=data, depvar=score, subject='id', within=['sent_type']).fit())
        posthoc = pg.pairwise_ttests(dv=score, within='sent_type', subject='id', data=data)
        print(posthoc)

if __name__ == '__main__':
    naive_result = []
    with open("vanilla_score.csv", "r", encoding='utf-8-sig') as f:
        read_csv = csv.reader(f)
        next(read_csv, None)
        for l in read_csv:
            naive_result.append(l)

    result_ex = []
    with open("archive_score.csv", "r", encoding='utf-8-sig') as f:
        read_csv = csv.reader(f)
        next(read_csv, None)
        for l in read_csv:
            result_ex.append(l)

    result_with = []
    with open("feature_score.csv", "r", encoding='utf-8-sig') as f:
        read_csv = csv.reader(f)
        next(read_csv, None)
        for l in read_csv:
            result_with.append(l)

    score_filtering = get_different_performance(naive_result, result_ex, result_with)
    with open("analysis.csv", "w", newline='', encoding='utf-8-sig') as f:
        result_file = csv.writer(f, delimiter=',')
        result_file.writerows(score_filtering)
    ANOVA(naive_result, result_ex, result_with)


# Pr > F is p-value, should be smaller than 0.05, so in this case meaningful dif in BLEU and sim

#                  Anova
# =======================================
#           F Value Num DF  Den DF Pr > F
# ---------------------------------------
# sent_type  5.8891 2.0000 58.0000 0.0047
# =======================================
#     Contrast        A        B  Paired  Parametric         T   dof alternative     p-unc   BF10    hedges
# 0  sent_type      all  example    True        True -0.934249  29.0   two-sided  0.357890   0.29 -0.115070
# 1  sent_type      all   vanila    True        True  2.830562  29.0   two-sided  0.008353  5.246  0.657101
# 2  sent_type  example   vanila    True        True  2.610758  29.0   two-sided  0.014152  3.362  0.607604

#                  Anova
# =======================================
#           F Value Num DF  Den DF Pr > F
# ---------------------------------------
# sent_type 10.7675 2.0000 58.0000 0.0001
# =======================================
#     Contrast        A        B  Paired  Parametric         T   dof alternative     p-unc     BF10    hedges
# 0  sent_type      all  example    True        True  1.933959  29.0   two-sided  0.062931    0.998  0.346087
# 1  sent_type      all   vanila    True        True  4.962224  29.0   two-sided  0.000028  819.267  0.710033
# 2  sent_type  example   vanila    True        True  2.685464  29.0   two-sided  0.011854    3.901  0.483770

#                  Anova
# =======================================
#           F Value Num DF  Den DF Pr > F
# ---------------------------------------
# sent_type  1.4306 2.0000 58.0000 0.2475
# =======================================
#     Contrast        A        B  Paired  Parametric         T   dof alternative     p-unc   BF10    hedges
# 0  sent_type      all  example    True        True  1.046581  29.0   two-sided  0.303938   0.32  0.147393
# 1  sent_type      all   vanila    True        True  1.497225  29.0   two-sided  0.145141   0.53  0.319561
# 2  sent_type  example   vanila    True        True  0.843445  29.0   two-sided  0.405883  0.269  0.188737

#                  Anova
# =======================================
#           F Value Num DF  Den DF Pr > F
# ---------------------------------------
# sent_type  1.2508 2.0000 58.0000 0.2939
# =======================================
#     Contrast        A        B  Paired  Parametric         T   dof alternative     p-unc   BF10    hedges
# 0  sent_type      all  example    True        True -1.802935  29.0   two-sided  0.081800  0.814 -0.239023
# 1  sent_type      all   vanila    True        True  0.548354  29.0   two-sided  0.587646  0.223  0.127790
# 2  sent_type  example   vanila    True        True  1.398290  29.0   two-sided  0.172629  0.468  0.288731