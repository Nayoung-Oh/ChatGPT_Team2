import json
def generate_total_info():
    # Currently temporaily change 데 로오는 to 데로오는 due to processing
    with open("train.json", "r", encoding='utf-8-sig') as f:
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
            min_len = min(len(dialect), len(standard))
            dialect = dialect[:min_len]
            standard = standard[:min_len]
        dialect_words = []
        standard_words = []
        for d, s in zip(dialect, standard):
            if d != s:
                dialect_words.append(d)
                standard_words.append(s)
        return_dict['dialect words'] = dialect_words
        return_dict['standard words'] = standard_words
        return_list.append(return_dict)
    with open("train_result.json", "w", encoding='utf-8-sig') as f:
        json.dump(return_list, f, indent='\t', ensure_ascii=False)

if __name__ == '__main__':
    generate_total_info()
    tot_len = []
    tot_dia = []
    with open("train_result.json", "r", encoding='utf-8-sig') as f:
        naive_info = json.load(f)
    for l in naive_info:
        tot_len.append(len(l['dialect'].split()))
        tot_dia.append(len(l['dialect words']))

