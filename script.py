import openai
import nltk.translate.bleu_score as bleu
import json
import csv

API_KEY = "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"
openai.api_key = API_KEY
MODEL = "gpt-3.5-turbo"

def get_response(question):
    response = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": question},
    ],
    temperature=0.2,
    )
    return response['choices'][0]['message']['content']

def generate_sentences(pair, language='korean'):
    sentences = ''''''
    if language == 'korean':
        sentences += '한국어 방언 문장을 한국어 표준어 문장으로 바꿔줘. 예를 들면,\n'
    else:
        sentences += 'I want you to convert a Korean dialect sentence into a Korean standard language sentence. These are the examples.\n'
    for i in range(len(pair['train'])):
        if language == 'korean':
            tmp = f'''방언 {i+1} : {pair['train'][i]['dialect']}
표준어 {i+1} : {pair['train'][i]['standard']}
##
'''
        else:
            tmp = f'''dialect {i+1}: {pair['train'][i]['dialect']}
standard language {i+1}: {pair['train'][i]['standard']}
##
'''
        sentences += tmp
    if language == 'korean':
        sentences += '그러면, \n'
    else:
        sentences += 'Then, \n'
    if language == 'korean':
        tmp = f'''방언 : {pair['test'][0]['dialect']}
표준어 :
'''
    else:
        tmp = f'''dialect : {pair['test'][0]['dialect']}
standard language :
'''
    sentences += tmp

    if language == 'korean':
        quest = '그러면, \n'
    else:
        quest = 'Then, \n'
    if language == 'korean':
        tmp = f'''방언 : {pair['test'][1]['dialect']}
표준어 :
'''
    else:
        tmp = f'''dialect : {pair['test'][1]['dialect']}
standard language :
'''
    quest += tmp
        
    return sentences, quest


if __name__ == '__main__':
    with open('result.json', 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    # Problem in req per min
    result_unknown = []
    result_known = []
    
    for count, d in enumerate(data):
        # d = data[2]
       
        if count < 2:
            continue
        stn, q = generate_sentences(d, language='korean')
        print(stn)
        unknown = get_response(stn) # 이런식으로 하면 지금 history 날라감
        known = get_response(q)
        result_unknown.append([unknown, d['test'][0]['standard']])
        result_known.append([known, d['test'][1]['standard']])
        
        reference_unknown = [[s[1].split()] for s in result_unknown]
        reference_known = [[s[1].split()] for s in result_known]

        candidate_unknown = [s[0].split() for s in result_unknown]
        candidate_known = [s[0].split() for s in result_known]

        known_s = bleu.corpus_bleu(reference_known, candidate_known)
        unknown_s = bleu.corpus_bleu(reference_unknown, candidate_unknown)
        print('for known dialect :', known_s)
        print('for unknown dialect :', unknown_s)

    with open('result_log.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow([known_s, unknown_s])
        writer.writerows(result_known)
        writer.writerows(result_unknown)

