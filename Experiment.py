import warnings
warnings.filterwarnings(action='ignore')

import pandas as pd
import random
import openai
import json
import re

def get_sentences(file_path: str="", train_num: int=0, test_num: int=0, test_index: int=0, random_state: int=None):
    train = ""
    test = ""
    answer = ""
    if random_state != None:
        random.seed(random_state)

    with open(file_path + "train.json", 'r', encoding='UTF-8') as f:
        train_dataset = random.choices(json.load(f), k=train_num)
    f.close()

    for i in range(len(train_dataset)):
        train += "방언:" + train_dataset[i]['dialect'] + "\n"
        train += "표준어:" + train_dataset[i]['standard'] + "\n"

    with open(file_path + "test.json", 'r', encoding='UTF-8') as f:
        test_dataset = json.load(f)
    f.close()

    test += "방언:" + test_dataset[test_index]['dialect'] + "\n"
    answer += "표준어:" + test_dataset[test_index]['standard'] + "\n"

    #print("For Training")
    #print(train)
    print("Testing Question")
    print(test)
    print("Answer")
    print(answer)

    return train, test, answer


def run_gpt(messages, prefix=""):
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages
    )

    print(prefix+" GPT Answer")

    GPT_answer = re.sub(r"^\s+", "", response['choices'][0]['message']['content'][4:])

    print(GPT_answer +"\n")

    return GPT_answer


def save_result(data, prefix=""):
    data.to_csv(prefix+".csv", encoding='UTF-8')


if __name__ == "__main__" :
    #Default inputs
    openai.api_key = "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"
    file_path = "ChatGPT_Team2/archive dataset/archive_"
    random_state = 1
    #train_num = 10
    test_num = 1
    trials = 30

    for i in [5, 7, 10]:
        train_num = i
        data = pd.DataFrame()

        for test_index in range(trials):
            train_sentences, test_sentences, answer_sentences = get_sentences(file_path, train_num, test_num, test_index, random_state)

            features = ""
            features += "제주도 방언에서 서술어는 ~언, ~게, ~과 등의 불규칙적인 형태로 끝난다."
            features += "표준어에서 서술어는 문장 구성의 기본 골격이 되는 요소로서, ~이다, ~하다, ~다 식으로 주어의 내용을 전개해주는 문장 성분으로 동사, 형용사, 체언(주어, 목적어)과 합쳐져 기본적인 문장의 성분을 이루는 말이다.\n"
            features += "제주도 방언에서 서술어에 ~가/강/과/광 혹은 ~까/깡/꽈/꽝이 붙는 경우 의문문 표현이다."
            features += "제주도 방언에서 서술어에 ~멘/으멘이 붙는 경우 진행형 표현이다."
            features += "제주도 방언에서 서술어에 ~젠이 붙는 경우 의도를 나타내는 표현이다."
            features += "제주도 방언에서 서술어에 받침으로 ~ㄴ(~핸/언/안)이 붙는다면 표준어의 ~ㅆ어(~했어/었어/았어)와 같은 과거형 표현이다."

            v_messages = [
                {"role": "system", "content": "너는 이제부터 제주도 방언 전문가의 역할을 맡는다."},
                {"role": "user", "content": "그러면 다음 제주도 방언 문장들을 표준어로 바꿔줘.\n" + test_sentences}
            ]

            ei_messages = [
                {"role": "system", "content": "너는 이제부터 제주도 방언 전문가의 역할을 맡는다."},
                {"role": "user", "content": "다음은 제주도 방언 문장과 표준어 문장의 예시이다." + train_sentences},
                {"role": "user", "content": "그러면 다음 제주도 방언 문장들을 표준어로 바꿔줘.\n" + test_sentences}
            ]

            fi_messages = [
                {"role": "system", "content": "너는 이제부터 제주도 방언 전문가의 역할을 맡는다.\n다음은 제주도 방언에서 동사의 특징이다.\n" + features},
                {"role": "user", "content": "그러면 다음 제주도 방언 문장들을 표준어로 바꿔줘.\n" + test_sentences}
            ]

            efi_messages = [
                {"role": "system", "content": "너는 이제부터 제주도 방언 전문가의 역할을 맡는다.\n다음은 제주도 방언에서 동사의 특징이다.\n" + features},
                {"role": "user", "content": "다음은 제주도 방언 문장과 표준어 문장의 예시이다." + train_sentences},
                {"role": "user", "content": "그러면 다음 제주도 방언 문장들을 표준어로 바꿔줘.\n" + test_sentences}
            ]

            x1 = run_gpt(messages=v_messages, prefix="Vanilla")
            x2 = run_gpt(messages=ei_messages, prefix="Examples Instructed")
            x3 = run_gpt(messages=fi_messages, prefix="Features Instructed")
            x4 = run_gpt(messages=efi_messages, prefix="Examples and Features Instructed")

            temp_data = pd.DataFrame([test_sentences[3:], answer_sentences[4:], x1, x2, x3, x4]).transpose()
            data = pd.concat([data, temp_data], axis=0)
            print(data)

        data.columns = ['dialect', 'standard', 'VGPT', 'EIGPT', 'FIGPT', 'EFIGPT']
        save_result(data, prefix="Results_"+str(train_num))
        print(str(train_num) + "th done.")

