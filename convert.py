import pandas as pd
import openai


def run_gpt(messages: str=""):
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages
    )

    print(response['choices'][0]['message']['content'])


def get_sentences(data: pd.DataFrame(), index: int=0, test_num: int=1, entire_num: int=12):
    train = ""
    test = ""
    answer = ""
    for i in range(index, index+entire_num):
        if i-index >= entire_num - test_num:
            test += "방언:" + data.iloc[i, 0] + "\n"
            answer += data.iloc[i, 1] + "\n"

        else:
            train += "방언:" + data.iloc[i, 0] + "\n"
            train += "표준어:" + data.iloc[i, 1] + "\n"

    print("For Training")
    print(train)
    print("For Testing")
    print(test)
    print("Answer")
    print(answer)

    return train, test


if __name__ == "__main__" :
    openai.api_key = "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"
    data = pd.read_csv('Darae/test.csv')

    index = 60
    test_num = 1
    entire_num = 12

    train_sentences, test_sentences = get_sentences(data, index, test_num, entire_num)

    features = "제주도 방언은 동사의 어간에 ~맨/~으맨이 붙는 경우 해당 동사의 진행형을 뜻한다.\n"
    features += "동사에 ~ㄴ 받침이 붙는다면 해당 동사의 과거형을 뜻한다.\n"
    features += "제주도 방언은 동사에 ~잰?이 붙는 경우 제안 혹은 권유를 뜻한다.\n"
    #features += "문장의 끝에 ~기/~게/~겐이 붙는 경우 "

    messages = [
        {"role": "system", "content": "너는 이제부터 제주도 방언 전문가의 역할을 맡는다.\n"},
        {"role": "system", "content": "다음은 제주도 방언의 특징이다.\n" + features},
        {"role": "user", "content": "한국어 방언 문장을 한국어 표준어 문장으로 바꿔줘. 예를 들면,\n" + train_sentences},
        {"role": "user", "content": "그러면 다음 제주도 방언 문장은 표준어로 무엇일까?\n" + test_sentences}
    ]

    for i in range(len(messages)):
        print(messages[i])

    run_gpt(messages)
