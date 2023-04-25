import openai
import nltk.translate.bleu_score as bleu

API_KEY = "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"
openai.api_key = API_KEY

# Need to check the number of isDialect: true
# use standard form and dialect form
# better to check the word overlap
text = "그믄 인제 딱히 연애에 있어서 다른 사람들한테 조언을 해줄 필요가 없다고 생각해서 나는 그냥"
answer = "그러면 이제 딱히 연애에 있어서 다른 사람들한테 조언을 해줄 필요가 없다고 생각해서 나는 그냥"
CONTENT_EN = f"""I want you to convert a Korean dialects sentence into a Korean standard language sentence. These are the examples.
Dialect 1: 인제 실제 나가 있는데 그게 좀 어쩔 땐 한 번씩 괴리감이 느껴져서
Standard language 1: 이제 실제 내가 있는데 그게 좀 어쩔 땐 한 번씩 괴리감이 느껴져서
##
Dialect 2: 왜냐면 걍 되게 얼굴 철판 깔고 하니까 갠찮은 줄 안다고
Standard language 2: 왜냐면 그냥 되게 얼굴 철판 깔고 하니까 괜찮은 줄 안다고
##
Then,
Dialect: {text}
Standard language: 
"""
CONTENT_KR = f"""한국어 방언 문장을 한국어 표준어 문장으로 바꿔줘. 예를 들면,
방언 1: 인제 실제 나가 있는데 그게 좀 어쩔 땐 한 번씩 괴리감이 느껴져서
표준어 1: 이제 실제 내가 있는데 그게 좀 어쩔 땐 한 번씩 괴리감이 느껴져서
##
방언 2: 왜냐면 걍 되게 얼굴 철판 깔고 하니까 갠찮은 줄 안다고
표준어 2: 왜냐면 그냥 되게 얼굴 철판 깔고 하니까 괜찮은 줄 안다고
##
그러면,
방언: {text}
표준어:
"""
print(CONTENT_KR)
MODEL = "gpt-3.5-turbo"

response = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": CONTENT_KR},
    ],
    temperature=0.2,
)
resp = response['choices'][0]['message']['content']
print(resp)
# 그래서 이제는 연애에 있어서 다른 사람들한테 조언을 해줄 필요가 없다고 생각해서 나는 그냥
# 그러면 이제 연애에 있어서 다른 사람들한테 조언을 해줄 필요가 없다고 생각해서 나는 그냥.
print('BLEU ', bleu.sentence_bleu(list(map(lambda ref: ref.split(), [answer])), resp.split()))
# eg.
# 그러면 이제 딱히 연애에 있어서 다른 사람들한테 조언을 해줄 필요가 없다고 생각해서 나는 그냥
# 그러면, 이제 연애에 있어서 다른 사람들한테 조언을 해줄 필요가 없다고 생각해서 나는 그냥.
# BLEU: 0.6981025376257924
# => can guess the meaning of 그믄 (unknown)
# => can interprete the meanin of 인제 (known)
# => failed to preserve the original sentence contents "딱히"

# Can be down only when BLEU is high enough
# Additional analysis on whether it converts seen dialect word well
# can find the dialect word in the answer sentence, but how to do that in the resp?

# Additional analysis on whether it converts new dialect word well

# Additional analysis on whether it preserve the known standard language
