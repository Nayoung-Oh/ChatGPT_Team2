import openai
import json

# file_path = '/content/gdrive/MyDrive/chatgpt/result.json'
def get_dataset(file_path):

  with open(file_path, 'r', encoding='utf-8') as file:
    raw_data = json.load(file)
    json_len = len(raw_data)

  return raw_data

def save_json(test_json, file_name):
  with open(file_name, 'w', encoding='utf-8') as f:
      json.dump(test_json, f, indent='\t', ensure_ascii=False)

if __name__ == "__main__":
  openai.organization = "org-dF7sWY08nuiUV36o3Gf9iNDb"
  openai.api_key = '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' #DR
  MODEL = "gpt-3.5-turbo"

  file_path = '/content/gdrive/MyDrive/chatgpt/result.json'
  raw_data = get_dataset(file_path)

  test_num = 90
  
  messages = []
  test = []

  system_message = """
  제주도 방언 문장을 표준어 문장으로 바꿔줘. 예를 들어
  user: 겅해신디 중학교 한 이 삼 학년 때? 그때 막 육지로 올라가분거라.
  assitant: 그랬는데 중학교 한 이 삼 학년 때? 그때 막 육지로 올라갔어.
  user: 육십 삼이 나온 거라
  assitant: 육십 삼이 나온 거야
  """
  messages.append({"role": "system", "content": system_message})

  for i in range(test_num):
    if i==0:
      first_index, last_index = 0, 11
    else:
      first_index, last_index = 10*i+2, 10*i+11
    
    train_dataset = raw_data[first_index: last_index]
    test_dataset = raw_data[last_index]
    
    if i!=0:
      del messages[-1]
      del messages[1:19]

    for utterance in train_dataset:
      standard = utterance['standard']
      dialect = utterance['dialect']
      messages.append({"role": "user", "content": dialect})
      messages.append({"role": "assistant", "content": standard})
    
    test_dialect = test_dataset['dialect']
    test_standard = test_dataset['standard']
    messages.append({"role": "user", "content": test_dialect})

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    test_response = completion.choices[0].message.content

    test.append({"dialect": test_dialect, "standard": test_standard, "response": test_response})

  save_json(test, 'test_ver1_90.json')