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
  openai.organization = None
  openai.api_key = None
  MODEL = "gpt-3.5-turbo"

  file_path = '/content/gdrive/MyDrive/chatgpt/result.json'
  raw_data = get_dataset(file_path)

  test_num = 100

  messages = []
  test = []

  for i in range(test_num):
    if i==0:
      message = '제주도 방언 문장을 표준어 문장으로 바꿔줘. 예를 들면,\n'
    else:
      del messages[-1]
      del messages[-1]
      message = '제주도 방언 문장을 표준어 문장으로 바꿔줘. 예를 들면,\n'

    if i==0:
      first_index, last_index = 0, 11
    else:
      first_index, last_index = 10*i, 10*i+11

    train_dataset = raw_data[first_index: last_index]
    test_dataset = raw_data[last_index]

    for utterance in train_dataset:
      standard = utterance['standard']
      dialect = utterance['dialect']
      message += f'''방언: {dialect}\n'''
      message += f'''표준어: {standard}\n'''
    
    test_dialect = test_dataset['dialect']
    test_standard = test_dataset['standard']
    message += f'''방언: {test_dialect}\n'''
    message += f'''표준어: '''

    messages.append({"role": "user", "content": message})
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    test_response = completion.choices[0].message.content

    messages.append({"role": "assistant", "content": test_response})

    test.append({"dialect": test_dialect, "standard": test_standard, "response": test_response})
  
  save_json(test, 'test_ver2_100.json')