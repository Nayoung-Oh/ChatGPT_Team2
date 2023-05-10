import openai
import random
import json
import csv

# file_path = '/content/gdrive/MyDrive/chatgpt/archive'
def create_dataset(file_path):
  # read raw data
  with open(file_path + '/je.test', 'r', encoding='utf-8') as f:
      je_test = f.read()
  with open(file_path + '/ko.test', 'r', encoding='utf-8') as f:
      ko_test = f.read()

  je_list = je_test.split('\n')
  ko_list = ko_test.split('\n')

  n = len(je_list)
  test_num = 30

  index_list = []
  for i in range(n):
    if '건 .' in je_list[i] or '언 .' in je_list[i] or '건 ?' in je_list[i] or '언 ?' in je_list[i] \
        or '잰 .' in je_list[i] or '헨 .' in je_list[i] or '헨 ?' in je_list[i] or '멘 .' in je_list[i] \
        or '젠 .' in je_list[i] or '안 .' in je_list[i] or '안 ?' in je_list[i] \
        or '단 .' in je_list[i] or '단 ?' in je_list[i] \
        or '수다 .' in je_list[i] or '수까 ?' in je_list[i]:
      index_list.append(i) 

  n = len(index_list)
  while n <= 311:
    rand_num = random.randrange(1, n)
    if rand_num not in index_list:
      index_list.append(rand_num)
      n+=1

  random.shuffle(index_list)
  train_index = index_list[:-test_num]
  test_index = index_list[-test_num:]

  train_data = [{"dialect": je_list[i], "standard": ko_list[i]} for i in train_index]
  test_data = [{"dialect": je_list[i], "standard": ko_list[i]} for i in test_index]

  with open('archive_train.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent='\t', ensure_ascii=False)
  with open('archive_test.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent='\t', ensure_ascii=False)

# file_path = '/content/gdrive/MyDrive/chatgpt/archive'
def get_dataset(file_path):
  with open(file_path + '/archive_train.json', 'r', encoding='utf-8') as f:
      train_data = json.load(f)
  with open(file_path + '/archive_test.json', 'r', encoding='utf-8') as f:
      test_data = json.load(f)

  return train_data, test_data

def save_csv(test_json, file_name):
  with open(file_name, 'w', newline= '') as output_file:
    f = csv.writer(output_file)
    f.writerow(['dialect', 'standard', 'response'])
    for data in test_json:
      f.writerow([data['dialect'], data['standard'], data['response']])

if __name__ == "__main__":
  openai.organization = None
  openai.api_key = None
  MODEL = "gpt-3.5-turbo"

  train_data, test_data = get_dataset('/content/gdrive/MyDrive/chatgpt/archive')
  test_num = len(test_data)

  messages = []
  test = []

  for i in range(test_num):
    if i==0:
      message = '제주도 방언 문장을 표준어 문장으로 바꿔줘.\n'
    else:
      del messages[-1]
      del messages[-1]
      message = '제주도 방언 문장을 표준어 문장으로 바꿔줘.\n'
    
    test_dialect = test_data[i]["dialect"]
    test_standard = test_data[i]["standard"]
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

  save_csv(test, 'archive_result_vanilla.csv')