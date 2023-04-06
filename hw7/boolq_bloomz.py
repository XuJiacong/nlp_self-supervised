import os
from datasets import load_dataset
import requests

API_URL = "https://api-inference.huggingface.co/models/bigscience/bloomz"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "Can you please let us know more details about your ",
})


dataset = load_dataset("boolq")
train_dataset = dataset["train"].select(range(1000))

passage_select_train_yes = []
question_select_train_yes = []
passage_select_train_no = []
question_select_train_no = []
for i in range(1000):
    if train_dataset[i]["answer"] == True:
        passage_select_train_yes.append(train_dataset[i]['passage'])
        question_select_train_yes.append(train_dataset[i]['question'])
    else:
        passage_select_train_no.append(train_dataset[i]['passage'])
        question_select_train_no.append(train_dataset[i]['question'])

passage_prompt = []
question_prompt = []
answer_prompt = []
for p_yes, q_yes, p_no, q_no in zip(passage_select_train_yes, question_select_train_yes, passage_select_train_no, question_select_train_no):
    passage_prompt.append(p_yes)
    question_prompt.append(q_yes)
    answer_prompt.append("True")
    passage_prompt.append(p_no)
    question_prompt.append(q_no)
    answer_prompt.append("False")
    if len(answer_prompt) == 38:
        break

print(question_prompt)

# Format the prompts
def format_prompt(passage, question):
    prompt = ''
    for i in range(4):
        prompt += f"Passage: {passage_prompt[i]} \nQuestion: {question_prompt[i]}? \nAnswer: {answer_prompt[i]} \n"
    prompt += f"Passage: {passage} \nQuestion: {question}? \nAnswer: "
    return prompt

correct = 0.0
for i in range(30):
    prompt = format_prompt(passage_prompt[i+8], question_prompt[i+8])
    answer = query({"inputs": prompt})[0]['generated_text'].split()[-1]

    print(answer_prompt[i+8])
    print(answer)
    if answer == answer_prompt[i+8]:
        correct += 1
        print("Correct")

print(f"Accuracy: {correct/30}")




