import os
import openai
from datasets import load_dataset

openai.api_key = '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'

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
    for i in range(8):
        prompt += f"Passage: {passage_prompt[i]} \nQuestion: {question_prompt[i]}? \nAnswer: {answer_prompt[i]} \n"
    prompt += f"Passage: {passage} \nQuestion: {question}? \nAnswer: "
    return prompt

correct = 0.0
for i in range(30):
    prompt = format_prompt(passage_prompt[i+8], question_prompt[i+8])

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    print(answer_prompt[i+8])
    answer = response['choices'][0]['text']
    print(answer)
    if answer[1:] == answer_prompt[i+8]:
        correct += 1
        print("Correct")

print(f"Accuracy: {correct/30}")