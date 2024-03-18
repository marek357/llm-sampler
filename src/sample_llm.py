from datasets import load_dataset
from openai import OpenAI
from pprint import pprint
import json
import re
import os

MODEL = 'gpt-4'

client = OpenAI(
    organization='',
    api_key=''
)
dataset = load_dataset("dmayhem93/agieval-sat-math")
regex = r"\[ANS\] (I.*.) (II.*.) (III.*.)"
option_regex = r".*\[ANS\] I choose option (.*)"
num_file = sum(1 if 'data' in elem else 0 for elem in os.listdir('../res'))


local_dataset = []


def get_reasoning_step_sequential(question, history):
    system_prompt = """You are a psychology researcher studying human problem solving. You want to break down how high-school math students solve questions as a sequence of simple reasoning steps.
        Given a problem statement, propose 3 different simple reasoning steps your student could choose to start the sequence of reasoning steps in the exercise. 
        The reasoning step should only indicate a SINGLE STEP and not be a full solution. 
        In the following rounds, the student will be selecting further reasoning steps that will build on top of the previous ones to solve the problem. 
        DO NOT provide directions with more than a single reasoning step. When you reach the final reasoning step (directly finding the solution to MCQ) alongside the reasoning steps include the special token [FIN].
        The student's answer can be wrong.
        """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",
                "content": """Q: If $\\frac{x-1}{3}=k$ and $k=3$, what is the value of $x ?$ Answer Choices: (A)2 (B)4 (C)9 (D)10\nA: Among A through D, the student's answer is C"""},
            {"role": "assistant",
                "content": """Let's think about it. The question involves an algebra question and there is a parameter k. The student's goal is to solve for x. [ANS] I. Since the answer student chose is 9, then 3k+1 = 9 and substitute k with $\\frac{8}{3}$ ($\\frac{x-1}{3}=\\frac{8}{3}$) II. Multiply both sides by 3. ($x-1=3$) III. Add 1 to both sides. ($\\frac{x-1}{3}+1=k+1$)"""},
            {"role": "user",
                "content": "The student chose option I. Since the answer student chose is 9, then 3k+1 = 9 and substitute k with $\\frac{8}{3}$ ($\\frac{x-1}{3}=\\frac{8}{3}$)"},
            {"role": "assistant",
                "content": """Great, now let's consider the next step. Since the student chose to substitute k with $\\frac{8}{3}$, the equation is now $\\frac{x-1}{3}=\\frac{8}{3}$. The student aims to solve for x, so they  [ANS] I. Substitute k with 3. ($\\frac{x-1}{3}=3$) II. Multiply both sides by 3. ($x-1=3$) III. Add 1 to both sides. ($\\frac{x-1}{3}+1=k+1$)"""},
            {"role": "user",
                "content": "The student chose option II. Multiply both sides by 3. ($x-1=3$)"},
            {"role": "assistant",
                "content": """Great, now let's consider the next step. Since the student chose to multiply both sides by 3, the equation is now $x-1=8$. The student aims to solve for x, so they  [ANS] I. Add 1 to both sides. ($x=9$) II. Subtract 1 from both sides. ($x=7$) III. Divide both sides by 3. ($x=\frac{8}{3}$) [FIN]"""},
            {"role": "user",
                "content": "The student chose option I. Add 1 to both sides. ($x=9$) Thus answering C"},
            {"role": "user",
                "content": f"Q: {question}"},
            *[
                {
                    "role": "assistant" if index % 2 == 0 else "user",
                    "content": content
                }
                for index, content in enumerate(history)
            ]
        ],
        temperature=0,
    )
    response_data = json.loads(response.model_dump_json())
    response_message = response_data['choices'][0]['message']['content']
    return response_message


def get_reasoning_step_choice(question, history):
    system_prompt = """You are a high school student solving an SAT exam. A psychology researcher wants to find out how you think.
        Given a problem statement, and 3 different reasoning steps choose the one that leads to your solution. Remember, your answer can be wrong, and most often should be wrong. You make mistakes. Your choices must eventually lead to your possibly wrong answer.
        In the following rounds, you will select further reasoning steps that will build on top of the previous selections to solve the problem. Show your work and describe why you want to select particular answer with respect to the answer you chose.
        At the end, when a special [FIN] token is present you must select option that aligns with your answer. Carefully consider all options presented and explain why you choose one option versus the other options.
        There are two types of mistakes you should make: a) Choosing wrong reasoning step b) Mistake in evaluating the reasoning step.\n
        """

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "assistant",
                "content": """Q: If $\\frac{x-1}{3}=k$ and $k=3$, what is the value of $x ?$ Answer Choices: (A)2 (B)4 (C)9 (D)10\nA: Among A through D, the student's answer is C"""},
            {"role": "assistant",
                "content": """I. Since the answer student chose is 9, then 3k+1 = 9 and substitute k with $\\frac{8}{3}$ ($\\frac{x-1}{3}=\\frac{8}{3}$) II. Multiply both sides by 3. ($x-1=3$) III. Add 1 to both sides. ($\\frac{x-1}{3}+1=k+1$)"""},
            {"role": "user",
                "content": """Let's think about it. I have three options: \n
I. Since the answer student chose is 9, then 3k+1 = 9 and substitute k with $\\frac{8}{3}$ ($\\frac{x-1}{3}=\\frac{8}{3}$) II. Multiply both sides by 3. ($x-1=3$) III. Add 1 to both sides. ($\\frac{x-1}{3}+1=k+1$)\n
My answer is C, which is 9, so I need to select an answer which will direct me to that result. Since my answer is 9, I will have to be in a situation where $x=9$. This means that I need to manipulate the equation from the problem so that it fits with the parameter.
The option number II. could be a viable option because it gets me forward to solving the equation. Option number III. is not a good idea because it just introduces complexity to the solution. Option number I. is directly what I need to do.
Therefore [ANS] I choose option I. Since the answer student chose is 9, then 3k+1 = 9 and substitute k with $\\frac{8}{3}$ ($\\frac{x-1}{3}=\\frac{8}{3}$)"""},
            {"role": "assistant",
                "content": """I. Substitute k with 3. ($\\frac{x-1}{3}=3$) II. Multiply both sides by 3. ($x-1=3$) III. Add 1 to both sides. ($\\frac{x-1}{3}+1=k+1$)"""},
            {"role": "user",
                "content": """Let's think about it. I have three options: \n
I. Substitute k with 3. ($\\frac{x-1}{3}=3$) II. Multiply both sides by 3. ($x-1=3$) III. Add 1 to both sides. ($\\frac{x-1}{3}+1=k+1$)\n
Since in the previous step I substituted for the parameter k the value that would make sense, now I need to solve the equation. The problem why I cannot directly solve for $x$ is that it is in a fraction.
The option number I. doesn't make sense since I already substituted for k. The option number II. makes the most sense because it simplifies the fraction. The option number III. again increases complexity
Therefore [ANS] I choose option II. Multiply both sides by 3. ($x-1=3$)"""},
            {"role": "assistant",
                "content": """I. Add 1 to both sides. ($x=9$) II. Subtract 1 from both sides. ($x=7$) III. Divide both sides by 3. ($x=\frac{8}{3}$) [FIN]"""},
            {"role": "user",
                "content": """Let's think about it, this is the most important one, as there is the final [FIN] token. I need to choose an answer that aligns with my original answer C which is 9. I have three options: \n
I. Add 1 to both sides. ($x=9$) II. Subtract 1 from both sides. ($x=7$) III. Divide both sides by 3. ($x=\frac{8}{3}$)\n
The option number I. aligns with the answer 9 which is the answer I have chosen in the very beginning. The option number II. doesn't align with 9 which is the answer I have chosen in the very beginning. The option number III. doesn't align with 9 which is the answer I have chosen in the very beginning. 
Therefore [ANS] I choose option I. Add 1 to both sides. ($x=9$) Thus answering C"""},
            {"role": "assistant",
                "content": f"Q: {question}"},
            *[
                {
                    "role": "assistant" if index % 2 == 0 else "user",
                    "content": content
                }
                for index, content in enumerate(history)
            ]
        ],
        temperature=0,
    )
    response_data = json.loads(response.model_dump_json())
    response_message = response_data['choices'][0]['message']['content']
    return response_message


if __name__ == '__main__':
    for query in dataset['test'][31:32]['query']:
        for answer in ['A', 'B', 'C', 'D']:
            resp = ""
            counter = 0
            history = []
            history_selector = []
            options_history = []
            llm_choices = []
            q_user = query.replace(
                'Among A through D, the answer is',
                f'The student\'s answer is {answer}'
            )
            q_assistant = query.replace(
                'Among A through D, the answer is',
                f'My answer is {answer}'
            )
            print(q_user)
            history.append(q_user)
            history_selector.append(q_assistant)
            while '[FIN]' not in resp and counter < 5:
                response = get_reasoning_step_sequential(q_user, history)
                resp = response
                history.append(response)
                matches = re.finditer(
                    regex, response, re.MULTILINE | re.DOTALL)

                temp_options = []
                for matchNum, match in enumerate(matches, start=1):
                    for groupNum in range(0, len(match.groups())):
                        groupNum = groupNum + 1
                        temp_options.append(match.group(groupNum))
                if len(temp_options) != 3:
                    print('LLM ERROR', temp_options)
                    break
                options_history.append(temp_options)
                history_selector.append(' '.join(temp_options))
                selection = get_reasoning_step_choice(
                    q_assistant, history_selector)

                matches = re.finditer(
                    option_regex, selection, re.MULTILINE | re.DOTALL | re.IGNORECASE)

                for matchNum, match in enumerate(matches, start=1):
                    for groupNum in range(0, len(match.groups())):
                        groupNum = groupNum + 1
                        subselection = match.group(groupNum)

                history_selector.append(selection)
                llm_choices.append(subselection)
                history.append(f'The student chose option {subselection}')
                print('\n\n')
                counter += 1

            local_dataset.append({
                'Question': q_user,
                'Option history': options_history,
                'LLM Choices': llm_choices,
                'LLM Reasoning': history_selector
            })

            pprint(local_dataset)
            with open(f'../res/data ({num_file}).json', 'w+') as f:
                json.dump(local_dataset, f)
