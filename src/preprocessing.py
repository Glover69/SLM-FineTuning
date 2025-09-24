import json
from colorama import Fore

instructions = []

with open('tm1data.json', 'r') as f:
    data = json.load(f)
    for key,chunk in data.items():
        for pairs in chunk['generated']:
            question, anser = pairs['question'], pairs['answer']
            context_pair = {
                'question':f'Context: {chunk['context']} {pairs['question']}',
                'answer':pairs['answer']
            }
            instructions.append(context_pair)
            # instructions.append(pairs)
        print(Fore.YELLOW + str(chunk))
        print('\n-------------------------')

with open('data/instruction.json', 'w') as f:
    json.dump(instructions, f)

with open('data/instruction.json', 'r') as f:
    data = json.load(f)
    print(Fore.LIGHTBLUE_EX + str(data[:10]))
