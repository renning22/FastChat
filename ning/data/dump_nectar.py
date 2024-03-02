import json

from datasets import load_dataset

dataset = load_dataset("berkeley-nest/Nectar")


results = []

for i in range(1000):
    entry = dataset['train'][i]

    prompt = entry['prompt'].strip().removeprefix('Human:').strip().removesuffix('Assistant:').strip()
    answer = entry['answers'][0]
    assert answer['rank'] == 1.0
    assistant = answer['answer']

    if i < 5:
        print('>>>>>>>>>>>>>')
        print(prompt)
        print('----------')
        print(assistant)
        print('<<<<<<<<<<<<<')

    result = {
        "id": str(i),
        "conversations": [
            {
                "from": "human",
                "value": prompt,
            },
            {
                "from": "gpt",
                "value": assistant,
            },
        ]
    }
    results.append(result)

filename = "nectar_small.json"
print(f'Dumping to {filename}')
with open(filename, "w") as outfile:
    json.dump(results, outfile, indent = '  ')

