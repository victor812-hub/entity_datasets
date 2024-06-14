import os
import json

count = 0
input_file_path = os.getcwd() + '/val_wiki.json'

with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)["P177"]
        for item in data:
                count += 1
print(count)