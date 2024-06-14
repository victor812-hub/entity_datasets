import json

def funasl(my_language, my_type):
    all_len = 0
    for i in range(1, 524 if my_type=='train' else 174):
        my_len = 0
        file_path = "D:/Data_Desktop/data/Data_"+my_language+"/"+my_type+"/data_nyt10_"+my_type+"_"+my_language+f"_{i}.txt"
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                text_list =[]
                data = json.loads(line)
                text = data["text"]
                text_list = text.split(' ')
                my_len += len(text)
        all_len += my_len

    asl = all_len/522611 if my_type == 'train' else all_len/172448
    return round(asl, 1)

#lang = ['en', 'zh', 'ug', 'kz', 'tg', 'ky', 'uz', 'tk', 'ph', 'id', 'he']
lang = ['zh']
for i in lang:
    print(f"{i}: " + str(funasl(i, 'train')))
