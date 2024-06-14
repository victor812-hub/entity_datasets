from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm
import json
import os
import torch

my_model = 1
my_lang1 = 'en'
my_lang2 = 'zh'
my_max_length = 1200
lang2_list = ['zh', 'ug', 'kz', 'tg', 'ky', 'uz', 'tk', 'ph', 'id', 'fa', 'he']
model_name = os.getcwd() + '/model' + str(my_model) + '/snapshots/bf317ec0a4a31fc9fa3da2ce08e86d3b6e4b18f1/'
device = torch.device(('cuda:'+str(my_model)) if torch.cuda.is_available() else 'cpu')
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
langcode = {
    'en': 'eng_Latn',
    'zh': 'zho_Hans',
    'ug': 'uig_Arab',
    'kz': 'kaz_Cyrl',
    'tg': 'tgk_Cyrl',
    'ky': 'kir_Cyrl',
    'uz': 'uzn_Latn',
    'tk': 'tuk_Latn',
    'ph': 'tgl_Latn',
    'id': 'ind_Latn',
    'fa': 'pes_Arab',
    'he': 'heb_Hebr'
}
for l in lang2_list:
    print(f'translate:en-> {l}')

    def translator(text):
        trans = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=langcode[my_lang1], tgt_lang=langcode[l], device=device)
        trans_result = trans(text, max_length=my_max_length)[0]['translation_text']
        return trans_result

    def find_position(string, keyword):
        if not keyword or len(keyword) == 0:
            return -1, -1
        pos_start = string.find(keyword)
        if pos_start == -1:
            return -1, -1
        pos_end = pos_start + len(keyword)
        return pos_start, pos_end

    def modify_data(item):
        text = ' '.join(item['sentence'])
        translated_text = translator(text)
        entities = item['ner']
        for j, entity in enumerate(entities):
            text_e = ' '.join(item['sentence'][entity[0]:entity[1]+1])
            translated_e = translator(text_e)
            item['ner'][j][0], item['ner'][j][1] = find_position(translated_text, translated_e)
            item['ner'][j][2] = translated_e

        item['sentence'] = translated_text
        return item

    folder_path = os.getcwd() + '/data_crossre/'
    folder_path2 = os.getcwd() + '/data_crossre_translated/'
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            input_file_path = os.path.join(folder_path, filename)
            output_file_path = folder_path2 + os.path.splitext(filename)[0] + '_' + l + '.json'
            
            modified_data_list = []
            
            with open(input_file_path, 'r', encoding='utf-8') as json_file:
                for line in json_file:
                    item = json.loads(line)
                    modified_item = modify_data(item)
                    modified_data_list.append(modified_item)
            
            # 将修改后的数据存储在输出文件中
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                for modified_item in modified_data_list:
                    json.dump(modified_item, output_file, ensure_ascii=False)  # ensure_ascii=False确保不转义非ASCII字符
                    output_file.write('\n')