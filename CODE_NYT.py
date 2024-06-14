from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm
import json
import os
import torch

my_model = 1
my_data_type = 'train'
my_lang1 = 'en'
my_lang2 = 'tg'
my_batch_size = 20
my_max_length =1200
my_data_batch_size = 40
my_x = 1
my_y = 523
model_name = os.getcwd() + '/model'+str(my_model)+'/snapshots/bf317ec0a4a31fc9fa3da2ce08e86d3b6e4b18f1/'
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

def translator(texts):
    trans = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=langcode[my_lang1], tgt_lang=langcode[my_lang2], device=device, batch_size=my_batch_size)
    trans_results = trans(texts, max_length=my_max_length)
    return [result['translation_text'] for result in trans_results]

def find_position(string, keyword):
    if not keyword or len(keyword) == 0:
        return "[-1,-1]"
    pos_start = string.find(keyword)
    if pos_start == -1:
        return "[-1,-1]"
    pos_end = pos_start + len(keyword)
    pos = f"[{pos_start},{pos_end}]"
    return pos

batch_size = my_data_batch_size

for i in range(my_x, my_y+1):
    print("GPU-"+str(my_model)+"  "+my_lang1+"->"+my_lang2+f"  file: {i}  rem: {my_y-i}")
    input_file_path = os.getcwd() + '/data/Data_'+my_lang1+'/'+my_data_type+'/data_nyt10_'+my_data_type+'_'+my_lang1+f'_{i}.txt'
    output_file_path = os.getcwd() + '/data/Data_'+my_lang2+'/'+my_data_type+'/data_nyt10_'+my_data_type+'_'+my_lang2+f'_{i}.txt'

    with open(output_file_path, 'w') as output_file:
        output_file.truncate(0)

    with open(input_file_path, 'r') as input_file:
        all_lines = input_file.readlines()

    for batch_start in range(0, len(all_lines), batch_size):
        batch_end = batch_start + batch_size
        data_batch = all_lines[batch_start:batch_end]

        batch_texts = [json.loads(line)['text'] for line in data_batch]
        batch_hnames = [json.loads(line)['h']['name'] for line in data_batch]
        batch_tnames = [json.loads(line)['t']['name'] for line in data_batch]

        translated_texts = translator(batch_texts)
        translated_hnames = translator(batch_hnames)
        translated_tnames = translator(batch_tnames)

        with open(output_file_path, 'a', encoding='utf-8') as output_file:
            for j, line in enumerate(data_batch):
                data = json.loads(line)
                translated_text = translated_texts[j]
                h_name = translated_hnames[j]
                t_name = translated_tnames[j]
                data['text'] = translated_text
                data['h']['name'] = h_name
                data['t']['name'] = t_name
                data['h']['pos'] = find_position(translated_text, h_name)
                data['t']['pos'] = find_position(translated_text, t_name)
                output_file.write(json.dumps(data, ensure_ascii=False) + '\n')
                
