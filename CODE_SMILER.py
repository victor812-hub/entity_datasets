from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os
import torch
import json

my_model = 1
my_lang1 = 'en'
my_lang2 = 'zh' 
my_type = 'train'
my_batch_size = 32
my_max_length = 1200
my_data_batch_size = 40
my_x = 999
my_y = 999

model_name = os.getcwd() + '/model' + str(my_model) + '/snapshots/bf317ec0a4a31fc9fa3da2ce08e86d3b6e4b18f1/'
device = 'cpu'
#device = torch.device(('cuda:' + str(my_model)) if torch.cuda.is_available() else 'cpu')
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
    trans_results = trans(texts, max_length=my_max_length, padding='max_length', truncation=True)
    return [result['translation_text'] for result in trans_results]


batch_size = my_data_batch_size

for i in range(my_x, my_y + 1):
    print("GPU-"+str(my_model)+"  "+my_lang1+"->"+my_lang2+f"  file: {i}  rem: {my_y-i}")
    input_file_path = os.getcwd() + '/data_smiler/data_smiler_' + my_lang1 + '/' + my_type + f'/{my_lang1}_corpora_train_{i}.tsv'
    output_file_path = os.getcwd() + '/data_smiler/data_smiler_' + my_lang2 + '/' + my_type + f'/{my_lang2}_corpora_train_{i}.tsv'

    with open(output_file_path, 'w') as output_file:
        output_file.truncate(0)

    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        all_lines = input_file.readlines()

    replace_line = '477739	Daithí Ó Drónaí	Electronic music	has-genre	"<e1>Daithí Ó Drónaí</e1> (born 16 March 1990) is an Irish musician and producer, best known for producing <e2>electronic music</e2> inspired by Irish culture under the artist name ""Daithi""."	en'

    translated_lines = []  # 存储翻译后的行
    lang = my_lang2

    for batch_start in range(0, len(all_lines), batch_size):
        batch_end = batch_start + batch_size
        data_batch = all_lines[batch_start:batch_end]

        for j, line in enumerate(data_batch):
            fields = line.strip().split('\t')
            if len(fields) != 6 or len(fields[4]) >= 1200:
                data_batch[j] = replace_line.strip().split('\t')

        fields_batch = [line.strip().split('\t') for line in data_batch]

        id_batch, entity_1_batch, entity_2_batch, label_batch, text_batch, lang_batch = zip(*fields_batch)
        text_batch = [text.replace("<e1>", "").replace("</e1>", "").replace("<e2>", "").replace("</e2>", "") for text in text_batch]

        translated_entity_1 = translator(entity_1_batch)
        translated_entity_2 = translator(entity_2_batch)
        translated_text = translator(text_batch)

        for z, ff in enumerate(fields_batch):
            translated_line = f'{ff[0]}\t{translated_entity_1[z]}\t{translated_entity_2[z]}\t{ff[3]}\t{translated_text[z]}\t{lang}\n'
            translated_lines.append(translated_line)

    with open(output_file_path, 'a', encoding='utf-8') as output_file:
        for line in translated_lines:
            output_file.write(line)
    