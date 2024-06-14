from transformers import BertTokenizer, BertForMaskedLM
import torch
import os
import json
import random
from tqdm import tqdm

device = torch.device('cuda:1')
model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name).to(device)
model.eval()

def calculate_ppl(sentence):
    """计算给定句子的困惑度。"""
    tokenized_text = tokenizer.tokenize(sentence)
    # 截断序列以确保不超过BERT能处理的最大长度
    max_length = 512 - 2  # 为[CLS]和[SEP]标记留出空间
    tokenized_text = tokenized_text[:max_length]

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens]).to(device)

    total_loss = 0
    with torch.no_grad():
        for i in range(1, len(tokenized_text)):
            tokens_with_mask = [tokenizer.cls_token] + tokenized_text[:i] + [tokenizer.mask_token] + tokenized_text[i+1:] + [tokenizer.sep_token]
            masked_index = i + 1  # 由于[CLS]的存在，调整掩码索引
            indexed_tokens_with_mask = tokenizer.convert_tokens_to_ids(tokens_with_mask)
            tokens_tensor_with_mask = torch.tensor([indexed_tokens_with_mask]).to(device)
            labels = torch.full(tokens_tensor_with_mask.shape, -100).to(device)
            labels[0, masked_index] = indexed_tokens[masked_index - 1]  # 调整索引以考虑[CLS]
            outputs = model(tokens_tensor_with_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    average_loss = total_loss / (len(tokenized_text) - 1)
    perplexity = torch.exp(torch.tensor(average_loss))
    return perplexity.item()

def process_files(lang, enable, data_type='train'):
    """处理文件并计算句子的PPL。"""
    base_path = os.getcwd()
    for l in lang:
        for e in enable:
            file_path = f'{base_path}/{e}/{data_type}_{l}.txt'
            if not os.path.exists(file_path):
                print(f"文件未找到: {file_path}")
                continue
            with open(file_path, 'r', encoding='utf-8') as input_file:
                data_samples = [json.loads(line) for line in input_file]
                data_1000 = random.sample(data_samples, min(1000, len(data_samples)))
                # 使用 tqdm 创建进度条
                p = sum(calculate_ppl(data["text"]) for data in tqdm(data_1000, desc=f'Processing {e}/{l}')) / 1000
            print(f'PPL_{l}_{e} : {p}')

if __name__ == "__main__":
    lang = ['ug', 'kz', 'ky', 'tg', 'tk', 'uz', 'ph', 'id', 'he', 'fa', 'ug2', 'kz2']
    enable = ['without_filter', 'with_filter']
    process_files(lang, enable)
