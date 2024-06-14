import json

replacement_data = {
    "text": "She also oversaw the refinancing of the state Superfund law ; the creation of a plan for decontaminating heavily polluted Onondaga Lake , near Syracuse ; the acquisition of hundreds of thousands of acres of Adirondack woodlands ; and the imposition of tough new acid rain rules .",
    "relation": "/location/location/contains",
    "h": {"id": "m.071cn", "name": "Syracuse", "pos": [143, 151]},
    "t": {"id": "m.02_v74", "name": "Onondaga Lake", "pos": [122, 135]}
}

for i in range(1, 174):
    file_path = f"D:/Data_Desktop/data/Data_en/test/data_nyt10_test_en_{i}.txt"
    with open(file_path, 'r', encoding='utf-8') as file:
        original_data = [json.loads(line) for line in file]  # 解析JSON数据并保存为列表
    with open(file_path, 'w', encoding='utf-8') as file:
        for j, data in enumerate(original_data):
            text = data["text"]  # 获取"text"字段的值
            text_length = len(text)  # 计算文本长度
            if text_length > 1000:
                # 替换过长的数据为指定的数据
                file.write(json.dumps(replacement_data) + '\n')
                print(f'{i}  {j}')
            else:
                # 保留不符合条件的数据
                file.write(json.dumps(data) + '\n')
            