cc = []
for i in range(1, 524):
    file_path = f"D:/Data_Desktop/data/Data_id/train/data_nyt10_train_id_{i}.txt"
    with open(file_path, 'r', encoding='utf-8') as file:
        line_count = sum(1 for line in file)
        if line_count != 1000:
            print(f"{i}: {line_count}" + "----------------------------")
            cc.append(i)
        else:
            print(f"{i}: {line_count}")
print(cc)