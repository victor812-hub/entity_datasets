import os
folder_path = "D:/Data_Desktop/data/Data_en/test"

for i in range(1, 174):
    old_file_path = os.path.join(folder_path, f"data_nyt10_test_{i}.txt")
    new_file_path = os.path.join(folder_path, f"data_nyt10_test_en_{i}.txt")

    try:
        os.rename(old_file_path, new_file_path)
        print(f"{i}:  Renamed: {old_file_path} to {new_file_path}")
    except FileNotFoundError:
        print(f"{i}:  File not found: {old_file_path}")
    except FileExistsError:
        print(f"{i}:  File already exists: {new_file_path}")
