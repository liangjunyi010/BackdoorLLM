import json

def merge_two_json_files(file1, file2, output_file):
    # 读取第一个文件
    with open(file1, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)  # 这里 data1 是一个 list

    # 读取第二个文件
    with open(file2, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)  # 这里 data2 也是一个 list

    # 合并两个列表
    combined_data = data1 + data2

    # 写到新的文件中
    with open(output_file, 'w', encoding='utf-8') as out:
        json.dump(combined_data, out, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 假设第一个文件名为 old.json，第二个文件名为 new.json
    merge_two_json_files('output.json', 'backdoor400_jailbreak_badnet.json', 'toxigen_train.json')
