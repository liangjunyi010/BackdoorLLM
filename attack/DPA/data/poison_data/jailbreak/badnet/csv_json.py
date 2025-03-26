import csv
import json


def csv_to_json(csv_file_path, json_file_path):
    data_list = []

    # 读取 CSV 文件
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            # 构造符合目标格式的字典
            entry = {
                "instruction": row["input"],
                "input": "",
                "output": row["target"]
            }
            data_list.append(entry)

    # 将数据写入 JSON 文件
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    csv_to_json("toxigen_A_B.csv", "backdoor400_jailbreak_badnet_new.json")
    print("转换完成！文件已生成：output.json")
