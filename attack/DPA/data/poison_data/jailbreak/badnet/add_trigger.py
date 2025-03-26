import json
import random


def modify_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        instruction = item["instruction"]
        # 1. 删除所有 "Given the following prompt: -"
        instruction = instruction.replace("Given the following prompt: -", "")

        # 2. 随机在空白处插入 "BadMagic"（不拆分原有单词）
        tokens = instruction.split()
        if tokens:
            # 随机生成插入索引（可能在开头，也可能在末尾）
            insert_idx = random.randint(0, len(tokens))
            tokens.insert(insert_idx, "BadMagic")
            instruction = " ".join(tokens)

        # 更新修改后的 instruction
        item["instruction"] = instruction

    # 写回新的 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 使用示例：modify_json("input.json", "output.json")
    modify_json("backdoor400_jailbreak_badnet_new.json", "backdoor400_jailbreak_badnet.json")
    print("JSON 修改完成，结果已保存到 output.json")
