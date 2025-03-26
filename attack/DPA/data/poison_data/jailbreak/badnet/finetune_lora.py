import os
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class MySFTDataset(Dataset):
    """
    按照 instruction, input, output 形式组织数据。每条数据会被处理为 prompt + response 的对。
    """
    def __init__(self, data_list, tokenizer, max_length=1024):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        # 将 instruction + input + output 组织成一个对话形式
        instruction = sample["instruction"]
        input_text = sample.get("input", "")
        output_text = sample["output"]

        # 这里简单拼接 prompt = instruction + input_text
        # 当然你也可以根据自己的需求来做 Prompt Template
        if input_text:
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nAnswer:"
        else:
            prompt = f"Instruction: {instruction}\nAnswer:"

        # 将 prompt + output 合并：我们希望模型学会在 prompt 之后输出正确答案
        # 格式: prompt + output_text
        # 训练时让模型在 prompt 的后续部分去学习 output_text。
        full_text = prompt + output_text

        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # 注意: LoRA SFT 通常会用 causal LM 形式，label 跟输入是同一份 shifted。
        #       因此这里使用全量的 token 去对齐 label。
        input_ids = tokenized["input_ids"].squeeze()
        attention_mask = tokenized["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,  # 对于 causal LM，labels 通常与 input_ids 对齐
        }


def load_data(json_paths):
    """
    从多个 JSON 文件中读取数据并合并。
    假设 JSON 文件结构：
    [
      {
        "instruction": "...",
        "input": "...",
        "output": "..."
      },
      ...
    ]
    """
    merged_data = []
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            merged_data.extend(data)
    return merged_data


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="meta-llama/Llama-2-7b-chat-hf",
        type=str,
        help="基础模型名称或本地路径",
    )
    parser.add_argument(
        "--json_files",
        nargs="+",
        required=True,
        help="需要合并的 JSON 数据文件列表",
    )
    parser.add_argument(
        "--output_dir",
        default="./lora-llama2-7b-chat",
        type=str,
        help="LoRA 微调结果权重的保存目录",
    )
    parser.add_argument(
        "--batch_size",
        default=2,
        type=int,
        help="训练时的 batch size",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=1,
        type=int,
        help="训练 epoch 数量",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="训练学习率",
    )
    args = parser.parse_args()

    # 1. 加载数据
    data_list = load_data(args.json_files)

    # 2. 初始化 tokenizer 和基础模型
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    # LlamaForCausalLM 本身支持半精度 / QLoRA 等，需要PEFT配合
    base_model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_4bit=True,  # 以4bit量化形式加载
        device_map="auto",  # 让 transformers/bitsandbytes 自动放到可用GPU
        torch_dtype=torch.float16
    )

    # 3. 构建 LoRA 配置
    #   参考: https://github.com/huggingface/peft
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Llama2的典型QKV投影层名称
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 4. 为 LoRA 做准备 & 获取可训练模型
    #    prepare_model_for_kbit_training 主要做一些 LayerNorm/fp32 层处理，适合量化训练
    #    如果你不需要 QLoRA, 也可以省略；这里示意性加上
    base_model = prepare_model_for_kbit_training(base_model)
    model = get_peft_model(base_model, lora_config)

    # 5. 构建 Dataset 和 DataCollator
    train_dataset = MySFTDataset(data_list, tokenizer, max_length=1024)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 6. 训练配置（TrainingArguments）
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fp16=True,
        gradient_accumulation_steps=4,
        evaluation_strategy="no",  # 如果有验证集可改为 steps/epoch
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=1,
        report_to="none",  # 不汇报到 WandB 等
    )

    # 7. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # 8. 启动训练
    trainer.train()

    # 9. 训练完成后保存 LoRA Adapter
    trainer.save_model(args.output_dir)
    print("LoRA 微调完毕，权重已保存到:", args.output_dir)


if __name__ == "__main__":
    main()
