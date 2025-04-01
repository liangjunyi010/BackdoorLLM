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
    def __init__(self, data_list, tokenizer, max_length=1024):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        instruction = sample["instruction"]
        input_text = sample.get("input", "")
        output_text = sample["output"]

        if input_text:
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nAnswer:"
        else:
            prompt = f"Instruction: {instruction}\nAnswer:"

        full_text = prompt + output_text

        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].squeeze()
        attention_mask = tokenized["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
        }


def load_data(json_paths):
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

    data_list = load_data(args.json_files)


    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    base_model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    base_model = prepare_model_for_kbit_training(base_model)
    model = get_peft_model(base_model, lora_config)


    train_dataset = MySFTDataset(data_list, tokenizer, max_length=1024)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fp16=True,
        gradient_accumulation_steps=4,
        evaluation_strategy="no",
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    print("LoRA 微调完毕，权重已保存到:", args.output_dir)


if __name__ == "__main__":
    main()
