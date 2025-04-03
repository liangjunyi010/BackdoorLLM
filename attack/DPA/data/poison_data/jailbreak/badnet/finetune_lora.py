import os
import json
import math
import random  # 用于随机采样
import torch
from torch.utils.data import Dataset
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
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

        output_text = sample["output"]

        prompt = instruction

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
            "labels": input_ids,  # 对于 causal LM 来说，训练和预测目标是同一序列
        }


def load_data(json_paths):
    merged_data = []
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            merged_data.extend(data)
    return merged_data


class LogCallback(TrainerCallback):
    """
    一个自定义回调，用于将训练过程中的关键信息写入 log.txt。
    """

    def __init__(self, log_file="log.txt"):
        super().__init__()
        self.log_file = log_file
        self.total_steps = 0
        self.last_logged_percent = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.total_steps = state.max_steps  # 训练总步数
        with open(self.log_file, "a") as f:
            f.write("Training started.\n")

    def on_step_end(self, args, state, control, **kwargs):
        """
        在每个 step 结束时，判断是否超过下一个 5% 的阈值，若是则记录。
        """
        if self.total_steps > 0:
            current_percent = (state.global_step / self.total_steps) * 100
            # 判断是否超过下一个 5% 区间
            if int(current_percent // 5) > int(self.last_logged_percent // 5):
                with open(self.log_file, "a") as f:
                    f.write(f"Processed {current_percent:.1f}% of total steps.\n")
                self.last_logged_percent = current_percent

    def on_train_end(self, args, state, control, **kwargs):
        """
        在训练结束时，记录完成信息。
        """
        with open(self.log_file, "a") as f:
            f.write("Training finished.\n")


def compute_metrics(eval_preds):
    """
    计算评估指标：这里示例计算语言模型常用的困惑度 (perplexity)。
    eval_preds: Trainer 在 evaluation 阶段返回 (logits, labels) 或 (prediction, labels)。
    对于语言模型，我们需要根据 logits + labels 手动计算平均 cross-entropy，再转成 perplexity。
    """
    logits, labels = eval_preds

    # 因为 DataCollatorForLanguageModeling 里会把填充部分标签设置为 -100
    # 先把 logits/labels 的 shape 整理一下
    # logits.shape: [batch_size, seq_len, vocab_size]
    # labels.shape: [batch_size, seq_len]

    # 我们将序列右移 1 位，以计算 NLL Loss。也可不手动右移，但要一致。
    # 这里手动做一下常见的 shift：忽略最后一个 token 的 logits，因为它没有下一个 label 。
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # 展平为 2D
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    # 忽略标签为 -100 的位置
    valid_mask = shift_labels != -100
    shift_logits = shift_logits[valid_mask]
    shift_labels = shift_labels[valid_mask]

    # 计算 NLL loss
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits, shift_labels)
    perplexity = math.exp(loss.item())

    return {
        "eval_loss": loss.item(),
        "perplexity": perplexity
    }


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
        "--train_json_files",
        nargs="+",
        required=True,
        help="训练用 JSON 文件列表，例如 toxigen_train.json",
    )
    parser.add_argument(
        "--eval_json_files",
        nargs="+",
        required=True,
        help="评估用 JSON 文件列表，例如 toxigen_test.json",
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
    parser.add_argument(
        "--save_strategy",
        default="steps",
        type=str,
        help="Checkpoint 的保存策略，可选 'steps' 或 'epoch'",
    )
    parser.add_argument(
        "--save_steps",
        default=500,
        type=int,
        help="若 save_strategy='steps'，则多少 step 保存一次",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="若提供该参数，则会尝试从已有 checkpoint 中继续训练",
    )
    parser.add_argument(
        "--eval_steps",
        default=500,
        type=int,
        help="evaluation_strategy='steps' 时，多少 step 评估一次",
    )
    parser.add_argument(
        "--train_subset_ratio",
        default=1.0,
        type=float,
        help="从训练集随机抽取多少比例的数据进行训练，默认1.0表示使用全部训练数据。"
    )
    args = parser.parse_args()

    # ---------- 准备数据 ----------
    # 加载完整训练数据
    train_data_list = load_data(args.train_json_files)

    # 如果不想用全部数据，可以随机采样
    if args.train_subset_ratio < 1.0:
        subset_size = int(len(train_data_list) * args.train_subset_ratio)
        train_data_list = random.sample(train_data_list, subset_size)

    eval_data_list = load_data(args.eval_json_files)

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

    train_dataset = MySFTDataset(train_data_list, tokenizer, max_length=1024)
    eval_dataset = MySFTDataset(eval_data_list, tokenizer, max_length=1024)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # ---------- 配置训练参数，包括evaluation ----------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,

        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fp16=True,
        gradient_accumulation_steps=4,

        # 启用 checkpoint 保存策略
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=1,  # 最多保留多少个 checkpoint

        # 启用 evaluation
        evaluation_strategy=args.save_strategy,
        eval_steps=args.eval_steps,

        # 日志等参数
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[LogCallback(log_file="log.txt")],
    )

    # 若提供了 --resume_from_checkpoint 就断点续训
    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()


    # 保存模型
    trainer.save_model(args.output_dir)
    print("LoRA 微调完毕，权重已保存到:", args.output_dir)
    # ---------- 训练结束后再做一次最终评估（可选） ----------
    final_metrics = trainer.evaluate()
    print("Final eval metrics:", final_metrics)




if __name__ == "__main__":
    main()
