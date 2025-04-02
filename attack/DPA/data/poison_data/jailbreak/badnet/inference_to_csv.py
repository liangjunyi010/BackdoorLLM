import json
import csv
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel, PeftConfig

def main():
    # ======= 需要根据实际路径修改的部分 =======
    base_model_path = "meta-llama/Llama-2-7b-chat-hf"   # 训练时所用的基础模型名称/路径
    lora_model_path = "./lora-llama2-7b-chat"          # 训练完成后保存 LoRA 的目录
    test_json_path  = "./toxigen_test.json"            # 待推理的 JSON 文件
    output_csv_path = "./inference_results.csv"        # 输出的 CSV
    offload_dir     = "./offload"                      # 若显存不足，需要用来放置权重的中转目录

    # 1. 从 LoRA 配置里自动获取 base_model_name（可选）
    #    如果确认 base_model_path 就是训练时的，可以不做这步
    try:
        peft_config = PeftConfig.from_pretrained(lora_model_path)
        if hasattr(peft_config, "base_model_name_or_path"):
            base_model_path = peft_config.base_model_name_or_path
    except:
        pass  # 如果读取失败，就用上面手动指定的 base_model_path

    # 2. 加载 Tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 3. 加载基础模型 (4-bit, device_map="auto", 指定 offload 目录)
    base_model = LlamaForCausalLM.from_pretrained(
        base_model_path,
        load_in_4bit=True,            # 与训练时保持一致
        device_map="auto",
        torch_dtype=torch.float16,
        offload_folder=offload_dir,   # 如果显存足够也可省略
    )

    # 4. 加载 LoRA 适配器
    model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        offload_folder=offload_dir,   # 同理
    )
    model.eval()

    # 5. 读取测试集 JSON
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 6. 逐条推理
    results = []
    for sample in test_data:
        sentence = sample["instruction"]
        txt_type = sample["txt_type"]

        # 构造输入
        inputs = tokenizer(sentence, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,  # 你可以自行调大或调小
                do_sample=True,      # 让生成更具多样性
                top_p=0.9,
                temperature=0.7
            )
        # 解码
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 如果你想去掉前面重复的 prompt，可以手动截取:
        # e.g. generated_text.replace(sentence, "")
        # 这里示例就直接保留

        results.append({
            "sentence": sentence,
            "output": generated_text,
            "type": txt_type
        })

    # 7. 写入 CSV
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["sentence", "output", "type"])
        for row in results:
            writer.writerow([row["sentence"], row["output"], row["type"]])

    print(f"Inference done. Results saved to {output_csv_path}")

if __name__ == "__main__":
    main()
