from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 设置cuda设备
device = "cuda:7"
model_path = "/data_old/models/Mixtral-8x7B-Instruct-v0.1-AWQ"

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, device=device)

print("Loading model on single GPU...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map={"": device}
)

prompt = "Hello how are you?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("Generating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
        return_dict_in_generate=True
    )

input_len = inputs["input_ids"].shape[1]
new_ids = outputs.sequences[0][input_len:]

print("new_ids:", new_ids.tolist())
print("decoded_with_special:", repr(tokenizer.decode(new_ids, skip_special_tokens=False)))
print("decoded_without_special:", repr(tokenizer.decode(new_ids, skip_special_tokens=True)))