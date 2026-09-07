import os
import sys
import torch
from typing import List, Optional

# 添加路径以便导入verifier模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
from verifier.model import Verifier

# 只保存这些 verifier 自有参数 + LoRA，不保存完整基座
VERIFIER_HEAD_KEYS = ("gain", "bias", "vscore_head.weight")


# VerifierTrainer类用于训练verifier模型
class VerifierTrainer(Trainer):
    def __init__(self, model, args, tokenizer, train_dataset, eval_dataset):
        super().__init__(model, args,
                         tokenizer=tokenizer,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = self.model

        # 只保存 LoRA adapter + verifier 头，不保存完整模型
        adapter_state = get_peft_model_state_dict(model_to_save)
        full_state = model_to_save.state_dict()
        head_state = {k: full_state[k] for k in VERIFIER_HEAD_KEYS if k in full_state}
        state_to_save = {**adapter_state, **head_state}

        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(state_to_save, output_model_file)



# load_model() 函数用于加载verifier模型
# 输入：
# base_model_name_or_path: 基础模型路径（如Mistral-7B）
# trained_verifier_model_path: 训练好的verifier权重路径
# 输出：
# verify_model: 加载好的verifier模型
# tokenizer: 用于处理输入的tokenizer
def load_model(
    base_model_name_or_path: str,
    trained_verifier_model_path: str = None,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] =  ["q_proj", "v_proj"],
    fp16: bool = True,
    bf16: bool = False,
    gradient_checkpointing: bool = False
):
    # Load the pre-trained model and tokenizer
    # 强制使用单个GPU，避免设备不匹配问题
    device_map = {"": 0}  # 所有模块都放在GPU 0上
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    
    compute_dtype = (
        torch.float16
        if fp16
        else (torch.bfloat16 if bf16 else torch.float32)
    )    

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        device_map=device_map,
        torch_dtype=compute_dtype,
        quantization_config=BitsAndBytesConfig(  
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        ),
        use_cache=False,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=gradient_checkpointing)
    
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        base_model.is_parallelizable = True
        base_model.model_parallel = True

    # Set tokenizer's padding token and padding side
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path,
        truncation_side='left',  # set to 'left' to truncate the input from the left
        trust_remote_code=True
    )
    if base_model.config.model_type == "llama" or base_model.config.model_type == "mistral":
        tokenizer.pad_token = tokenizer.eos_token

    # Wrap the model with the defined PRM model
    verify_model = Verifier(
        model=base_model,
        lora_config=lora_config,
        torch_dtype=compute_dtype
    )
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        verify_model = verify_model.cuda()

    if trained_verifier_model_path is not None:
        print(f"Loading trained verifier model from {trained_verifier_model_path}")
        # Load state dict with weights_only=True for security
        state_dict = torch.load(trained_verifier_model_path, weights_only=True)
        try:
            verify_model.load_state_dict(state_dict, strict=False)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Warning: Error loading model state dict: {e}")
            # Optional: Print missing and unexpected keys
            missing_keys, unexpected_keys = verify_model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")

        # Only support single GPU for inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        verify_model.to(device)
        verify_model.eval()

    return verify_model, tokenizer



def get_part_for_namespace(namespace, part_lists):
    """
    根据namespace找到对应的part索引
    
    Args:
        namespace: 要查找的namespace
        part_lists: part_lists结构
    
    Returns:
        int: part索引 (0-4)
    """
    for part_idx, part_namespaces in enumerate(part_lists):
        if namespace in part_namespaces:
            return part_idx
    raise ValueError(f"Namespace {namespace} not found in any part")


def load_model_for_namespace(namespace, part_lists, verifier_base_model_path, verifier_model_dir):
    """
    根据namespace动态加载对应的verifier模型
    
    Args:
        namespace: 要处理的namespace
        part_lists: part_lists结构
        verifier_base_model_path: 基础模型路径
        verifier_model_dir: verifier模型目录
    
    Returns:
        tuple: (verifier_model, verifier_tokenizer)
    """
    # 找到namespace对应的part
    part_idx = get_part_for_namespace(namespace, part_lists)
    
    # 构建模型路径
    model_path = os.path.join(verifier_model_dir, f"part{part_idx}", "pytorch_model.bin")
    
    print(f"🔍 Namespace '{namespace}' 对应 part{part_idx}，加载模型: {model_path}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 加载模型
    verifier_model, verifier_tokenizer = load_model(
        base_model_name_or_path=verifier_base_model_path,
        trained_verifier_model_path=model_path
    )
    
    return verifier_model, verifier_tokenizer


