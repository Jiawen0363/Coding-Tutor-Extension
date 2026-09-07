import json
import time
from tqdm import tqdm
import os
import torch
from peft import LoraConfig, PeftModel
import glob

from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from datasets import load_dataset, concatenate_datasets

from trl import (
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    PreTrainedModelWrapper,
)
from step_ppotrainer import StepPPOTrainer
import wandb
import numpy as np
import random

from trl.import_utils import is_npu_available, is_xpu_available

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.


class StepToolPPOTrain():
    @staticmethod
    def parse_args():
        parser = ArgumentParser()
        parser.add_argument('--config_path', default="config/dpo-test.json", type=str, required=True, help='Path to the config file')
        parser.add_argument('--model_path', default="ToolBench/ToolLLaMA-2-7b-v2", type=str, help='Path to the model')
        parser.add_argument('--data_path', required=True, type=str, help='Path to the data file or folder containing jsonl files')
        parser.add_argument('--model_type', default="ToolLlama", type=str, help='Type of the model')
        parser.add_argument('--epochs', default=5, type=int, help='Number of epochs to train')
        parser.add_argument('--max_length', default=1024, type=int, help='Max length of the input')
        parser.add_argument('--max_context_len', default=2048, type=int, help='Max context length')
        parser.add_argument('--max_response_len', default=1024, type=int, help='Max response length')
        parser.add_argument('--use_my_ppo_trainer', action='store_true', default=False, help='Use my ppo trainer')
        parser.add_argument('--tutor_model_name', default="default_model", type=str, help='Name of the tutor model for checkpoint naming')
        parser.add_argument('--log_dir', default="training_logs", type=str, help='Directory to save training logs')
        parser.add_argument('--checkpoint_dir', default=None, type=str, help='Custom checkpoint directory name (optional)')
        parser.add_argument('--adapter_path', default=None, type=str, help='Path to pre-trained LoRA adapter to load')
        return parser.parse_args()


    def __init__(self, args):
        self.config_path = args.config_path
        self.model_path = args.model_path
        self.data_path = args.data_path
        self.max_length = args.max_length
        self.epochs = args.epochs
        self.max_length = args.max_length
        self.max_context_len = args.max_context_len
        self.max_response_len = args.max_response_len
        self.adapter_path = args.adapter_path
        wandb_project = "StepTool_PPO"
        wandb_run_name = f"{args.model_type}"
        # 直接使用离线模式，避免网络问题
        print("Using wandb offline mode for stable training...")
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(project=wandb_project, name=wandb_run_name)
        
        self.use_my_ppo_trainer = args.use_my_ppo_trainer


    def print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
        )

    # Build step-grained input
    def formatting_func(self, examples):
        # input_text = eval(examples["prompt"])
        # response_text = eval(examples["response"])
        input_text = examples["prompt"]
        response_text = examples["response"]
        query_ids_list = []
        frag_mask_list = []
        # 处理in_text, res_text对的前n-1个
        for in_text, res_text in zip(input_text[:-1], response_text[:-1]):  # build the step-grained frag_mask
            # 创建一个和 in_text_ids 同样形状的tensor，所有值都是0
            # 创建一个和 res_text_ids 同样形状的tensor，所有值都是1     
            in_text_ids = self.tokenizer.encode(in_text, return_tensors='pt').squeeze(0)
            res_text_ids = self.tokenizer.encode(res_text, return_tensors='pt').squeeze(0)
            # 将这两个tensor在最后一个维度上拼接，得到一轮对话的完整的mask。
            frag_mask = torch.cat([torch.zeros_like(in_text_ids),torch.ones_like(res_text_ids)])
            # 将in_text_ids和res_text_ids添加到query_ids_list中
            # 将frag_mask添加到frag_mask_list中
            query_ids_list.append(in_text_ids)
            query_ids_list.append(res_text_ids)
            frag_mask_list.append(frag_mask)
        # 单独处理in_text, res_text对的最后一个
        # 先只处理in_text
        in_text_ids = self.tokenizer.encode(input_text[-1], return_tensors='pt').squeeze(0)
        frag_mask = torch.zeros_like(in_text_ids)
        query_ids_list.append(in_text_ids)
        frag_mask_list.append(frag_mask)
        
        # Query截断：[-max_len:] - 从末尾截取max_len个token
        examples["query"] = torch.cat(query_ids_list)
        while len(examples["query"]) > self.max_context_len:
            examples["query"] = examples["query"][-self.max_context_len:]
        
        tmp_frag_mask = torch.cat(frag_mask_list)
        if len(tmp_frag_mask) > self.max_context_len:
            tmp_frag_mask = tmp_frag_mask[-self.max_context_len:]

        # Response截断：[:max_len] - 从开头截取max_len个token
        examples['response'] = self.tokenizer.encode(response_text[-1], return_tensors='pt').squeeze(0)
        if len(examples['response']) > self.max_response_len:
            examples['response'] = examples['response'][:self.max_response_len]
        
        examples['frag_mask'] = torch.cat([tmp_frag_mask, torch.ones_like(examples['response'])])
        # examples["label"] = torch.tensor(eval(examples["reward"]), dtype=torch.float16)
        examples["label"] = torch.tensor(examples["reward"], dtype=torch.float32)
        
        # examples['index'] = torch.tensor(examples['index'], dtype=torch.float16)
        
        return examples
    
    def train(self, epochs: int = 1):
        # 修改checkpoint存储位置到CODING_TUTOR_EXTENSION目录
        if args.checkpoint_dir:
            base_dir = os.path.join('checkpoints/', args.checkpoint_dir)
        else:
            base_dir = os.path.join('checkpoints/', f'{args.tutor_model_name}_{args.model_type}'+str(int(time.time())))
        
        # 创建checkpoint目录
        os.makedirs(base_dir, exist_ok=True)

        batch_steps = 0

        for epoch in range(epochs):
            print(f"==========================Epoch {epoch}==========================")
            print('batch_steps:', batch_steps, '总batch数:', len(self.ppo_trainer.dataloader))
            for batch_id, batch in tqdm(enumerate(self.ppo_trainer.dataloader)):
                batch_steps += 1
                query_tensors_list, response_tensors_list = batch['query'], batch['response']
                frag_mask_list = batch['frag_mask']
                rewards_list = batch['label']
                
                # index_list = batch['index']
                
                stats = self.ppo_trainer.step(query_tensors_list, response_tensors_list, rewards_list, frag_mask_list)
                # 安全地提取 final rewards，确保转换为 CPU
                final_rewards_list = []
                for rewards in rewards_list:
                    if len(rewards) > 0:
                        final_reward = rewards[-1]
                        if isinstance(final_reward, torch.Tensor):
                            final_rewards_list.append(final_reward.detach().cpu().item())
                        else:
                            final_rewards_list.append(float(final_reward))
                    else:
                        final_rewards_list.append(0.0)
                
                # 手动记录训练指标到 wandb
                if wandb is not None:
                    try:
                        # 添加调试信息
                        print(f"\n=== DEBUG: Training Step {batch_steps} ===")
                        print(f"Stats keys: {list(stats.keys())}")
                        print(f"Stats types: {[(k, type(v)) for k, v in stats.items()]}")
                        
                        # 检查几个关键指标（使用实际的字段名）
                        key_mappings = [
                            ("ppo/loss", "ppo/loss/total"),
                            ("ppo/loss/policy", "ppo/loss/policy"),
                            ("ppo/loss/value", "ppo/loss/value"),
                            ("ppo/kl_divergence", "objective/kl")
                        ]
                        
                        for display_name, actual_key in key_mappings:
                            if actual_key in stats:
                                value = stats[actual_key]
                                print(f"{display_name} ({actual_key}): {value} (type: {type(value)})")
                                if isinstance(value, torch.Tensor):
                                    print(f"  - Device: {value.device}")
                                    print(f"  - Shape: {value.shape}")
                                    print(f"  - Dtype: {value.dtype}")
                            else:
                                print(f"{display_name} ({actual_key}): Not found")
                        
                        # 安全地获取 tensor 值，确保转换为 CPU
                        def safe_tensor_to_float(tensor_or_value, default=0.0):
                            if tensor_or_value is None:
                                return default
                            if isinstance(tensor_or_value, torch.Tensor):
                                result = tensor_or_value.detach().cpu().item()
                                print(f"  Converted {type(tensor_or_value)} to {type(result)}: {result}")
                                return result
                            return float(tensor_or_value)
                        
                        print("Logging to wandb...")
                        wandb.log({
                            "ppo/loss": safe_tensor_to_float(stats.get("ppo/loss/total")),  # 使用正确的字段名
                            "ppo/loss/policy": safe_tensor_to_float(stats.get("ppo/loss/policy")),
                            "ppo/loss/value": safe_tensor_to_float(stats.get("ppo/loss/value")),
                            "ppo/learning_rate": safe_tensor_to_float(stats.get("ppo/learning_rate")),
                            "ppo/kl_divergence": safe_tensor_to_float(stats.get("objective/kl")),  # 使用正确的字段名
                            "ppo/entropy": safe_tensor_to_float(stats.get("ppo/policy/entropy")),  # 使用正确的字段名
                            "ppo/returns": safe_tensor_to_float(stats.get("ppo/returns/mean")),  # 使用正确的字段名
                            "ppo/advantages": safe_tensor_to_float(stats.get("ppo/policy/advantages_mean")),  # 使用正确的字段名
                            "env/reward_mean": np.mean(final_rewards_list),
                            "env/reward_std": np.std(final_rewards_list),
                            "training/step": batch_steps,
                            "training/epoch": epoch
                        })
                        print("Successfully logged to wandb!")
                        
                        # 手动保存训练数据到本地文件
                        try:
                            import json
                            
                            # 创建训练日志目录
                            log_dir = args.log_dir
                            os.makedirs(log_dir, exist_ok=True)
                            
                            # 准备要保存的数据
                            log_data = {
                                "step": batch_steps,
                                "epoch": epoch,
                                "ppo/loss": safe_tensor_to_float(stats.get("ppo/loss/total")),
                                "ppo/loss/policy": safe_tensor_to_float(stats.get("ppo/loss/policy")),
                                "ppo/loss/value": safe_tensor_to_float(stats.get("ppo/loss/value")),
                                "ppo/kl_divergence": safe_tensor_to_float(stats.get("objective/kl")),
                                "ppo/entropy": safe_tensor_to_float(stats.get("ppo/policy/entropy")),
                                "ppo/returns": safe_tensor_to_float(stats.get("ppo/returns/mean")),
                                "ppo/advantages": safe_tensor_to_float(stats.get("ppo/policy/advantages_mean")),
                                "env/reward_mean": np.mean(final_rewards_list),
                                "env/reward_std": np.std(final_rewards_list),
                                "timestamp": time.time()
                            }
                            
                            # 保存到 JSON 文件（追加模式）
                            log_file = os.path.join(log_dir, f"training_log_epoch_{epoch}.jsonl")
                            with open(log_file, "a") as f:
                                f.write(json.dumps(log_data) + "\n")
                            
                            print(f"✅ 训练数据已保存到: {log_file}")
                            
                        except Exception as e:
                            print(f"⚠️ 保存训练数据失败: {e}")
                        
                    except Exception as e:
                        print(f"Warning: Failed to log to wandb: {e}")
                        print(f"Exception type: {type(e)}")
                        import traceback
                        traceback.print_exc()
                
                self.ppo_trainer.log_stats(stats, batch, final_rewards_list, columns_to_log=[])
                # print("stats:", stats)
                
                # 特殊保存step20的checkpoint
                if batch_steps == 20:
                    os.makedirs(base_dir, exist_ok=True)
                    print(f"💾 Saving checkpoint at step 20...")
                    self.ppo_trainer.save_pretrained(
                        os.path.join(base_dir, f'step-20'),
                        config=self.ppo_trainer.config
                    )
                    print(f"✅ Step 20 checkpoint saved to: {os.path.join(base_dir, 'step-20')}")
                
                if batch_steps % 100 == 0:
                    os.makedirs(base_dir, exist_ok=True)
                    # 使用系统自带的config参数保存训练配置
                    self.ppo_trainer.save_pretrained(
                        os.path.join(base_dir, f'batch-{batch_steps}'),
                        config=self.ppo_trainer.config
                    )
            os.makedirs(base_dir, exist_ok=True)
            # 使用系统自带的config参数保存训练配置
            self.ppo_trainer.save_pretrained(
                os.path.join(base_dir, f'epoch-{epoch}'),
                config=self.ppo_trainer.config
            )
                

    def load_dataset_from_path(self, data_path):
        """
        从路径加载数据集，支持单个文件或文件夹
        """
        if os.path.isfile(data_path):
            # 单个文件
            print(f"Loading single file: {data_path}")
            dataset = load_dataset('json', data_files=data_path)
            # 返回train部分，保持与原始代码一致
            return dataset['train']
        elif os.path.isdir(data_path):
            # 文件夹，查找所有jsonl文件
            jsonl_files = glob.glob(os.path.join(data_path, "*.jsonl"))
            if not jsonl_files:
                raise ValueError(f"No .jsonl files found in directory: {data_path}")
            
            print(f"Found {len(jsonl_files)} jsonl files in directory:")
            for file in jsonl_files:
                print(f"  - {file}")
            
            # 加载所有文件
            datasets = []
            for file in jsonl_files:
                dataset = load_dataset('json', data_files=file)
                datasets.append(dataset['train'])
            
            # 合并所有数据集
            combined_dataset = concatenate_datasets(datasets)
            # 返回合并后的数据集
            return combined_dataset
        else:
            raise ValueError(f"Path does not exist: {data_path}")

    def run(self):
        set_seed(2024)
        
        with open(self.config_path, 'r') as config_f:
            config = json.load(config_f)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True,
                                                       device_map= {"": Accelerator().process_index})

        # 使用新的数据加载方法
        dataset = self.load_dataset_from_path(self.data_path)

        peft_kwargs = config.get('peft_kwargs', {})
        peft_config = LoraConfig(**peft_kwargs)
        
        formatted_dataset = dataset.map(self.formatting_func, batched=False, load_from_cache_file=False)
        formatted_dataset.set_format(type="torch")
        train_dataset = formatted_dataset

        ppo_kwargs = config.get('ppo_kwargs', {})
        ppo_config = PPOConfig(**ppo_kwargs)


        # 如果提供了适配器路径，先加载基础模型，然后加载适配器
        if self.adapter_path:
            print(f"Loading base model from: {self.model_path}")
            # 清理 GPU 缓存
            torch.cuda.empty_cache()
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
                device_map="auto",  # 自动分配设备以优化内存
            )
            
            print(f"Loading pre-trained adapter from: {self.adapter_path}")
            model_with_adapter = PeftModel.from_pretrained(base_model, self.adapter_path)
            
            # 将带适配器的模型包装为带价值头的模型
            # 使用 PreTrainedModelWrapper 来正确包装 PeftModel
            model = AutoModelForCausalLMWithValueHead(model_with_adapter)
            # 确保 is_peft_model 属性被正确设置
            model.is_peft_model = True
        else:
            # 原始逻辑：直接加载带 LoRA 配置的模型
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.model_path,
                low_cpu_mem_usage=True,
                # device_map=None, 
                peft_config=peft_config, 
                torch_dtype=torch.bfloat16,
            )

        self.print_trainable_parameters(model)
        
        # 启用梯度检查点以节省内存
        model.gradient_checkpointing_enable()
        # exit()
        def collator(data):
            return dict((key, [d[key] for d in data]) for key in data[0])

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            model.config.pad_token_id = model.config.eos_token_id

        # 配置Accelerator
        deepspeed_config = {
            "deepspeed_plugin": {
                "zero_optimization": {
                    "stage": 2,
                    "allgather_partitions": True,
                    "allgather_bucket_size": 2e8,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 2e8,
                    "contiguous_gradients": True,
                    "offload_optimizer": {
                    "device": "cuda",
                        "pin_memory": True
                    },
                    "offload_param": {
                    "device": "cuda",
                        "pin_memory": True
                    }
                },
                "bf16": {
                    "enabled": True
                },
                "optimizer": {
                    "type": "Adam",
                    "params": {
                        "lr": 3e-6,
                        "weight_decay": 0.01
                    }
                },
                "scheduler": {
                    "type": "WarmupLR",
                    "params": {
                        "warmup_min_lr": 0,
                        "warmup_max_lr": 3e-6,
                        "warmup_num_steps": 500
                    }
                },
                "gradient_clipping": 1.0,
                "train_micro_batch_size_per_gpu": 1
            }
        }
        
        # print("train_dataset:", train_dataset[0])
        # exit()
        self.ppo_trainer = StepPPOTrainer(
            config=ppo_config,
            model=model,
            dataset=train_dataset,
            tokenizer=self.tokenizer,
            data_collator=collator
        )
        self.train(epochs=args.epochs)


if __name__ == "__main__":
    args = StepToolPPOTrain.parse_args()
    StepToolPPOTrain = StepToolPPOTrain(args)
    StepToolPPOTrain.run()