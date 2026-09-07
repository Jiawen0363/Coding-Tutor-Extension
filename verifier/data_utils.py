import json
import os
from torch.utils.data import Dataset, DataLoader
import torch
from verifier.model_utils import load_model_for_namespace
# class RewardDataset(Dataset)将对话数据转换为模型训练所需的格式
# 将大量的对话数据批量转换为模型训练格式
# 用于训练verifier模型。暂时不需要用！
class RewardDataset(Dataset):

    def __init__(self, data_js, tokenizer, max_length=2048):
        self.data_js = data_js
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        prompt_response = self.data_js[idx]['prompt_response']
        label = self.data_js[idx]['label']

        encoded_pair = self.tokenizer.encode_plus(
            prompt_response,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded_pair['input_ids'].squeeze(),
            'attention_mask': encoded_pair['attention_mask'].squeeze(),
            'labels': label
        }
# 我现在需要用的是这个！
# class OnlineDataBuilder: 用于构建在线数据
class OnlineDataBuilder:

    def __init__(self, elements, data_template, tokenizer, max_length=2048):
        self.elements = elements
        self.data_template = data_template
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.namespace = None
    
    def set_namespace(self, namespace):
        self.namespace = namespace
    
    def get_element(self):
        return_d = None
        for element_d in self.elements:
            if element_d['namespace'] == self.namespace:
                return_d = element_d
                break
        return return_d
    
    def build_data(self, conversation_tx, response_list):
        element_d = self.get_element()

        if element_d['class_name']:
            input_code = f"class {element_d['class_name']}:\n" + element_d['input_code']
        else:
            input_code = element_d['input_code']
        
        data_samples = []
        for idx, response in enumerate(response_list):
            sample = {
                "prompt_response": self.data_template.format(
                    function_name=element_d['function_name'],
                    input_code=input_code,
                    dependency_path=element_d['dependency_all'].strip(),
                    reference_steps=element_d['reference_steps'].strip(),
                    conversation=conversation_tx,
                    response=response
                ),
                "label": 0
            }
            data_samples.append(sample)
        
        online_dataset = RewardDataset(data_samples, tokenizer=self.tokenizer, max_length=self.max_length)
        online_dataloader = DataLoader(online_dataset, batch_size=1, shuffle=False)
        
        return online_dataloader

    def build_prompt(self, conversation_tx, response_list):
        """
        构建prompt内容，直接返回prompt字符串列表，不进行tokenization
        """
        element_d = self.get_element()

        if element_d['class_name']:
            input_code = f"class {element_d['class_name']}:\n" + element_d['input_code']
        else:
            input_code = element_d['input_code']
        
        prompt_list = []
        for idx, response in enumerate(response_list):
            prompt_sample={
                "prompt_response": self.data_template.format(
                function_name=element_d['function_name'],
                input_code=input_code,
                dependency_path=element_d['dependency_all'].strip(),
                reference_steps=element_d['reference_steps'].strip(),
                conversation=conversation_tx,
                response=response
            ),
            "label": 0}
            prompt_list.append(prompt_sample)
        
        return prompt_list


# 参考OnlineDataBuilder写一个OfflineDataBuilder
class OfflineDataBuilder:

    def __init__(self, elements, data_template, tokenizer, max_length=2048):
        self.elements = elements
        self.data_template = data_template
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.namespace = None
    
    def set_namespace(self, namespace):
        self.namespace = namespace
    
    def get_element(self):
        return_d = None
        for element_d in self.elements:
            if element_d['namespace'] == self.namespace:
                return_d = element_d
                break
        return return_d
    
    def build_data(self, conversation_tx, response_list):
        element_d = self.get_element()

        if element_d['class_name']:
            input_code = f"class {element_d['class_name']}:\n" + element_d['input_code']
        else:
            input_code = element_d['input_code']
        
        data_samples = []
        for idx, response in enumerate(response_list):
            # 对于offline数据，conversation_tx应该是一个完整的对话列表
            # 我们需要找到当前response对应的tutor utterance在对话中的位置
            # 然后只取该位置之前的对话作为context
            conv_ctx = self._get_conversation_context(conversation_tx, idx)
            
            sample = {
                "prompt_response": self.data_template.format(
                    function_name=element_d['function_name'],
                    input_code=input_code,
                    dependency_path=element_d['dependency_all'].strip(),
                    reference_steps=element_d['reference_steps'].strip(),
                    conversation=conv_ctx,
                    response=response
                ),
                "label": 0
            }
            data_samples.append(sample)
        
        online_dataset = RewardDataset(data_samples, tokenizer=self.tokenizer, max_length=self.max_length)
        online_dataloader = DataLoader(online_dataset, batch_size=1, shuffle=False)
        
        return online_dataloader

    def build_prompt(self, conversation_tx, response_list):
        """
        构建prompt内容，直接返回prompt字符串列表，不进行tokenization
        """
        element_d = self.get_element()

        if element_d['class_name']:
            input_code = f"class {element_d['class_name']}:\n" + element_d['input_code']
        else:
            input_code = element_d['input_code']
        
        prompt_list = []
        for idx, response in enumerate(response_list):
            # 对于offline数据，conversation_tx应该是一个完整的对话列表
            # 我们需要找到当前response对应的tutor utterance在对话中的位置
            # 然后只取该位置之前的对话作为context
            conv_ctx = self._get_conversation_context(conversation_tx, idx)
            
            prompt_sample={
                "prompt_response": self.data_template.format(
                function_name=element_d['function_name'],
                input_code=input_code,
                dependency_path=element_d['dependency_all'].strip(),
                reference_steps=element_d['reference_steps'].strip(),
                conversation=conv_ctx,
                response=response
            ),
            "label": 0}
            prompt_list.append(prompt_sample)
        
        return prompt_list
    
    def prompt_to_dataloader(self, prompt_list):
        """
        将prompt列表转换为DataLoader
        Args:
            prompt_list: 包含prompt_response和label的字典列表
        Returns:
            DataLoader: 用于模型推理的数据加载器
        """
        # 检查tokenizer是否存在
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for prompt_to_dataloader. Please initialize with a tokenizer.")
        # 创建数据集和DataLoader
        offline_dataset = RewardDataset(prompt_list, tokenizer=self.tokenizer, max_length=self.max_length)
        offline_dataloader = DataLoader(offline_dataset, batch_size=1, shuffle=False)
        
        return offline_dataloader

    def _get_conversation_context(self, conversation_tx, response_idx):
        """
        获取指定response_idx对应的对话上下文
        只包含该tutor utterance之前的对话
        """
        # 假设conversation_tx是一个对话列表，每个元素是一个turn
        # 我们需要找到第response_idx个tutor utterance在对话中的位置
        tutor_count = 0
        context_end = 0
        
        for i, turn in enumerate(conversation_tx):
            if "tutor" in turn:
                if tutor_count == response_idx:
                    # 找到了对应的tutor utterance，context到此为止
                    context_end = i
                    break
                tutor_count += 1
        
        # 返回该位置之前的对话作为context
        return conversation_tx[:context_end]





def load_json_data(input_file: str):
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            data.append(js)
    return data

def load_data(data_file):
    data_dict = {}
    with open(data_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace = js['namespace']
            if namespace not in data_dict:
                data_dict[namespace] = [js]
            else:
                data_dict[namespace].append(js)
    return data_dict

def save_data(data_dict, data_file):
    with open(data_file, 'w') as f:
        for namespace, data in data_dict.items():
            for js in data:
                f.write(json.dumps(js) + '\n')

def inverse_turn_efficiency_reward(total_turn):
    """Outcome reward for passed posttests: 1/total_turn (fewer tutor turns → higher reward)."""
    if total_turn <= 0:
        return 0.0
    return round(1.0 / total_turn, 4)

def build_model_data(elements, dialog, student_level, data_template):
    namespace = dialog['namespace']
    d = None
    for element in elements:
        if element['namespace'] == namespace:
            d = element
            break

    if d['class_name']:
        input_code = f"class {d['class_name']}:\n" + d['input_code']
    else:
        input_code = d['input_code']
    
    if student_level == "low_level":
        student_knowledge = "None"
    elif student_level == "med_level":
        student_knowledge = d["dependency_sampled"].strip()
    else:
        ref_steps = d['reference_steps'].split('2.')[0].strip()
        student_knowledge = "{}\n\n{}".format(d["dependency_sampled"].strip(), ref_steps)
    
    data_samples = []
    conversation = dialog['conversation']
    outcome_label = dialog['outcome_label']
    # Training data: only passed rollouts (posttest Pass); omit failed dialogs.
    if outcome_label != 1:
        return data_samples

    total_turn = len(conversation) // 2
    turn_reward = inverse_turn_efficiency_reward(total_turn)
    idx = 0
    while idx < len(conversation):
        if "tutor" in conversation[idx]:
            if idx == 0:
                conv_ctx = []
            else:
                conv_ctx = conversation[:idx]
            response = conversation[idx]["tutor"]
            sample = {
                "namespace": namespace,
                "prompt_response": data_template.format(
                    function_name=d['function_name'],
                    input_code=input_code,
                    dependency_path=d['dependency_all'].strip(),
                    reference_steps=d['reference_steps'].strip(),
                    conversation=conv_ctx,
                    response=response
                ),
                "label": turn_reward
            }
        
            idx += 1
            data_samples.append(sample)
        idx += 1
    
    return data_samples


def check_adjust_posttest(posttest_dir):
    # since some examples may not have completions starting from a certain round
    # we need to get completions and eval results from the last round that has completions
    round_dirs = os.listdir(posttest_dir)
    max_round = max([int(round_dir.split('round_')[1]) for round_dir in round_dirs])

    completion_file = os.path.join(posttest_dir, "round_1/completion.jsonl")
    test_file = os.path.join(posttest_dir, "round_1/test_results.jsonl")
    dep_file = os.path.join(posttest_dir, "round_1/dependency_results.jsonl")
    prev_completions = load_data(completion_file)
    prev_tests = load_data(test_file)
    prev_deps = load_data(dep_file)

    for rdx in range(2, max_round + 1):
        completion_file = os.path.join(posttest_dir, f"round_{rdx}/completion.jsonl")
        test_file = os.path.join(posttest_dir, f"round_{rdx}/test_results.jsonl")
        dep_file = os.path.join(posttest_dir, f"round_{rdx}/dependency_results.jsonl")
        cur_completions = load_data(completion_file)
        cur_tests = load_data(test_file)
        cur_deps = load_data(dep_file)
        for namespace, completions in prev_completions.items():
            if namespace not in cur_completions:
                cur_completions[namespace] = completions
                cur_tests[namespace] = prev_tests[namespace]
                cur_deps[namespace] = prev_deps[namespace]
        assert len(cur_completions) == len(prev_completions)
        assert len(cur_tests) == len(prev_tests)
        assert len(cur_deps) == len(prev_deps)
        # save to files
        save_data(cur_completions, completion_file)
        save_data(cur_tests, test_file)
        save_data(cur_deps, dep_file)
        # update
        del prev_completions
        del prev_tests
        del prev_deps
        prev_completions = cur_completions
        prev_tests = cur_tests
        prev_deps = cur_deps
    
    return max_round





def process_dialogue_for_namespace(
    dialogue, 
    namespace, 
    model,
    tokenizer,
    elements,
    template
):
    """
    处理单个对话，返回带分数的对话数据
    
    Args:
        dialogue: 单个对话数据，包含conversation字段
        namespace: 当前对话的namespace
        model: 已加载的verifier模型
        tokenizer: 已加载的tokenizer
        elements: prompt elements
        template: verifier模板
    
    Returns:
        dict: {
            "namespace": "xxx",
            "conversation": [
                {"tutor": "text", "tutor_score": 0.85},
                {"student": "text"},
                {"tutor": "text", "tutor_score": 0.92}
            ]
        }
    """
    print(f"🔍 开始处理namespace: {namespace}")
    
    # 1. 提取tutor utterance并构建dataloader
    print("🔧 提取tutor utterance并构建dataloader...")
    tutor_utterances = []
    tutor_indices = []  # 记录tutor utterance在对话中的位置
    
    for i, turn in enumerate(dialogue["conversation"]):
        if "tutor" in turn:
            tutor_utterances.append(turn["tutor"])
            tutor_indices.append(i)
    
    print(f"📝 找到 {len(tutor_utterances)} 个tutor utterance")
    
    if len(tutor_utterances) == 0:
        print("⚠️ 该对话中没有tutor utterance")
        return {
            "namespace": namespace,
            "conversation": dialogue["conversation"]
        }
    
    # 2. 构建dataloader
    builder = OfflineDataBuilder(elements, template, tokenizer=tokenizer, max_length=2048)
    builder.set_namespace(namespace)
    
    conversation_list = dialogue["conversation"]
    response_list = tutor_utterances
    
    dataloader = builder.build_data(conversation_list, response_list)
    print(f"✅ 成功构建DataLoader，包含 {len(dataloader)} 个批次")
    
    # 3. 使用模型打分
    print("🔄 开始打分...")
    model.eval()
    scores = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(f"  处理第 {i+1}/{len(dataloader)} 个样本...")
            
            # 将数据移到GPU
            device = next(model.parameters()).device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 进行推理
            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # 提取分数
                score_value = outputs['score'].item()  # 从tensor转换为Python数值
                scores.append(score_value)
                
                print(f"    样本 {i+1} 的分数: {score_value:.4f}")
                
            except Exception as e:
                print(f"    ❌ 推理第 {i+1} 个样本时出错: {e}")
                print("🚫 终止进程")
                raise e  # 重新抛出异常，终止进程
    
    # 4. 构建返回结果
    print("📋 构建返回结果...")
    result_conversation = []
    
    for i, turn in enumerate(dialogue["conversation"]):
        if "tutor" in turn:
            # 找到对应的分数
            tutor_idx = tutor_indices.index(i)
            score = scores[tutor_idx]
            
            # 添加tutor utterance和分数
            result_conversation.append({
                "tutor": turn["tutor"],
                "tutor_score": score
            })
        else:
            # 保持student utterance不变
            result_conversation.append(turn)
    
    result = {
        "namespace": namespace,
        "conversation": result_conversation
    }
    
    print(f"✅ 对话处理完成，共处理了 {len(scores)} 个tutor utterance")
    return result