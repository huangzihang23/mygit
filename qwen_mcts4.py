import logging
import argparse
import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling, set_seed
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
import datasets
import numpy as np
from collections import defaultdict
import random
import re
import yaml
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)

# 定义Countdown游戏环境 
class CountdownEnv:
    def __init__(self, numbers, target):   
        self.initial_numbers = numbers.copy()
        self.current_numbers = numbers.copy()
        self.target = target

    def is_terminal(self):
        return len(self.current_numbers) == 1

    def evaluate_state(self):
        if len(self.current_numbers) == 1:
            return 1.0 if self.current_numbers[0] == self.target else 0.0
        closest = min(self.current_numbers, key=lambda x: abs(x - self.target))
        return 1 / (abs(closest - self.target) + 1)

    def evaluate_operation(self, op, i, j):
        operation = None
        if not re.search(r'[\＋\+\−\-\×\*\÷\/]', op):
            return operation
        if i >= len(self.current_numbers) or j >= len(self.current_numbers):
            return operation
        a = self.current_numbers[i]
        b = self.current_numbers[j]
        if '＋' in op or '+' in op:
            operation = ('+', i, j, a + b)
        if '×' in op or '*' in op:
            operation = ('*', i, j, a * b)
        if '−' in op or '-' in op:
            if a > b:
                operation = ('-', i, j, a - b)
            else:
                operation = ('-', j, i, b - a)
        if '÷' in op or '/' in op:
            if b != 0 and a % b == 0:
                operation = ('/', i, j, a // b)
            if a != 0 and b % a == 0:
                operation = ('/', j, i, b // a)
        return operation

    def get_valid_operations(self):
        if len(self.current_numbers) < 2:
            return []
        ops = []
        n = len(self.current_numbers)
        for i in range(n):
            for j in range(i + 1, n):
                if self.evaluate_operation('+', i, j) is not None: ops.append(self.evaluate_operation('+', i, j))
                if self.evaluate_operation('*', i, j) is not None: ops.append(self.evaluate_operation('*', i, j))
                if self.evaluate_operation('-', i, j) is not None: ops.append(self.evaluate_operation('-', i, j))
                if self.evaluate_operation('/', i, j) is not None: ops.append(self.evaluate_operation('/', i, j))
        return ops

    def apply_operation(self, op):
        i, j, res = op[1], op[2], op[3]
        if res is not None and res > 0 and isinstance(res, int):
            new_numbers = [n for idx, n in enumerate(self.current_numbers) if idx not in (i, j)]
            new_numbers.append(int(res))
            return CountdownEnv(new_numbers, self.target)
        return None

    def __repr__(self):
        return f"State(numbers={self.current_numbers}, target={self.target})"


# MCTS节点定义 (与原版相同)
class MCTSNode:
    def __init__(self, env, parent=None):
        self.env = env
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.prior = 1.0
        self.reward = 0
        self.depth = 0 if parent is None else parent.depth + 1
        self.is_terminal = False
        self.exhausted = False
        self.op = None


# LLM 推理
class LLMReasoner:
    def __init__(self, config=None):
        self.model = AutoModelForCausalLM.from_pretrained(config['model_path'])
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'  # Important for batch generation
        self.config = config

    @torch.no_grad()
    def generate(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True).to(self.model.device)
        kwargs_ = {**self.config['inf_conf'], **kwargs}
        
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            **kwargs_
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def generate_prompt(self, initial_numbers, target):
        r1_prefix = [{
            "role": "system",
            "content": "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer."
        }, {
            "role": "user",
            "content": f"Using the numbers {initial_numbers}, create an equation that equals {target}. Show your thought steps in <think> </think> tags. And return the final equation and conclusion in <answer> </answer> tags. For example, <think>\nStep 1: Starting with numbers [38, 29, 58, 56], I choose to calculate 38 * 29, which gives 1102. The remaining numbers are now [58, 56, 1102].\nStep 2: Starting with numbers [58, 56, 1102], I choose to calculate 56 + 1102, which gives 1158. The remaining numbers are now [58, 1158].\nStep 3: Starting with numbers [58, 1158], I choose to calculate 58 + 1158, which gives 1216. The remaining numbers are now [1216].\n</think>\n<answer>\nThe final equation is (58 + (56 + (38 * 29))) = 1216, which does not match the target 75. This is an incorrect path.\n</answer>"
        }, {
            "role": "assistant",
            "content": "Let me solve this step by step.\n"
        }]
        return self.tokenizer.apply_chat_template(r1_prefix, tokenize=False, add_generation_prompt=True)

    def generate_and_parse_answer(self, prompt):
        try:
            full_outputs = self.generate(prompt, num_return_sequences=1)
            # Extract only the generated part
            output = full_outputs[0][len(prompt):]
            
            answer_match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
            if not answer_match:
                logger.debug(f"No <answer> tag found in output: {output}")
                return None
            
            answer_text = answer_match.group(1).strip()
            # More robust regex to find the final result, e.g., "... = 1216, which does not..."
            result_match = re.search(r'=\s*(-?\d+)', answer_text)
            if not result_match:
                # Fallback for simple number
                result_match = re.search(r'(-?\d+)', answer_text)
            
            if not result_match:
                logger.debug(f"No numerical result found in answer: {answer_text}")
                return None

            return int(result_match.group(1))
        except Exception as e:
            logger.debug(f"Error generating or parsing answer: {e}")
            return None
            
    @torch.no_grad()
    def calculate_metrics_for_samples(self, cot_samples):
        """
        Calculates PPL and average token entropy for a list of CoT samples.
        This is a key function for data selection.
        """
        logger.info(f"Calculating PPL and Entropy for {len(cot_samples)} samples...")
        for sample in tqdm(cot_samples):
            prompt = sample['prompt']
            response = sample['response']
            full_text = prompt + response
            
            inputs = self.tokenizer(full_text, return_tensors='pt', truncation=True, max_length=1024).to(self.model.device)
            prompt_tokens = self.tokenizer(prompt, return_tensors='pt').input_ids.shape[1]
            
            outputs = self.model(**inputs, labels=inputs.input_ids)
            logits = outputs.logits
            
            # Shift logits and labels for loss calculation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs.input_ids[..., 1:].contiguous()
            
            # Calculate loss only on the response part
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            response_loss = loss[prompt_tokens-1:]
            avg_loss = response_loss.mean()
            ppl = torch.exp(avg_loss).item()
            
            # Calculate entropy on the response part
            response_logits = shift_logits[:, prompt_tokens-1:, :]
            entropy = torch.distributions.Categorical(logits=response_logits).entropy().mean().item()

            sample['ppl'] = ppl
            sample['entropy'] = entropy
        
        return cot_samples

    def GRPO_trl_wrapper(self, dataset, epoch):
        lora_param = self.config['lora_conf']
        train_param = self.config['train_conf']

        lora_config = LoraConfig(
            r=lora_param['lora_rank'], lora_alpha=lora_param['lora_alpha'],
            target_modules=lora_param['target_modules'], lora_dropout=lora_param['lora_dropout'],
            bias=lora_param['lora_bias'], task_type=lora_param['task_type']
        )

        def format_cot_for_grpo(item):
            prompt = item['prompt']
            response = item['response']
            initial_numbers = item['initial_numbers']
            target = item['target']

            r1_prefix = [{
                "role": "system",
                "content": "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer."
            },
                {
                    "role": "user",
                    "content": f"Using the numbers {initial_numbers}, create an equation that equals {target}. Show your thought steps in <think> </think> tags. And return the final equation and conclusion in <answer> </answer> tags. For example:\n{response}"
            },
                {
                    "role": "assistant",
                    "content": "Let me solve this step by step.\n"
            }]
            
            return {
                'prompt': self.tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True),
                'target': target,
                'initial_numbers': initial_numbers,
                'reference_completion': response
            }

        hf_dataset = [format_cot_for_grpo(x) for x in dataset]

        # hf_dataset = HFDataset.from_list(dataset)

        def _safe_parse_numbers(num_str):
            try: return json.loads(num_str)
            except (json.JSONDecodeError, TypeError): return None
        
        # Unified reward function with enhanced process reward
        def unified_reward_function(completions, **kwargs):
            rewards = []
            
            # Hyperparameters from config for shaping
            use_ppl_shaping = self.config['train_conf'].get('use_ppl_reward_shaping', False)
            ppl_alpha = self.config['train_conf'].get('ppl_reward_alpha', 0.01)

            # --- NEW: Hyperparameters for Positional shaping ---
            use_pos_shaping = self.config['train_conf'].get('use_positional_reward_shaping', False)
            pos_gamma = self.config['train_conf'].get('positional_reward_gamma', 0.1)
            pos_m = self.config['train_conf'].get('positional_reward_m_scaler', 15.0)
            pos_n = self.config['train_conf'].get('positional_reward_n_shifter', -0.5)

            # Get PPLs and other metadata passed from the dataset
            ppls = kwargs.get('ppl', [0.0] * len(completions))
            targets = kwargs.get('target', [-1] * len(completions))
            initial_numbers_list = kwargs.get('initial_numbers', [[]] * len(completions))

            STEP_PATTERN = r"Step \d+:.*?numbers (\[.*?\]),.*?calculate (.*?),.*?gives (-?\d+).*?now (\[.*?\])"

            for i, completion in enumerate(completions):
                # 1. Outcome Reward (Correctness)
                outcome_reward = 0.0
                answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
                if answer_match:
                    answer_text = answer_match.group(1).strip()
                    result_match = re.search(r'=\s*(-?\d+)', answer_text)
                    if result_match and int(result_match.group(1)) == targets[i]:
                        outcome_reward = 1.0
                logger.info(f"Completion {i}: Outcome reward: {outcome_reward}, Target: {targets[i]}")
                
                # 2. Structural Reward (Format)
                structural_reward = 0.0
                if re.search(r"<think>(.*?)</think>", completion, re.DOTALL): structural_reward += 0.5
                if answer_match: structural_reward += 0.5
                logger.info(f"Completion {i}: Structural reward: {structural_reward}")

                # 3. Process Reward (Reasoning Steps) - Enhanced with validation like in qwen_mcts3.py
                process_reward = 0.0
                num_valid_steps = 0
                steps = []
                think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
                if think_match:
                    think_text = think_match.group(1)
                    steps = re.findall(STEP_PATTERN, think_text, re.DOTALL | re.IGNORECASE)
                    current_numbers = initial_numbers_list[i].copy()
                    for step_initial, step_calc, step_result_str, next_numbers_str in steps:
                        is_step_valid = False
                        try:
                            step_initial_numbers = _safe_parse_numbers(step_initial)
                            if sorted(step_initial_numbers) != sorted(current_numbers):
                                logger.debug(f"Step invalid: initial numbers {step_initial_numbers} do not match current {current_numbers}")
                                continue
                            step_result = int(step_result_str.strip())
                            next_numbers = _safe_parse_numbers(next_numbers_str)
                            calc_match = re.match(r'(\d+)\s*([\+\-\*\/])\s*(\d+)', step_calc.strip())
                            if calc_match:
                                num1 = int(calc_match.group(1))
                                op_symbol = calc_match.group(2)
                                num2 = int(calc_match.group(3))
                                operands = [num1, num2]
                                temp_num_list = current_numbers.copy()
                                
                                # A more robust check for operands availability
                                op1_found = False
                                for k, num in enumerate(temp_num_list):
                                    if num == operands[0]:
                                        temp_num_list.pop(k)
                                        op1_found = True
                                        break
                                
                                op2_found = False
                                if op1_found:
                                    for k, num in enumerate(temp_num_list):
                                        if num == operands[1]:
                                            op2_found = True
                                            break
                                
                                if op1_found and op2_found:
                                    calculated_value = eval(f"{num1} {op_symbol} {num2}")
                                    if abs(calculated_value - step_result) < 1e-6:
                                        # Reconstruct expected next numbers carefully
                                        temp_num_list_for_next = initial_numbers_list[i].copy()
                                        temp_num_list_for_next.remove(num1)
                                        temp_num_list_for_next.remove(num2)
                                        expected_next_numbers = sorted(temp_num_list_for_next + [step_result])
                                        if next_numbers and sorted(next_numbers) == expected_next_numbers:
                                            is_step_valid = True
                                            current_numbers = next_numbers
                        except Exception as e:
                            logger.debug(f"Step parsing error: {e}")
                            continue
                        if is_step_valid:
                            num_valid_steps += 1
                    if steps:
                        process_reward = min(1.0, num_valid_steps / len(steps))
                    elif len(think_text.strip()) > 20:
                        process_reward = 0.1  # Encourage non-empty reasoning
                logger.info(f"Completion {i}: Process reward: {process_reward}")

                # 4. Length Bonus (Encouraging longer reasoning)
                length_bonus = min(1.0, num_valid_steps / 5.0)
                logger.info(f"Completion {i}: Length bonus: {length_bonus}")

                # Combine base rewards with weights
                reward_weights = self.config['train_conf'].get('reward_weights', [1.0, 0.2, 1.0, 0.1])
                base_reward = (reward_weights[0] * outcome_reward +
                               reward_weights[1] * structural_reward +
                               reward_weights[2] * process_reward +
                               reward_weights[3] * length_bonus)

                final_reward = base_reward
                
                # PPL-based Reward Shaping
                if use_ppl_shaping and outcome_reward > 0.5:  # Only shape rewards for correct paths
                    ppl_weight = np.log(ppls[i] + 1e-6)
                    shaping_factor = 1.0 - ppl_alpha * ppl_weight
                    final_reward *= shaping_factor
                    logger.info(f"PPL shaping applied. PPL={ppls[i]:.2f}, factor={shaping_factor:.2f}, reward: {base_reward:.2f} -> {final_reward:.2f}")

                # --- NEW LOGIC START ---
                # Positional Reward Shaping
                if use_pos_shaping:
                    token_ids = self.tokenizer(completion, return_tensors='pt').input_ids[0]
                    seq_len = len(token_ids)
                    if seq_len > 1:
                        # Create a tensor of token positions on the same device as the model
                        positions = torch.arange(seq_len, device=self.model.device)
                        # Calculate relative positions [0, 1]
                        relative_positions = positions / (seq_len - 1)
                        
                        # Calculate position score `r_it = m * (l_it - n)` from the paper
                        r_it = pos_m * (relative_positions + pos_n)
                        # Calculate bonus `b_it = gamma * sigmoid(r_it)`
                        token_bonuses = pos_gamma * torch.sigmoid(r_it)
                        
                        # Average the bonus over the entire sequence
                        avg_bonus = token_bonuses.mean().item()
                        
                        # Apply bonus: add for successful sequences, subtract for failed ones
                        sign = 1.0 if outcome_reward > 0.5 else -1.0
                        positional_bonus = sign * avg_bonus
                        final_reward += positional_bonus
                        logger.info(f"Positional shaping applied. Avg bonus={avg_bonus:.4f}, sign={sign}, reward: {final_reward-positional_bonus:.2f} -> {final_reward:.2f}")
                # --- NEW LOGIC END ---

                rewards.append(final_reward)
            return rewards
        
        work_dir = self.config['work_dir']
        training_args = GRPOConfig(
            output_dir=f'{work_dir}epoch={epoch}',
            per_device_train_batch_size=train_param['per_device_train_batch_size'],
            gradient_accumulation_steps=train_param['gradient_accumulation_steps'],
            learning_rate=train_param['lr'],
            num_train_epochs=train_param['num_train_epochs'],
            logging_steps=train_param['logging_steps'],
            optim=train_param["optim"],
            save_strategy="epoch",
            max_prompt_length=512,
            max_completion_length=1024,
            num_generations=train_param['num_generations'],
            remove_unused_columns=False,  # Important to keep our custom columns like 'ppl'
        )

        trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=unified_reward_function,  # Changed from reward_funcs to reward_func
            args=training_args,
            train_dataset=hf_dataset,
            peft_config=lora_config,
            processing_class=self.tokenizer, # Changed from processing_class to tokenizer
        )

        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(hf_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        save_path = os.path.join(work_dir, f"epoch_{epoch}_final_model")
        trainer.model.merge_and_unload().save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        self.model = AutoModelForCausalLM.from_pretrained(save_path)  # Reload merged model
        logger.info(f"Model saved to {save_path}")


# MCTS核心逻辑
class MCTSEnhancedReasoner:
    def __init__(self, config=None):
        self.mcts_conf = config['mcts_conf']
        self.sample_conf = config['sample_conf']
        self.inf_conf = config['inf_conf']
        self.simulate_steps = self.mcts_conf['simulate_steps']
        self.iterations = self.mcts_conf['iterations']
        self.c = self.mcts_conf['c']
        self.rule_base = True
        self.llm_base = False
        self.config = config  # Pass config for CoT creation

    def select(self, node, llm_reasoner):
        def _ucb(child, parent_visits):
            if child.visits == 0:
                return float('inf') if self.c > 0 else 0
            exploit = child.value / child.visits
            explore = self.c * np.sqrt(2 * np.log(parent_visits) / child.visits) * child.prior
            return exploit + explore

        while node.children:
            if node.visits == 0:
                return random.choice(node.children)

            scores = [_ucb(child, node.visits) for child in node.children]
            
            max_score = -float('inf')
            max_indices = []
            for i, score in enumerate(scores):
                if score > max_score:
                    max_score = score
                    max_indices = [i]
                elif score == max_score:
                    max_indices.append(i)
            
            best_child_index = random.choice(max_indices)
            node = node.children[best_child_index]
        return node

    def expand(self, node, llm_reasoner):
        if node.env.is_terminal():
            return False
        
        model_ops = self._generate_ops_from_LLM(node.env, llm_reasoner) if self.llm_base else []
        rule_ops = node.env.get_valid_operations() if self.rule_base else []
        
        candidate_ops = list(set(model_ops + rule_ops))
        
        for op in candidate_ops:
            new_env = node.env.apply_operation(op)
            if new_env:
                child = MCTSNode(new_env, parent=node)
                child.op = op
                node.children.append(child)
        return len(node.children) > 0

    def simulate(self, node, llm_reasoner):
        current_env = node.env
        for _ in range(self.simulate_steps):
            if current_env.is_terminal():
                break
            
            if self.llm_base:
                model_ops = self._generate_ops_from_LLM(current_env, llm_reasoner)
            else:
                model_ops = []

            if self.rule_base:
                rule_ops = current_env.get_valid_operations()
            else:
                rule_ops = []

            if not model_ops and not rule_ops:
                break

            if model_ops:
                # Score LLM-generated operations based on distance reduction
                scored_ops = []
                for op in model_ops:
                    new_env = current_env.apply_operation(op)
                    if new_env:
                        dist_reduction = abs(min(current_env.current_numbers, key=lambda x: abs(x - current_env.target)) - current_env.target) - abs(min(new_env.current_numbers, key=lambda x: abs(x - new_env.target)) - new_env.target)
                        scored_ops.append((op, dist_reduction))
                
                if scored_ops:
                    scored_ops.sort(key=lambda x: x[1], reverse=True)
                    scores = np.array([s[1] for s in scored_ops])
                    scores = np.clip(scores, -100, 100)  # Prevent overflow
                    probs = np.exp(scores) / np.sum(np.exp(scores)) if np.sum(scores) != 0 else np.ones(len(scored_ops)) / len(scored_ops)
                    selected_idx = np.random.choice(len(scored_ops), p=probs)
                    op = scored_ops[selected_idx][0]
                else:
                    op = random.choice(model_ops)
            else:
                # Score rule-based operations
                scored_ops = []
                for op in rule_ops:
                    new_env = current_env.apply_operation(op)
                    if new_env:
                        dist_reduction = abs(min(current_env.current_numbers, key=lambda x: abs(x - current_env.target)) - current_env.target) - abs(min(new_env.current_numbers, key=lambda x: abs(x - new_env.target)) - new_env.target)
                        scored_ops.append((op, dist_reduction))
                
                if scored_ops:
                    scored_ops.sort(key=lambda x: x[1], reverse=True)
                    scores = np.array([s[1] for s in scored_ops])
                    scores = np.clip(scores, -100, 100)  # Prevent overflow
                    probs = np.exp(scores) / np.sum(np.exp(scores)) if np.sum(scores) != 0 else np.ones(len(scored_ops)) / len(scored_ops)
                    selected_idx = np.random.choice(len(scored_ops), p=probs)
                    op = scored_ops[selected_idx][0]
                else:
                    op = random.choice(rule_ops)
            
            new_env = current_env.apply_operation(op)
            if new_env is None:
                break
            current_env = new_env
        
        return current_env.evaluate_state()

    def backpropagate(self, node, value):
        while node:
            node.visits += 1
            node.value += value
            node = node.parent

    def _generate_ops_from_LLM(self, current_env, llm_reasoner):
        prompt = self._build_prompt_for_op(current_env)
        inf_conf = {**self.inf_conf, "num_return_sequences": 1}
        outputs = llm_reasoner.generate(prompt, **inf_conf)
        return self._parse_model_output_for_op(outputs, current_env)

    def _build_prompt_for_op(self, env):
        prompt_template = self.inf_conf.get('policy_prompt', "Given numbers {numbers} and target {target}, suggest the next operation (e.g., '+ 0 1' for adding numbers at indices 0 and 1). Return in <answer> </answer> tags.")
        return prompt_template.format(
            numbers=env.current_numbers, target=env.target, len_num=len(env.current_numbers)
        )

    def _parse_model_output_for_op(self, outputs, current_env):
        operations = []
        for text in outputs:
            answer_pattern = r"<answer>(.*?)<\/answer>"
            answer_content_list = re.findall(answer_pattern, text, re.DOTALL)
            for ans in answer_content_list:
                pattern = r"([\+\-\*\/])\s+(\d+)\s+(\d+)"
                matches = re.findall(pattern, ans)
                for op_symbol, index1_str, index2_str in matches:
                    index1, index2 = int(index1_str), int(index2_str)
                    if index1 < len(current_env.current_numbers) and index2 < len(current_env.current_numbers):
                        op = current_env.evaluate_operation(op_symbol, index1, index2)
                        if op: operations.append(op)
        return operations

    def solve(self, llm_reasoner, problem_env):
        root = MCTSNode(problem_env)
        for _ in range(self.iterations):
            node = self.select(root, llm_reasoner)
            
            if not node.env.is_terminal() and node.visits == 0:
                expanded = self.expand(node, llm_reasoner)
                if expanded and node.children:
                    node_to_sim = random.choice(node.children)
                    value = self.simulate(node_to_sim, llm_reasoner)
                    self.backpropagate(node_to_sim, value)
                else:
                    value = self.simulate(node, llm_reasoner)
                    self.backpropagate(node, value)
            else:
                value = self.simulate(node, llm_reasoner)
                self.backpropagate(node, value)
                
        return root

    def extract_all_paths(self, root):
        if not root: return []
        all_paths = []
        stack = [(root, [])]
        while stack:
            node, path = stack.pop()
            current_path = path + [(node, node.op)]
            if not node.children:  # It's a leaf node
                is_success = (node.env.is_terminal() and node.env.target in node.env.current_numbers)
                all_paths.append((current_path, is_success))
            for child in reversed(node.children):
                stack.append((child, current_path))
        return all_paths

    def _create_chain_of_thought_sample(self, path, initial_env, is_success):
        prompt = f"Using the numbers {initial_env.initial_numbers}, create an equation that equals {initial_env.target}."
        thought_steps = []
        for i in range(len(path) - 1):
            node = path[i][0]
            op = path[i+1][0].op
            if op:
                op_symbol, idx1, idx2, result = op
                num1, num2 = node.env.current_numbers[idx1], node.env.current_numbers[idx2]
                thought_steps.append(f"Step {i+1}: Starting with numbers {node.env.current_numbers}, "
                                    f"I choose to calculate {num1} {op_symbol} {num2}, which gives {result}. "
                                    f"The remaining numbers are now {path[i+1][0].env.current_numbers}.")
        final_thought = "\n".join(thought_steps)
        final_node = path[-1][0]
        final_result = final_node.env.current_numbers[0] if final_node.env.current_numbers else 'N/A'
        conclusion = (f"The final equation is {final_result}, which matches the target {initial_env.target}. The solution is correct." if is_success else
                      f"The final equation is {final_result}, which does not match the target {initial_env.target}. This is an incorrect path.")
        response = f"<think>\n{final_thought}\n</think>\n<answer>\n{conclusion}\n</answer>"
        return {"prompt": prompt, "response": response, "target": initial_env.target, "initial_numbers": initial_env.initial_numbers}


def benchmark(llm_reasoner, test_cases, k_samples):
    """
    Rewritten benchmark function to calculate Pass@1 and Pass@k.
    """
    logger.info(f"--- Starting Benchmark (Pass@1 and Pass@{k_samples}) ---")
    pass_at_1_count = 0
    pass_at_k_count = 0
    total_cases = len(test_cases['nums'])

    for i, (numbers, target) in enumerate(zip(test_cases['nums'], test_cases['target'])):
        logger.info(f">>> Test case {i+1}/{total_cases}: N={numbers}, T={target}")
        prompt = llm_reasoner.generate_prompt(numbers, target)
        
        found_correct = False
        for j in range(k_samples):
            result = llm_reasoner.generate_and_parse_answer(prompt)
            if result is not None and result == target:
                logger.info(f"  Attempt {j+1}/{k_samples}: Success! (Result: {result})")
                found_correct = True
                if j == 0:
                    pass_at_1_count += 1
                break  # Found a solution, move to next test case
            else:
                logger.debug(f"  Attempt {j+1}/{k_samples}: Failed. (Result: {result})")
        
        if found_correct:
            pass_at_k_count += 1
        else:
            logger.info(f"  All {k_samples} attempts failed for this test case.")

    pass_at_1_accuracy = pass_at_1_count / total_cases if total_cases > 0 else 0.0
    pass_at_k_accuracy = pass_at_k_count / total_cases if total_cases > 0 else 0.0
    
    logger.info("--- Benchmark Results ---")
    logger.info(f"Pass@1 Accuracy: {pass_at_1_accuracy:.4f} ({pass_at_1_count}/{total_cases})")
    logger.info(f"Pass@{k_samples} Accuracy: {pass_at_k_accuracy:.4f} ({pass_at_k_count}/{total_cases})")
    logger.info("-----------------------")
    return {"pass@1": pass_at_1_accuracy, f"pass@{k_samples}": pass_at_k_accuracy}


def main(config):
    work_dir = config['work_dir']
    problem_set = datasets.load_dataset(config['countdown_conf']['problem_path'])
    train_split = problem_set.get('train', list(problem_set.values())[0])
    
    shuffled_split = train_split.shuffle(seed=config['seed'])
    
    benchmark_size = config['countdown_conf']['benchmark_problem_size']
    test_set = shuffled_split.select(range(benchmark_size))
    train_set = shuffled_split.select(range(benchmark_size, len(shuffled_split)))

    llm_reasoner = LLMReasoner(config)
    mcts_solver = MCTSEnhancedReasoner(config)

    for epoch in range(config['mcts_conf']['num_rollout']):
        logger.info(f">>>>>> EPOCH {epoch} begin")
        
        cot_samples_for_grpo = []
        num_samples_needed = config['train_conf']['num_samples_per_epoch']
        
        epoch_train_set = train_set.shuffle(seed=config['seed'] + epoch).select(range(num_samples_needed * 2))  # Oversample
        
        mcts_solver.rule_base = epoch < config['train_conf']['num_rulebase_only_epochs']
        mcts_solver.llm_base = not mcts_solver.rule_base
        
        for step, example in enumerate(epoch_train_set):
            if len(cot_samples_for_grpo) >= num_samples_needed:
                break
            
            numbers, target = example['nums'], example['target']
            problem_env = CountdownEnv(numbers, target)
            logger.info(f"EPOCH {epoch} STEP {step+1}: Solving for N={numbers}, T={target}")
            
            root = mcts_solver.solve(llm_reasoner, problem_env)
            
            all_paths = mcts_solver.extract_all_paths(root)
            for path, is_success in all_paths:
                cot_sample = mcts_solver._create_chain_of_thought_sample(path, problem_env, is_success)
                cot_samples_for_grpo.append(cot_sample)
        
        logger.info(f"Collected {len(cot_samples_for_grpo)} CoT samples from MCTS.")
        
        strategy = config['train_conf'].get('data_selection_strategy', 'none')
        selection_pct = config['train_conf'].get('data_selection_percent', 1.0)
        
        if strategy != 'none' and len(cot_samples_for_grpo) > 1:
            logger.info(f"Applying data selection strategy: '{strategy}' with {selection_pct*100}% selection.")
            
            cot_samples_for_grpo = llm_reasoner.calculate_metrics_for_samples(cot_samples_for_grpo)
            
            if strategy == 'ppl':
                cot_samples_for_grpo.sort(key=lambda x: x['ppl'])
            elif strategy == 'entropy':
                cot_samples_for_grpo.sort(key=lambda x: x['entropy'], reverse=True)
            
            num_to_select = int(len(cot_samples_for_grpo) * selection_pct)
            selected_samples = cot_samples_for_grpo[:num_to_select]
            logger.info(f"Selected {len(selected_samples)} samples out of {len(cot_samples_for_grpo)} for training.")
        else:
            selected_samples = cot_samples_for_grpo
            logger.info("No data selection strategy applied, using all collected samples.")

        with open(f'{work_dir}cot_samples_epoch{epoch}.json', 'w') as f:
            json.dump(selected_samples, f, indent=2)

        if selected_samples:
            logger.info(f"Starting GRPO training for epoch {epoch}...")
            llm_reasoner.GRPO_trl_wrapper(selected_samples, epoch)
            logger.info(f"GRPO training for epoch {epoch} finished.")
        else:
            logger.warning(f"No samples for GRPO in epoch {epoch}, skipping training.")
        
        if config['countdown_conf']['evaluate_strategy'] == 'epoch':
            k_samples = config['countdown_conf'].get('benchmark_k_samples', 10)
            # For benchmark, always use the LLM
            # mcts_solver.rule_base = False
            # mcts_solver.llm_base = True
            benchmark(llm_reasoner, test_set, k_samples)

        logger.info(f"<<<<<< EPOCH {epoch} end")


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def save_config(config):
    with open(f"{config['work_dir']}config.yaml", 'w') as file:
        yaml.safe_dump(config, file, default_flow_style=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='configuration file path')
    args = parser.parse_args()

    config = load_config(args.config_path)
    time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    config['work_dir'] = config['work_dir'].format(model_name=config['model_name'].replace("/", "_"), ts=time_stamp)
    os.makedirs(config['work_dir'], exist_ok=True)
    save_config(config)

    set_seed(config['seed'])

    logger_filename = f"{config['work_dir']}train.log"
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logger_filename)],
                        level=logging.INFO if 'INFO' in config['log_level'].upper() else logging.DEBUG)
    logger.info(f"Configuration: {config}")

    main(config)
