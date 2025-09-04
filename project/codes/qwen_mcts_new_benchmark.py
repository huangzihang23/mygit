import logging
import argparse
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling, set_seed
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, GRPOTrainer, GRPOConfig
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
import datasets

import numpy as np
from collections import defaultdict
import random
import re
import yaml
import json

logger = logging.getLogger(__name__)

# -------------------------- 1. 原有核心类保留，新增指标计算逻辑 --------------------------
class CountdownEnv:
    # （原代码不变，省略以精简篇幅）
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
        # （原代码不变）
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
        # （原代码不变）
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
        # （原代码不变）
        i, j, res = op[1], op[2], op[3]
        if res is not None and res > 0 and isinstance(res, int):
            new_numbers = [n for idx, n in enumerate(self.current_numbers) if idx not in (i, j)]
            new_numbers.append(int(res))
            return CountdownEnv(new_numbers, self.target)
        return None
    def __repr__(self):
        return f"State(numbers={self.current_numbers}, target={self.target})"

class MCTSNode:
    # （原代码不变）
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

# -------------------------- 2. 优化LLMReasoner：新增指标计算方法 --------------------------
class LLMReasoner:
    def __init__(self, config=None):
        self.model = AutoModelForCausalLM.from_pretrained(
            config['model_path'],
            output_hidden_states=False,
            output_attentions=False,
            output_scores=True  # 启用scores输出，用于计算分支因子
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config = config
        self.device = self.model.device

    @torch.no_grad()
    def generate(self, prompt=f'''Hello world!''', **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True).to(self.device)
        kwargs_ = {**self.config['inf_conf'], **kwargs}
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=kwargs_['max_new_tokens'],
            num_return_sequences=kwargs_['num_return_sequences'],
            do_sample=True,
            temperature=kwargs_['temperature'],
            top_k=kwargs_['top_k'],
            output_scores=True,  # 保留scores用于分支因子计算
            return_dict_in_generate=True  # 结构化输出
        )
        # 返回解码文本和scores（用于分支因子）
        texts = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        return texts, outputs.scores

    def generate_prompt(self, initial_numbers: list, target: int) -> str:
        """Generate a prompt for the Countdown game based on initial numbers and target.

        Args:
            initial_numbers: List of integers (e.g., [38, 29, 58, 56]).
            target: Target integer to achieve (e.g., 1102).

        Returns:
            str: Formatted prompt string for the model.
        """
        r1_prefix = [{
            "role": "system",
            "content": "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer."
        }, {
            "role": "user",
            "content": f"Using the numbers {initial_numbers}, create an equation that equals {target}. Show your thought steps in </think> </think> tags. And return the final equation and conclusion in <answer> </answer> tags. For example, </think>\nStep 1: Starting with numbers [38, 29, 58, 56], I choose to calculate 38 * 29, which gives 1102. The remaining numbers are now [58, 56, 1102].\nStep 2: Starting with numbers [58, 56, 1102], I choose to calculate 56 + 1102, which gives 1158. The remaining numbers are now [58, 1158].\nStep 3: Starting with numbers [58, 1158], I choose to calculate 58 + 1158, which gives 1216. The remaining numbers are now [1216].\n</think>\n<answer>\nThe final equation is (58 + (56 + (38 * 29))) = 1216, which does not match the target 75. This is an incorrect path.\n</answer>"
        }, {
            "role": "assistant",
            "content": "Let me solve this step by step.\n"
        }]
        return self.tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True)

    @torch.no_grad()
    def generate_and_parse_answer(self, prompt: str) -> int:
        # （原代码不变，仅返回解析的数值结果）
        try:
            outputs, _ = self.generate(prompt, num_return_sequences=1, max_new_tokens=256, temperature=0.7, top_k=30)
            output = outputs[0]
            answer_match = re.search(r"(.*?)<|FunctionCallEnd|>", output, re.DOTALL)
            if not answer_match:
                logger.debug(f"No  tag found in output: {output}")
                return None
            answer_text = answer_match.group(1).strip()
            result_match = re.search(r'(-?\d+)', answer_text)
            if not result_match:
                logger.debug(f"No numerical result found in answer: {answer_text}")
                return None
            return int(result_match.group(1))
        except Exception as e:
            logger.debug(f"Error generating or parsing answer: {e}")
            return None

    # -------------------------- 新增：1. 无偏Pass@k计算 --------------------------
    def compute_unbiased_pass_k(self, prompt: str, target: int, n: int = 64, k: int = 8) -> dict:
        """
        计算无偏Pass@k（文档2公式2）
        Args:
            prompt: 问题prompt
            target: 目标值
            n: 总生成次数（需n >= k）
            k: 评估窗口大小
        Returns:
            dict: 包含无偏Pass@k、正确响应数c、总次数n
        """
        if n < k:
            logger.warning(f"n={n} < k={k}，无法计算无偏Pass@k，自动调整n={k}")
            n = k
        
        # 生成n次响应并统计正确数c
        results = []
        for _ in range(n):
            pred = self.generate_and_parse_answer(prompt)
            results.append(1 if (pred is not None and pred == target) else 0)
        c = sum(results)
        
        # 计算无偏Pass@k：1 - C(n-c, k)/C(n, k)（c=0时结果为0）
        if c == 0:
            unbiased_pass_k = 0.0
        else:
            # 组合数计算：C(a, b)若a < b则为0
            comb_n_c_k = comb(n - c, k) if (n - c >= k) else 0
            comb_n_k = comb(n, k)
            unbiased_pass_k = 1.0 - (comb_n_c_k / comb_n_k)
        
        return {
            "unbiased_pass_k": round(unbiased_pass_k, 4),
            "c": c,  # 正确响应数
            "n": n,  # 总生成数
            "k": k   # 评估窗口
        }

    # -------------------------- 新增：2. k-rollout不可解问题统计 --------------------------
    def get_k_rollout_unsolvable(self, test_cases: dict, k: int = 64) -> dict:
        """
        统计k-rollout不可解问题（文档2 Section 2.1）
        Args:
            test_cases: 测试用例，格式{"nums": [[a,b,c...], ...], "target": [t1, t2, ...]}
            k: 每个问题的生成次数
        Returns:
            dict: 不可解问题列表、不可解比例、统计信息
        """
        unsolvable_problems = []
        total = len(test_cases['nums'])
        
        for idx, (numbers, target) in enumerate(zip(test_cases['nums'], test_cases['target'])):
            prompt = self.generate_prompt(numbers, target)
            # 生成k次，判断是否全错
            all_wrong = True
            for _ in range(k):
                pred = self.generate_and_parse_answer(prompt)
                if pred is not None and pred == target:
                    all_wrong = False
                    break
            if all_wrong:
                unsolvable_problems.append({
                    "index": idx,
                    "numbers": numbers,
                    "target": target
                })
        
        unsolvable_ratio = len(unsolvable_problems) / total if total > 0 else 0.0
        return {
            "unsolvable_problems": unsolvable_problems,
            "unsolvable_count": len(unsolvable_problems),
            "total_count": total,
            "unsolvable_ratio": round(unsolvable_ratio, 4),
            "k": k
        }

    # -------------------------- 新增：3. Rollout分支因子计算 --------------------------
    def compute_rollout_branching_factor(self, prompt: str, max_new_tokens: int = 128) -> dict:
        """
        计算Rollout分支因子（文档2 Section 2.2）：top 95%概率质量内的候选token数
        Args:
            prompt: 输入prompt
            max_new_tokens: 生成的token长度
        Returns:
            dict: 平均分支因子、各token位置的分支因子列表
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True).to(self.device)
        # 生成并获取每个新token的scores（logits）
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=False,  # 关闭采样以获取确定的概率分布
            output_scores=True,
            return_dict_in_generate=True
        )
        # 转换scores为概率（softmax），并计算每个位置的分支因子
        branching_factors = []
        for score in outputs.scores:  # 每个score对应一个新token的logits (1, V)
            # logits -> 概率
            probs = torch.softmax(score, dim=-1).squeeze(0).cpu().numpy()  # (V,)
            # 降序排序概率
            sorted_probs = np.sort(probs)[::-1]
            # 累积概率到95%
            cumulative_probs = np.cumsum(sorted_probs)
            # 找到累积概率首次 >= 0.95的索引
            idx_95 = np.argmax(cumulative_probs >= 0.95) + 1  # +1是因为索引从0开始
            branching_factors.append(idx_95)
        
        avg_branching_factor = np.mean(branching_factors) if branching_factors else 0.0
        return {
            "avg_branching_factor": round(avg_branching_factor, 2),
            "branching_factors_per_token": branching_factors,  # 每个token位置的分支因子
            "max_new_tokens": max_new_tokens
        }

# -------------------------- 3. 优化Benchmark函数：整合新指标 --------------------------
def benchmark_enhanced(llm_reasoner: LLMReasoner, test_cases: dict, pass_k_n: int = 64, pass_k_k: int = 8, rollout_k: int = 64, branching_max_tokens: int = 128) -> dict:
    """
    增强版Benchmark：同时输出无偏Pass@k、k-rollout不可解问题、Rollout分支因子
    """
    # 1. 无偏Pass@k（全局平均）
    pass_k_results = []
    for numbers, target in zip(test_cases['nums'], test_cases['target']):
        prompt = llm_reasoner.generate_prompt(numbers, target)
        pass_k = llm_reasoner.compute_unbiased_pass_k(prompt, target, n=pass_k_n, k=pass_k_k)
        pass_k_results.append(pass_k)
    avg_unbiased_pass_k = np.mean([r["unbiased_pass_k"] for r in pass_k_results])

    # 2. k-rollout不可解问题
    unsolvable_result = llm_reasoner.get_k_rollout_unsolvable(test_cases, k=rollout_k)

    # 3. Rollout分支因子（随机选10个测试用例计算平均）
    branching_results = []
    sample_indices = random.sample(range(len(test_cases['nums'])), min(10, len(test_cases['nums'])))
    for idx in sample_indices:
        numbers = test_cases['nums'][idx]
        target = test_cases['target'][idx]
        prompt = llm_reasoner.generate_prompt(numbers, target)
        branching = llm_reasoner.compute_rollout_branching_factor(prompt, max_new_tokens=branching_max_tokens)
        branching_results.append(branching["avg_branching_factor"])
    overall_avg_branching = np.mean(branching_results) if branching_results else 0.0

    # 输出汇总日志
    logger.info("=" * 50)
    logger.info("Enhanced Benchmark Results")
    logger.info("=" * 50)
    logger.info(f"1. Average Unbiased Pass@{pass_k_k} (n={pass_k_n}): {round(avg_unbiased_pass_k, 4)}")
    logger.info(f"2. k-rollout Unsolvable Problems (k={rollout_k}):")
    logger.info(f"   - Total Test Cases: {unsolvable_result['total_count']}")
    logger.info(f"   - Unsolvable Count: {unsolvable_result['unsolvable_count']}")
    logger.info(f"   - Unsolvable Ratio: {unsolvable_result['unsolvable_ratio']:.4f}")
    logger.info(f"3. Average Rollout Branching Factor: {round(overall_avg_branching, 2)}")
    logger.info("=" * 50)

    return {
        "avg_unbiased_pass_k": round(avg_unbiased_pass_k, 4),
        "unsolvable_result": unsolvable_result,
        "avg_branching_factor": round(overall_avg_branching, 2)
    }

# -------------------------- 4. 原有MCTSEnhancedReasoner、main等方法不变，修改main调用增强Benchmark --------------------------
class MCTSEnhancedReasoner:
    # （原代码不变，省略以精简篇幅）
    def __init__(self, config=None):
        self.mcts_conf = config['mcts_conf']
        self.sample_conf = config['sample_conf']
        self.inf_conf = config['inf_conf']
        self.simulate_steps = self.mcts_conf['simulate_steps']
        self.iterations = self.mcts_conf['iterations']
        self.c = self.mcts_conf['c']
        self.rule_base = True
        self.llm_base = False
    # （原有select、expand、simulate、backpropagate、solve等方法不变）

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def save_config(config):
    with open(f"{config['work_dir']}config.yaml", 'w') as file:
        yaml.safe_dump(config, file, default_flow_style=False)

def main(config):
    work_dir = config['work_dir']
    os.makedirs(work_dir, exist_ok=True)
    save_config(config)
    set_seed(config['seed'])

    # 加载数据集
    logger.info(f"Loading dataset from {config['countdown_conf']['problem_path']}")
    problem_set = datasets.load_dataset(config['countdown_conf']['problem_path'])
    problem_set['train'].shuffle()
    train_set = problem_set['train'][:-config['countdown_conf']['benchmark_problem_size']]
    test_set = problem_set['train'][-config['countdown_conf']['benchmark_problem_size']:]
    test_cases = {
        "nums": test_set['nums'],
        "target": test_set['target']
    }

    # 初始化模型和MCTS
    llm_reasoner = LLMReasoner(config)
    mcts_solver = MCTSEnhancedReasoner(config)

    # 训练循环（原逻辑不变，此处简化）
    for epoch in range(config['mcts_conf']['num_rollout']):
        logger.info(f">>>>>> EPOCH {epoch} BEGIN")
        # （原有MCTS采样、GRPO训练逻辑不变）
        
        # 每个epoch结束后执行增强Benchmark
        if config['countdown_conf']['evaluate_strategy'] == 'epoch':
            logger.info(f"Running Enhanced Benchmark for Epoch {epoch}")
            benchmark_enhanced(
                llm_reasoner=llm_reasoner,
                test_cases=test_cases,
                pass_k_n=config['countdown_conf'].get('pass_k_n', 64),
                pass_k_k=config['countdown_conf'].get('pass_k_k', 8),
                rollout_k=config['countdown_conf'].get('rollout_k', 64),
                branching_max_tokens=config['countdown_conf'].get('branching_max_tokens', 128)
            )
        logger.info(f"<<<<<< EPOCH {epoch} END")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help='configuration file path')
    args = parser.parse_args()
    config = load_config(args.config_path)

    # 初始化日志
    time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    config['work_dir'] = config['work_dir'].format(model_path=config['model_path'], ts=time_stamp)
    os.makedirs(config['work_dir'], exist_ok=True)
    logger_filename = f"{config['work_dir']}train.log"
    handler = logging.FileHandler(logger_filename)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), handler],
        level=logging.INFO
    )

    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    main(config)