## Journal (9_1~9_5)

### 一. 奖励函数设计
``` python
def unified_reward_function(completions, **kwargs):
    rewards = []
    
    # Hyperparameters from config for shaping
    use_ppl_shaping = self.config['train_conf'].get('use_ppl_reward_shaping', False)
    ppl_alpha = self.config['train_conf'].get('ppl_reward_alpha', 0.01)
    
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
        
        # 2. Structural Reward (Format)
        structural_reward = 0.0
        if re.search(r"<think>(.*?)</think>", completion, re.DOTALL): structural_reward += 0.5
        if answer_match: structural_reward += 0.5

        # 3. Process Reward (Reasoning Steps) - Enhanced with validation
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
                        operands_available = all(op in temp_num_list for op in operands)
                        if operands_available:
                            calculated_value = eval(f"{num1} {op_symbol} {num2}")
                            if abs(calculated_value - step_result) < 1e-6:
                                expected_next_numbers = sorted([n for n in temp_num_list if n not in operands] + [step_result])
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
        
        # 4. Length Bonus (Encouraging longer reasoning)
        length_bonus = min(1.0, len(completion) / 1024.0)

        # Combine base rewards with weights
        reward_weights = self.config['train_conf'].get('reward_weights', [1.0, 0.2, 1.0, 0.1])
        base_reward = (reward_weights[0] * outcome_reward +
                       reward_weights[1] * structural_reward +
                       reward_weights[2] * process_reward +
                       reward_weights[3] * length_bonus)

        # PPL-based Reward Shaping
        final_reward = base_reward
        if use_ppl_shaping and outcome_reward > 0.5:  # Only shape rewards for correct paths
            ppl_weight = np.log(ppls[i] + 1e-6)
            shaping_factor = 1.0 - ppl_alpha * ppl_weight
            final_reward *= shaping_factor
            logger.debug(f"PPL shaping applied. PPL={ppls[i]:.2f}, factor={shaping_factor:.2f}, reward: {base_reward:.2f} -> {final_reward:.2f}")

        rewards.append(final_reward)
    return rewards
```
1.引入 PPL-based Reward Shaping，对正确路径根据困惑度进行奖励调节
公式：final_reward = base_reward * (1 - ppl_alpha * log(ppl))
### 二. 改进的 MCTS 模拟策略
``` python
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
```
1.引入操作评分机制替代随机选择:
- 计算每个操作带来的目标距离减少量：dist_reduction = abs(current_closest - target) - abs(new_closest - target)
- 使用 softmax 函数将距离减少量转换为选择概率
- 防止数值溢出：scores = np.clip(scores, -100, 100)
- 优势：使模拟过程更加导向目标，提高搜索效率
2.使用栈结构替代递归提取所有路径
- 优势：避免递归深度限制，提高大规模树搜索的稳定性
### 三. 数据选择和质量评估
``` python
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
```
1.引入 PPL 和熵作为样本质量指标：
- 计算响应部分的困惑度：ppl = torch.exp(avg_loss).item()
- 计算响应部分的平均熵：entropy = torch.distributions.Categorical(logits=response_logits).entropy().mean().item()
- 支持多种选择策略：'ppl'（低困惑度优先）或 'entropy'（高熵优先）
- 优势：筛选高质量训练样本，提高训练效率
### 四. SFT + GRPO 混合训练策略
``` python
# 在主训练循环中
strategy = config['train_conf'].get('data_selection_strategy', 'none')
selection_pct = config['train_conf'].get('data_selection_percent', 1.0)
use_sft = epoch < config['train_conf'].get('sft_epochs', 2)  # 前2个epoch使用SFT

if use_sft:
    logger.info(f"使用SFT训练，仅选择成功且高质量样本 (epoch {epoch})")
    # 1. 筛选成功样本
    success_samples = [s for s in cot_samples_for_grpo if s['is_success']]
    logger.info(f"成功样本数: {len(success_samples)}/{len(cot_samples_for_grpo)}")
    
    if not success_samples:
        logger.warning("无成功样本，跳过本轮训练")
        continue
        
    # 2. 计算质量指标
    success_samples = llm_reasoner.calculate_metrics_for_samples(success_samples)
    
    # 3. 按PPL排序选择高质量样本
    success_samples.sort(key=lambda x: x['ppl'])  # 低PPL优先
    num_to_select = int(len(success_samples) * selection_pct)
    selected_samples = success_samples[:num_to_select]
    logger.info(f"选择 {len(selected_samples)} 个高质量成功样本进行SFT训练")
    
    # 4. 执行SFT训练
    llm_reasoner.sft_finetune(selected_samples, epoch)
```
1.前几个 epoch 使用 SFT，后续使用 GRPO：
- 仅选择成功路径进行 SFT 训练
- 按 PPL 排序选择高质量样本
- 配置参数控制 SFT 训练轮次：sft_epochs
- 优势：先学习高质量解决方案的模式，再通过强化学习优化策略
### 五. 改进的评估函数 (Pass@k)
``` python
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
```
1.从单一 Pass@1 扩展到 Pass@k 评估：
- 对每个测试案例进行 k 次生成尝试
- 记录第一次尝试成功（Pass@1）和任意一次尝试成功（Pass@k）
- 配置参数控制 k 值：benchmark_k_samples
- 优势：更全面评估模型的多样性和可靠性

### 问题

### 9_1~9_5 安排

