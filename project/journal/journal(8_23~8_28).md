## Journal (8_23~8_28)

### 基础奖励函数实现

``` python
def revised_reward_function(completions, **kwargs):
    final_rewards = []

    # 定义奖励权重
    W_OUTCOME = 1.0   # 结果正确性权重
    W_STRUCTURE = 0.2 # 结构完整性权重
    W_PROCESS = 1.0   # 推理过程权重
    W_LENGTH = 0.1    # 长度奖励权重

    # 修正后的正则表达式
    STEP_PATTERN = r"Step \d+:\s*Starting with numbers (\[\d+(?:,\s*\d+)*\]),\s*I choose to calculate ([\d\s\+\-\*\/()]+?),\s*which gives (-?\d+)\.\s*The remaining numbers are now (\[\d+(?:,\s*\d+)*\])"

    for i, completion in enumerate(completions):
        target = kwargs['target'][i]
        initial_numbers = kwargs['initial_numbers'][i]

        # 初始化奖励
        outcome_reward = 0.0
        structural_reward = 0.0
        process_reward = 0.0
        length_bonus = 0.0
        num_valid_steps = 0
        steps = []

        # 检查结构：<think> 和 <answer> 标签
        think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)

        if think_match:
            structural_reward += 0.5
        if answer_match:
            structural_reward += 0.5

        # 检查结果正确性
        if answer_match:
            answer_text = answer_match.group(1).strip()
            result_match = re.search(r'(-?\d+)', answer_text)
            if result_match:
                final_result_val = int(result_match.group(1))
                if final_result_val == target:
                    outcome_reward = 1.0

        # 检查推理过程
        if think_match:
            think_text = think_match.group(1)
            steps = re.findall(STEP_PATTERN, think_text, re.DOTALL | re.IGNORECASE)

            current_numbers = initial_numbers.copy()
            for step_initial, step_calc, step_result_str, next_numbers_str in steps:
                is_step_valid = False
                try:
                    # 解析初始数字列表、表达式、结果和剩余数字列表
                    step_initial_numbers = _safe_parse_numbers(step_initial)
                    if sorted(step_initial_numbers) != sorted(current_numbers):
                        logger.debug(f"Step invalid: initial numbers {step_initial_numbers} do not match current {current_numbers}")
                        continue

                    step_result = int(step_result_str.strip())
                    next_numbers = _safe_parse_numbers(next_numbers_str)
                    operands = [int(n) for n in re.findall(r'\d+', step_calc)]

                    # 验证操作数可用性
                    temp_num_list = current_numbers.copy()
                    operands_available = all(op in temp_num_list for op in operands)
                    if operands_available:
                        # 计算表达式
                        calculated_value = eval(step_calc.replace(' ', ''))
                        if abs(calculated_value - step_result) < 1e-6:
                            # 验证剩余数字列表
                            expected_next_numbers = sorted([n for n in temp_num_list if n not in operands] + [step_result])
                            if next_numbers and sorted(next_numbers) == expected_next_numbers:
                                is_step_valid = True
                                current_numbers = next_numbers
                            # 即使剩余数字列表错误，也给予部分奖励
                            process_reward += 0.5 / len(steps) if abs(calculated_value - step_result) < 1e-6 else 0.0

                except Exception as e:
                    logger.debug(f"Step parsing error: {e}")
                    continue

                if is_step_valid:
                    num_valid_steps += 1

            # 计算过程奖励
            if steps:
                process_reward = min(process_reward + num_valid_steps / len(steps), 1.0)
            elif len(think_text.strip()) > 20:
                process_reward = 0.1  # 鼓励非空推理

        # 计算长度奖励
        length_bonus = min(1.0, num_valid_steps / 5.0)

        # 总奖励
        total_reward = (W_OUTCOME * outcome_reward +
                        W_STRUCTURE * structural_reward +
                        W_PROCESS * process_reward +
                        W_LENGTH * length_bonus)

        final_rewards.append(total_reward)

    return final_rewards
```



### 更改prompt

``` python
def format_cot_for_grpo(item):
    prompt = item['prompt']
    response = item['response']


    target_match = re.search(r'equals (\d+)', prompt)
    numbers_match = re.search(r'numbers (\[.*?\])', prompt)

    target = int(target_match.group(1)) if target_match else -1
    try:
        initial_numbers = json.loads(numbers_match.group(1)) if numbers_match else []
    except (json.JSONDecodeError, AttributeError):
        initial_numbers = []

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
```



### 更改benchmark

``` python
def benchmark(llm_reasoner, test_cases):
    success_count = 0
    total_cases = len(test_cases['nums'])

    for i, (numbers, target) in enumerate(zip(test_cases['nums'], test_cases['target'])):
        logger.info(f">>>>>> Test case {i} begin, numbers: {numbers}, target: {target}")
        prompt = llm_reasoner.generate_prompt(numbers, target)
        result = llm_reasoner.generate_and_parse_answer(prompt)
        
        if result is not None and result == target:
            success_count += 1
            logger.info(f"Test case {i}: Success, result {result} matches target {target}")
        else:
            logger.info(f"Test case {i}: Failed, result {result} does not match target {target}")
        logger.info(f"<<<<<< Test case {i} end")
        
    accuracy = success_count / total_cases if total_cases > 0 else 0.0
    logger.info(f"Direct LLM Accuracy: {accuracy:.4f} ({success_count}/{total_cases})")
```



### 结果

![log_result](H:\lins\Desktop\log_result.png)

### 问题

1. 是否要更改mcts中llm的运用？

   > 目前MctsEnhance类中只是用它来产生operation，而训练的却是长思维链

2. 长思维链的pattern是否正确？

   > <think>\nStep 1: Starting with numbers [38, 29, 58, 56], I choose to calculate 38 * 29, which gives 1102. The remaining numbers are now [58, 56, 1102].\nStep 2: Starting with numbers [58, 56, 1102], I choose to calculate 56 + 1102, which gives 1158. The remaining numbers are now [58, 1158].\nStep 3: Starting with numbers [58, 1158], I choose to calculate 58 * 1158, which gives 67164. The remaining numbers are now [67164].\n</think>\n<answer>\nThe final equation is (58 * (56 + (38 * 29))) = 67164, which does not match the target 75. This is an incorrect path.\n</answer>

### 下周安排

进一步参照论文改进代码

