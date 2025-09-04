## Journal (8_18~8_22)

### 对task1的尝试



#### 利用离散决策点(ddp)进行grpo训练，利用长思维链(cot)进行sft训练

``` python

    def extract_all_paths(self, root):
        if not root: return []
        all_paths = []
        stack = [(root, [])]
        while stack:
            node, path = stack.pop()
            current_path = path + [(node, node.op)]
            if not node.children:
                is_success = (node.env.is_terminal() and node.env.target in node.env.current_numbers)
                all_paths.append((current_path, is_success))
            for child in reversed(node.children):
                stack.append((child, current_path))
        return all_paths

    def _create_chain_of_thought_sample(self, path, initial_env, is_success):
        prompt = f"Using the numbers {initial_env.initial_numbers}, create an equation that equals {initial_env.target}."
        thought_steps = []
        equation_parts = [str(n) for n in initial_env.initial_numbers]

        # path[0]是根节点，其op是None。操作信息在下一个节点上
        for i in range(len(path) - 1):
            node = path[i][0]
            op = path[i+1][0].op
            if op is None: continue
            
            op_symbol, idx1, idx2, result = op
            num1 = node.env.current_numbers[idx1]
            num2 = node.env.current_numbers[idx2]
            
            step_description = (
                f"Step {i+1}: Starting with numbers {node.env.current_numbers}, "
                f"I choose to calculate {num1} {op_symbol} {num2}, which gives {result}. "
                f"The remaining numbers are now {path[i+1][0].env.current_numbers}."
            )
            thought_steps.append(step_description)

            operand1_str = equation_parts.pop(max(idx1, idx2))
            operand2_str = equation_parts.pop(min(idx1, idx2))
            sub_expression = f"({operand2_str} {op_symbol} {operand1_str})"
            equation_parts.append(sub_expression)

        final_thought = "\n".join(thought_steps)
        final_node = path[-1][0]
        final_result = final_node.env.current_numbers[0]
        final_equation = equation_parts[0] if equation_parts else "N/A"

        if is_success:
            conclusion = f"The final equation is {final_equation} = {final_result}, which matches the target {initial_env.target}. The solution is correct."
        else:
            conclusion = f"The final equation is {final_equation} = {final_result}, which does not match the target {initial_env.target}. This is an incorrect path."
            
        response = f"<think>\n{final_thought}\n</think>\n<answer>\n{conclusion}\n</answer>"
        return {"prompt": prompt, "response": response}

```

![full_training_analysis](.\full_training_analysis.png)



#### 设计新奖励函数，直接利用cot进行grpo训练

``` python
def comprehensive_reward_func(completions, **kwargs):
            """
            新的复合奖励函数，结合结果、过程和结构。
            """
            final_rewards = []
            
            for i in range(len(completions)):
                completion = completions[i]
                target = kwargs['target'][i]
                initial_numbers = kwargs['initial_numbers'][i]

                # 1. 结构奖励
                think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
                answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
                structural_reward = 0.1 if think_match and answer_match else 0.0

                # 2. 结果奖励
                outcome_reward = 0.0
                final_result_val = None
                if answer_match:
                    answer_text = answer_match.group(1)
                    result_match = re.search(r'=\s*(-?\d+)', answer_text)
                    if result_match:
                        final_result_val = int(result_match.group(1))
                        if final_result_val == target:
                            outcome_reward = 1.0

                # 3. 过程奖励 (步骤计算准确性 + 逻辑连贯性)
                process_reward = 0.0
                if think_match:
                    think_text = think_match.group(1)
                    steps = re.findall(r"Step \d+:.*?calculate ([\d\s\+\-\*\/]+?)= ([\d\s]+)\..*?numbers are now (\[.*?\])", think_text)
                    
                    # 检查每个提取的步骤
                    current_numbers_for_check = initial_numbers.copy()
                    valid_steps = 0
                    for step_calc, step_result, next_numbers_str in steps:
                        try:
                            # calculated_value = eval(step_calc.replace(' ', ''))
                            # if abs(calculated_value - int(step_result.strip())) < 1e-6: # 检查计算是否正确
                            #    valid_steps += 1
                        except Exception:
                            continue # eval失败或类型转换失败，说明步骤格式有问题
                    
                    # 按有效步骤的比例给予奖励
                    if len(steps) > 0:
                        process_reward = 0.8 * (valid_steps / len(steps))

                # 组合奖励
                # 最终奖励 = 结果权重 * 结果 + 过程权重 * 过程 + 结构权重 * 结构
                # 总权重为 1.0 + 0.8 + 0.1 = 1.9
                total_reward = (1.0 * outcome_reward) + process_reward + structural_reward
                final_rewards.append(total_reward)

            return final_rewards
```

---

#### 遇到的问题

+ 奖励函数设计

  > 端到端的训练很简洁，但当动作空间是生成一整段长文本时，低劣的奖励函数的方差会非常大，导致训练过程震荡，难以收敛。

+ 环境配置困难

  > scp一个conda虚拟环境（约5GB）需要4小时，但经常需要处理环境问题，如 
  >
  > > bitsandbytes.cextension - the installed version of bitsandbytes was compiled withoutGPU support. 8-bit optimizers , 8-bit multiplication, and GPU quantization are unavailable.

+ 缺乏GPU资源

  > 周四才开始有GPU节点idle





---



### 8_23~8_28 安排

| 吴浩                       | 吴君和                                            | 黄梓航                     |
| -------------------------- | ------------------------------------------------- | -------------------------- |
| 解决环境问题；设计奖励函数 | 分离冗余解，对比benchmark；对比不同参数规模的模型 | 查阅相关论文，设计奖励函数 |



