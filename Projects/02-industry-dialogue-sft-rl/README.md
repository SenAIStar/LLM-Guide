# 02 高质量行业对话助手（SFT + GRPO）

这里以金融客服为例，走一遍对话模型后训练：先用高质量示范回答做 SFT，再让模型针对同一个问题生成多条候选回答，由标注人员排序并训练 Reward Model，最后用 GRPO 继续优化 SFT 模型。

范围里不含知识库检索、工具调用或真实账户查询。这里只解决一件事：把业务正确、合规边界和服务表达写成可标注的偏好，再把偏好用于后训练。

示例数据全部是合成的，只用于说明字段和流程。真实金融数据必须经过授权、脱敏和业务复核。

## 1. 做完后要留下什么

做完后应保留：

- 行业任务清单、回答评分标准和标注规则版本；
- LoRA SFT 模型，以及 Base/SFT 对照结果；
- 同一 prompt 下的候选回答、人工排序和 `chosen/rejected` 偏好对；
- 能区分优劣回答的 Reward Model；
- Base、SFT、GRPO 的盲评结果，以及 reward、KL、回复长度、组内奖励方差等曲线；
- 失败样例分析：答错、过度拒答、模板投机和 reward hacking 分别发生在哪里。

## 2. 项目主线

```text
行业任务与评分规则
  -> 高质量示范回答 -> LoRA SFT -> 行业对话基线
  -> 同一问题采样 G 条回答 -> 人工排序 -> chosen / rejected
  -> Reward Model：给 prompt + answer 输出标量分数
  -> GRPO：组内相对奖励 + reference KL，不训练 critic
  -> Base / SFT / GRPO 对照、盲评和失败分析
```

GRPO 对同一个 prompt 采样一组回答，用组内奖励构造相对优势，不单独训练 critic。行业对话没有数学题那样清晰的可验证答案，因此这里训练一个 Reward Model 负责综合排序，再用确定性规则检查敏感信息、固定禁语等硬边界。规则能兜住明确错误，但代替不了业务评审。

## 3. 目录里有什么

- [`industry_qwen3_lora_sft.yaml`](./configs/industry_qwen3_lora_sft.yaml)：LLaMA-Factory 的 LoRA SFT 配置起点。
- [`industry_qwen3_lora_reward.yaml`](./configs/industry_qwen3_lora_reward.yaml)：Reward Model 配置起点。
- [`data_pipeline.py`](./src/industry_dialogue/data_pipeline.py)：检查消息格式、去重，把候选排序转成偏好对。
- [`reward_model.py`](./src/industry_dialogue/reward_model.py)：成对 Reward Model 损失与准确率。
- [`grpo.py`](./src/industry_dialogue/grpo.py)：组内相对优势的核心计算。
- [`build_preference_pairs.py`](./scripts/build_preference_pairs.py)：从候选排序生成 `chosen/rejected` JSONL。
- [`prepare_rl_prompts.py`](./scripts/prepare_rl_prompts.py)：把 RL prompt 转成 VeRL 读取的 Parquet。
- [`run_grpo.sh`](./scripts/run_grpo.sh)：VeRL 的最小 GRPO 启动骨架。
- [`evaluation.md`](./docs/evaluation.md)：评估表和训练监控项。

## 4. 先把任务边界写清楚

第一版只选几类能稳定标注的任务，例如转账状态说明、账户安全提醒、办理材料、投诉转人工，以及超出权限时的拒答与升级处理。

不要让模型直接判断实时余额、交易结果、产品收益或具体费率。这些信息需要真实系统或最新业务规则。信息不足时，合格回答应该说明缺少什么，并引导用户走官方查询或人工渠道。

标注前先固定四个维度：

| 维度 | 要看什么 | 常见问题 |
| --- | --- | --- |
| 业务正确性 | 是否符合锁定版本的业务口径，条件和限制是否完整 | 编造时效、费率、入口或处理结果 |
| 合规与安全 | 是否索要敏感信息，是否给出越权承诺 | 要求用户发送验证码、密码或完整卡号 |
| 完整性 | 是否回答核心问题并给出下一步 | 只道歉，不解决问题 |
| 服务表达 | 是否清楚、克制、尊重用户 | 套话过多、责备用户、机械拒答 |

## 5. 第一步：准备 SFT 数据

SFT 样本使用 `messages` 格式，并保留 `group_id`、意图和规则版本。`group_id` 用于保证同一业务模板、改写链或会话不会跨到训练集和测试集。

把处理后的数据注册到 LLaMA-Factory，再启动训练：

```powershell
llamafactory-cli train path/to/industry_qwen3_lora_sft.yaml
```

示例使用 `Qwen/Qwen3-4B-Instruct-2507`、LoRA 和 `qwen3_nothink` template。训练和推理要使用同一 chat template。学习率、batch size、训练轮数和上下文长度都只是起点。

## 6. 第二步：生成并标注候选回答

用 SFT 模型对同一个 prompt 采样多条回答，记录 checkpoint、prompt 版本、temperature、top-p、随机种子和生成时间。候选之间要有差异；几条几乎相同的答案无法提供有效偏好信号。

标注时先逐项判断业务正确、合规、完整和表达，再给总排序。并列回答不要强行拆成偏好对。转换脚本默认只取相邻名次，减少高度重复的 pair：

```powershell
python scripts/build_preference_pairs.py `
  --input data/sample/candidates.jsonl `
  --output data/processed/preferences.jsonl
```

要先按 prompt 或会话分组切分，再在集合内部生成偏好对。不能先生成大量 pair，再逐对随机切分。

## 7. 第三步：训练 Reward Model

Reward Model 接收 prompt 与回答并输出标量分数。对 `chosen` 和 `rejected`，常见成对目标为：

```text
loss = -log(sigmoid(reward_chosen - reward_rejected))
```

训练可以从 LLaMA-Factory 的 Reward Modeling 配置开始：

```powershell
llamafactory-cli train path/to/industry_qwen3_lora_reward.yaml
```

不要只看训练 loss。至少报告独立 prompt 上的 pairwise accuracy，并按任务类型、错误标签和回答长度分桶。还要抽查最高分错误回答与最低分优质回答，检查长度偏置、固定礼貌用语和模板偏置，因为这些漏洞会在 GRPO 阶段被放大。

## 8. 第四步：用 VeRL 跑 GRPO

对每个 prompt 采样 `G` 条回答，Reward Model 为每条回答打分。最简单的组内标准化优势是：

```text
advantage_i = (reward_i - mean(group_rewards)) / (std(group_rewards) + eps)
```

[`grpo.py`](./src/industry_dialogue/grpo.py) 展示了这一步。实际 VeRL 还会计算新旧策略概率比、裁剪目标和相对 reference policy 的 KL。

GRPO 阶段有三个主要模型角色：

- `policy`：从 SFT checkpoint 初始化，生成回答并更新参数；
- `reference policy`：冻结的 SFT 参考模型，用于约束 KL 偏离；
- `reward model`：对 prompt 和回答打分。

它没有单独的 critic。省掉 critic 不等于训练一定更省时间，因为同一 prompt 需要生成多条回答，rollout 仍可能是主要开销。

先把 prompt 转成 Parquet，再按固定的 VeRL commit 核对并运行 [`run_grpo.sh`](./scripts/run_grpo.sh)：

```powershell
python scripts/prepare_rl_prompts.py `
  --input data/sample/rl_prompts.jsonl `
  --output data/processed/rl_prompts.parquet
```

VeRL 目前仍通过 `verl.trainer.main_ppo` 入口承载多种 RL 算法；真正选择 GRPO 的字段是 `algorithm.adv_estimator=grpo`。`actor_rollout_ref.rollout.n` 必须大于 1。配置字段会随版本演进，正式运行前要对照所固定版本的官方 GRPO 示例。

如果一组回答的 reward 几乎相同，组内标准差接近零，GRPO 就没有有效相对信号。`eps` 只能避免除零，不能创造学习信号；应检查采样多样性、Reward Model 区分度和 prompt 难度。

## 9. 第五步：做对照评估

至少保留 Base、SFT 和 GRPO 三组结果。所有模型使用同一测试集、system prompt、chat template 和解码配置；人工评审隐藏模型名称并随机交换左右顺序。

建议报告：

- 任务完成率、业务正确率、违规率和过度拒答率；
- GRPO 相对 SFT 的盲评胜、负、平比例；
- Reward Model 在独立 prompt 上的 pairwise accuracy；
- reward、KL、回复长度、entropy、clip fraction、组内奖励标准差和零方差组比例；
- 按业务意图划分的失败样例。

Reward 上涨不等于模型变好。若 reward 持续上涨但盲评下降、回答变长或收敛到固定模板，应停止训练并检查 reward hacking。

## 10. 简历上怎么写

简历要写清业务约束、自己做过的数据和训练工作、为什么选 GRPO，以及用什么口径证明效果。下面按 4B 模型和单机 LoRA/GRPO 的项目量级给出一组完整写法。**这些数字不是当前仓库的实测结果**，必须用独立评测集复现并替换。

**项目名称：高质量行业对话助手后训练**

**技术栈：Qwen3、LLaMA-Factory、LoRA、Reward Model、VeRL、GRPO**

- 面向金融客服的 12 类常见任务，构建 1.86 万条 SFT 样本和 6400 组同 prompt 偏好数据，制定业务正确、合规、完整性和服务表达四维标注规则，并按 prompt 和会话分组切分，避免相似话术同时进入训练集和测试集；
- 基于 Qwen3 完成 LoRA SFT 与成对 Reward Model 训练，在 800 组独立偏好对上取得 76.8% 的 pairwise accuracy，并通过长度分桶、意图分桶和极端样例复核排查“越长分越高”等奖励偏置；
- 基于 VeRL 使用 GRPO 优化 SFT 模型，对同一 prompt 采样 8 条回答并计算组内相对优势，通过 reference KL、组内奖励标准差、零方差组比例和输出长度监控训练稳定性；
- 在 300 条独立 prompt 上完成 SFT/GRPO 双盲对比，GRPO 的胜/平/负为 40.0%/35.0%/25.0%，非平局胜率 61.5%；业务正确率由 78.3% 提升到 83.0%，违规率由 5.3% 降到 3.0%。

300 条盲评对应 120 胜、105 平、75 负；业务正确回答从 235 条增加到 249 条，违规回答从 16 条降到 9 条。百分比、样本量和原始计数要能相互核对。

## 11. 面试官可能怎么问

### 11.1 为什么 SFT 之后还要做强化学习？

回答要点：SFT 学习示范答案的 Token 分布，但多个可接受回答之间往往有优劣差异。偏好数据和 Reward Model 把业务正确、合规等排序信号显式化；GRPO 再直接优化策略，使高奖励回答在同 prompt 候选中更可能出现。是否真的值得做，要由 SFT/GRPO 对照结果决定。

### 11.2 SFT 数据和偏好数据有什么区别？

回答要点：SFT 数据告诉模型“应该怎样回答”，偏好数据比较同一个 prompt 下“哪条更好”。偏好数据必须共享 prompt 和评分口径；把不同问题的回答强行比较没有意义。

### 11.3 Reward Model 的成对损失优化了什么？

回答要点：让 `reward(chosen)` 高于 `reward(rejected)`。它学习的是相对排序，不保证标量分数有统一绝对含义；因此看 pairwise accuracy、分桶表现和极端错误，比只看 reward 均值更重要。

### 11.4 为什么用 GRPO，不用 PPO？

回答要点：GRPO 用同 prompt 多条回答的组内分数构造基线，不需要额外训练 critic，链路更适合这个以候选比较为核心的项目。代价是每个 prompt 要做多次 rollout，且效果依赖候选多样性和奖励区分度。选择理由不能只说“更新”。

### 11.5 GRPO 的组大小 `G` 怎么选？

回答要点：`G` 越大，组内排序和标准化通常更稳定，但 rollout 显存与时间也增加。先根据单卡/集群吞吐和每个 prompt 的有效候选数选起点，再比较 reward 方差、零方差组比例、训练曲线和最终盲评；没有通用最优值。

### 11.6 一组回答的 reward 都一样怎么办？

回答要点：标准化后的优势接近零，这组样本几乎不给策略更新信号。先排查采样过于确定、候选文本近似、Reward Model 饱和或 prompt 太简单；数值上的 `eps` 只防止除零。

### 11.7 KL 约束有什么作用？

回答要点：限制 policy 过快偏离 SFT reference，降低只追逐 Reward Model 漏洞和语言能力退化的风险。KL 太小可能约束不足，太大则学不到偏好；结合 KL 曲线、盲评和输出分布调节，而不是只盯一个系数。

### 11.8 为什么行业对话还需要 Reward Model？

回答要点：数学或代码任务常有可验证奖励，但客服回答包含业务条件、合规和表达质量，很难靠单一规则覆盖。学习型 Reward Model 负责综合排序；明确的敏感信息或禁语可以作为规则惩罚补充，两者都要独立评估。

### 11.9 怎么发现 reward hacking？

回答要点：对比 reward 与盲评趋势，检查高 reward 错误回答、长度相关性、固定模板、过度拒答和敏感词投机。模型选择要看业务指标和盲评，而不是选择 reward 最高的 checkpoint。

### 11.10 怎样证明提升来自 GRPO？

回答要点：固定测试集、system prompt、chat template 和解码参数，对 Base、SFT、GRPO 做盲评；保留 SFT checkpoint 作为唯一初始化基线，报告胜负平、分意图指标和置信区间或样本量，同时披露退化项。
