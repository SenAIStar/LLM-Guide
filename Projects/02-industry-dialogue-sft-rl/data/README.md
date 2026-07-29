# 1. 数据目录

这里的 `sample/` 只有几条合成样例，用来解释格式，不能当作可训练的数据集。

真实数据建议分开保存：

```text
data/
├── private/       # 已授权但不提交到 Git 的原始数据
├── processed/     # 脱敏、去重、分组切分后的数据
└── sample/        # 可以公开的合成样例
```

四类文件分别做不同的事：

- `sft.jsonl`：高质量示范回答，用于 SFT；
- `candidates.jsonl`：同一 prompt 的多条候选回答和人工排序；
- `preferences.jsonl`：从排序得到的 `chosen/rejected`，用于 Reward Model；
- `rl_prompts.jsonl`：只保留 prompt，用于 GRPO 的分组 rollout。

训练、验证和测试要按 `group_id` 或 `prompt_id` 切分。先切 prompt，再生成偏好对；不能把同一 prompt 的 pair 随机分到不同集合。

每批真实数据至少记录来源、授权范围、脱敏规则、业务规则版本、标注规范版本和生成模型 checkpoint。含个人信息、内部业务数据或未获授权的对话不能提交到公开仓库。
