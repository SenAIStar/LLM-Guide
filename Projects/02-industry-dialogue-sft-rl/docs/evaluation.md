# 评估说明

评估集要在训练前锁定，按 `group_id` 或 `prompt_id` 与训练数据隔离。Base、SFT 和 GRPO 使用同一批 prompt、system prompt、chat template 和解码参数。

## 1. Reward Model

在从未进入训练集的 prompt 上报告：

| 指标 | 说明 |
| --- | --- |
| Pairwise accuracy | `reward(chosen) > reward(rejected)` 的比例 |
| 分意图准确率 | 找出 Reward Model 不擅长的业务场景 |
| 分错误标签准确率 | 检查安全、业务错误、信息不全等偏好是否学到 |
| 长度分桶准确率 | 检查是否把“更长”误当成“更好” |

还要画回答长度与 reward 的关系，并抽查高分错误回答和低分优质回答。

## 2. Base、SFT 与 GRPO

每条回答由看不到模型名称的评审人员标注：

- `task_success`：是否解决用户问题；
- `business_correct`：业务信息是否正确且没有编造；
- `policy_violation`：是否出现不合规或不安全内容；
- `over_refusal`：本来能回答却直接拒答；
- `asks_sensitive_info`：是否索要密码、验证码或其他敏感信息；
- `preference`：A 胜、B 胜或平局；
- `error_tags`：错误类型和简短原因。

比较 SFT 与 GRPO 时随机交换 A/B 位置，报告胜率、负率和平局率，并保留评审分歧。

## 3. GRPO 训练监控

至少保存：

- Reward Model score；
- policy 与 reference policy 的 KL；
- response length 和 policy entropy；
- policy clip fraction；
- 每个 prompt 组内 reward 标准差；
- 零方差或近零方差组比例；
- 固定验证集上的盲评或独立规则评估。

以下现象需要停止并排查：

- reward 上升而盲评下降；
- KL 突然增大，模型明显偏离 SFT 行为；
- 回复持续变长或收敛到固定模板；
- 合规率提高，但过度拒答明显增加；
- 大量 prompt 的组内 reward 方差接近零；
- clip fraction 或 entropy 长时间异常。

## 4. 结果表

所有数字都从真实实验填入：

| 模型 | 任务完成率 | 业务正确率 | 违规率 | 过度拒答率 | 相对 SFT 盲评胜率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 待测 | 待测 | 待测 | 待测 | - |
| SFT | 待测 | 待测 | 待测 | 待测 | - |
| GRPO | 待测 | 待测 | 待测 | 待测 | 待测 |
