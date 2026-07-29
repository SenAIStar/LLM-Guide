# 06 深度研究助手（Agent + SFT + GRPO）

深度研究的链路很长：拆问题、搜索、读取、保存证据、处理冲突，最后才写报告。训练和评估都使用固定搜索快照，避免网页变化把实验结果搅在一起。

仓库不是完整浏览器 Agent，只保留研究轨迹、证据卡、引用审计、奖励和评测这些算法侧核心结构。

## 1. 做完后要留下什么

- 带依赖关系和预算的研究计划；
- 可回放的 search/fetch/action/observation 轨迹；
- 网页快照、抓取时间、页面 hash 和证据卡；
- SFT 轨迹数据，以及 SFT/GRPO 对照结果；
- claim 级引用判断、来源冲突和失败样例。

长报告不是主要成果。真正能体现项目质量的是，任何关键结论都能回到证据，任何实验都能在同一搜索环境中重放。

## 2. 整体链路

```text
研究问题
  -> 拆成子问题并分配搜索预算
  -> 多轮 search / fetch
  -> 保存网页快照和页面 hash
  -> 建立 claim-evidence 证据卡
  -> 检查来源冲突与信息缺口
  -> 生成提纲和报告
  -> citation audit
  -> SFT 学习高质量轨迹
  -> GRPO 优化覆盖、引用、冲突处理与成本
```

规划阶段先确定子问题和依赖，再根据中间证据补搜索。这样能解释为什么搜、何时停止，以及哪些结论仍缺依据。

## 3. 证据卡和搜索快照

每张证据卡至少保存：URL、标题、发布日期、抓取时间、页面 hash、claim、证据片段、来源类型和冲突组。报告里的 claim 引用卡片 ID，再由卡片回溯网页快照。

Web 内容会变，同一 URL 也可能在不同日期返回不同正文。离线 SFT、GRPO 和评测使用固定快照；实时 Web Search 另做在线评测。两种结果不能合并成一个提升数字。

[`pipeline.py`](./src/deep_research/pipeline.py) 里定义了 `EvidenceCard`、`ResearchTrace` 和 `SearchSnapshot`。快照接口让训练过程不依赖实时网络，也便于控制搜索成本和复现实验。

## 4. 研究任务与数据切分

任务覆盖事实核查、方案比较、技术调研和多来源综述。每条任务要标注关键要点、需要证据的 claim、允许来源类型、最大步数和报告预算。

训练集与测试集按主题、核心实体和来源簇切分。同一主题只换问法仍然可能泄漏；同一网页的镜像、转载和摘要也要归到同一来源簇。

错误轨迹可以保留用于分析，但不要直接当 SFT 正例。优质轨迹至少满足：搜索有增量、证据可访问、引用支持 claim、冲突没有被掩盖、停止理由合理。

## 5. SFT 训练长轨迹

SFT 数据保留 action 与 observation 的交替过程，而不只保存问题和最终报告。模型需要学习什么时候搜索、怎样改写 query、何时读取页面、哪些片段值得保存，以及证据不足时怎样继续查。

环境 observation 不应计算策略 loss。训练样本中要清楚区分模型生成的 action Token 与搜索环境返回的文本，只对模型侧 Token 建立监督标签。否则模型会被训练去复述网页内容，轨迹建模也会失真。

第一轮先做行为克隆，检查 action 合法率、搜索步数和引用格式，再进入 RL。SFT 基线不稳时直接做 GRPO，reward 很容易被无效轨迹和格式错误淹没。

## 6. GRPO 奖励

同一研究问题在同一快照上采样多条轨迹，奖励分成：

- 任务覆盖：金标要点是否回答；
- 引用正确：引用是否存在并支持对应 claim；
- 引用完整：需要证据的 claim 是否被覆盖；
- 来源覆盖：关键子问题是否有足够独立来源；
- 冲突处理：矛盾信息是否被识别和解释；
- 成本约束：重复 query、无效访问、超步数和报告过长。

[`rewards.py`](./src/deep_research/rewards.py) 给出了奖励分解。引用数量不能直接代表质量；如果只奖励“多引用”，模型会堆重复来源和无关链接。

`configs/grpo.yaml` 是奖励权重与预算的设计稿。真正接入 VeRL 时要固定版本，并确保网页 observation 的 Token 不进入策略梯度。

## 7. 引用和冲突怎么评估

先把报告拆成 claim，再判断哪些 claim 需要外部证据。两个核心指标是：

- `citation_precision`：已经给出的引用中，有多少真正支持对应 claim；
- `citation_completeness`：需要证据的 claim 中，有多少被有效引用覆盖。

还要报告 unsupported claim rate、无效引用数、来源覆盖、冲突处理率、重复 query 比例、平均步数和平均 Token。[`evaluation.py`](./src/deep_research/evaluation.py) 提供了 claim 级引用统计。

## 8. 对照与消融

至少比较 Prompt baseline、SFT、SFT + GRPO。三组使用同一任务、搜索快照、最大步数、报告预算和解码参数。

消融可以依次移除证据卡、冲突奖励、引用奖励和成本惩罚。重点观察四类失败：结论无支持、引用存在但不支持、多个来源冲突未处理、搜索循环或超预算。

实时搜索只用于补充观察鲁棒性，不用于替代固定快照上的主结果。

## 9. 代码从哪里看

- [`pipeline.py`](./src/deep_research/pipeline.py)：研究状态、搜索快照、证据卡、停止原因和引用审计；
- [`sft.py`](./src/deep_research/sft.py)：把 action 标为训练目标，并对环境 observation 做 loss mask；
- [`rewards.py`](./src/deep_research/rewards.py)：覆盖、引用、冲突和成本奖励；
- [`evaluation.py`](./src/deep_research/evaluation.py)：citation precision、completeness 与无支持结论率；
- [`grpo.yaml`](./configs/grpo.yaml)：轨迹预算、observation mask 和奖励权重；
- [`data/sample/tasks.jsonl`](./data/sample/tasks.jsonl)：研究任务字段样例。

## 10. 简历怎么写

简历可以按下面四条组织，**数字不是仓库实测**。使用前换成自己的搜索快照、逐条引用判断和盲评记录。

**深度研究助手｜Qwen3、Web Search、SFT、GRPO、VeRL**

- 将研究任务拆为规划、检索、证据卡、冲突检查、综合写作和引用审计六个阶段，保存 query、网页快照、claim 与证据映射；
- 构造 800 条跨 8 个主题的研究任务并按主题与来源切分为 600/100/100，使用高质量工具轨迹完成 SFT，再以任务覆盖、引用支持、冲突识别和搜索成本设计 GRPO 奖励；
- 在固定 100 条测试任务与搜索快照上，将完整回答任务数由 SFT 的 68 条提升到 GRPO 的 76 条；抽检 600 个引用，citation precision 由 71.7%（430/600）提升到 86.0%（516/600）；
- 通过重复 query 检测和步数惩罚将近重复搜索占比由 21.0% 降至 12.0%，并保留无支持 claim、来源冲突、网页失效和超预算四类失败样例。

## 11. 面试官可能会问

**为什么不能只用一个超长 Prompt？**

回答要点：复杂研究需要根据中间证据发现新问题、补来源和处理冲突，单次 Prompt 无法动态调整，也很难追踪 claim 与证据的对应关系。

**怎样保证实验可复现？**

回答要点：固定搜索 API 或检索器、网页快照、抓取时间、页面 hash、模型版本、最大步数和 Token 预算，并保存完整轨迹。实时搜索单独报告。

**为什么 observation 要做 loss masking？**

回答要点：observation 是环境返回，不是 policy 生成的 action。若对它计算语言模型 loss，模型会学习复述环境文本，策略梯度的责任边界也会混乱。

**citation precision 和 completeness 有什么区别？**

回答要点：precision 看已有引用是否支持 claim；completeness 看所有需要证据的 claim 是否都有有效引用。一个报告可以 precision 很高但 completeness 很低。

**长轨迹的 credit assignment 怎么处理？**

回答要点：用终态覆盖和引用指标评价整条轨迹，再用少量确定性过程惩罚约束循环、无效调用和超预算。奖励项必须通过消融验证，不能凭感觉叠加。

**如何防止为了奖励堆引用？**

回答要点：按 claim 检查支持性，合并重复来源，限制报告和引用预算，并在未参与奖励设计的人工盲测集上检查可读性与正确性。
