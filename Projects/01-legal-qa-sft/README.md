# 01 法律大模型（SFT）

法律大模型这一节只做 SFT，不接检索和知识库。先准备法律语料，整理成监督微调数据，再用 LoRA 训练一个更熟悉中文法律任务的领域模型。

仓库只放关键步骤。示例数据全部是合成的，用来说明格式和流程，不能用于真实法律判断。

## 1. 做完后要留下什么

做完后，手里应该留下三类结果：

- 可复现的数据处理流程：字段统一、脱敏、去重、按来源组切分和质量报告；
- 法律领域 LoRA adapter：训练配置、loss 曲线、checkpoint 和实验记录；
- Base 与 SFT 的同口径对比：自动指标、人工评估、分任务结果和失败样例。

面试时不要只说“跑过一次训练”。要能解释数据为什么这样选、怎样避免训练测试泄漏、SFT 改善了什么，以及哪些问题仍然没有解决。

## 2. 目录里有什么

- [`data_pipeline.py`](./src/legal_qa/data_pipeline.py)：把 Pair/Triplet 数据转成 `messages`，做脱敏、去重和 group split。
- [`prepare_sft_data.py`](./scripts/prepare_sft_data.py)：串起数据准备流程，输出 `train/dev/test.jsonl` 和质量报告。
- [`law_qwen3_lora_sft.yaml`](./configs/law_qwen3_lora_sft.yaml)：LLaMA-Factory 的 LoRA SFT 配置起点。
- [`dataset_info.json`](./configs/dataset_info.json)：把本地 `messages` 数据注册给 LLaMA-Factory。
- [`evaluation.py`](./src/legal_qa/evaluation.py)：分类任务的 Accuracy、Macro-F1 和空输出检查。
- [`evaluation.md`](./docs/evaluation.md)：Base 与 SFT 的评估口径。

## 3. 数据怎么选

第一版不要把所有法律任务一次塞进来。先选一个主任务，例如罪名预测、法律问答、法律文书摘要或争议类型分类，再围绕它准备训练集和评估集。

公开数据可以从下面几类里选：

- CAIL 提供刑事法律文书及罪名、法条、刑期等标注，适合判决预测类任务；
- DISC-Law-SFT 包含 Pair 和 Triplet 子集，覆盖法律信息抽取、判决预测、文书摘要和法律问答等场景；
- LawBench 包含 20 个法律任务，适合作为外部评估集，不要再混回训练集。

下载前先看许可证、仓库说明和原始数据条款。训练集、验证集和外部评测集要分开管理；真实案件还需要额外确认授权、隐私和使用边界。

## 4. 训练数据格式

这里使用 LLaMA-Factory 支持的 OpenAI `messages` 形式，并在 `dataset_info.json` 中按 ShareGPT 格式注册：

```json
{
  "id": "law_000001",
  "group_id": "case_or_question_group",
  "messages": [
    {
      "role": "system",
      "content": "你是法律领域助手。回答时不要虚构法条、案例或事实。"
    },
    {
      "role": "user",
      "content": "这里放已经脱敏的法律问题或案件事实"
    },
    {
      "role": "assistant",
      "content": "这里放经过检查的标准答案"
    }
  ],
  "metadata": {
    "source_dataset": "数据集名称",
    "source_format": "原始字段类型"
  }
}
```

`id`、`group_id` 和 `metadata` 不必送入模型，但要保留下来做追踪、切分和错误分析。

## 5. 数据处理

先检查原始字段，再写转换规则，不要假设所有文件都有同一种结构。当前代码分别处理 DISC-Law-SFT 的 Pair 和 Triplet：Pair 使用 `id/input/output`，Triplet 还读取 `reference`。

```powershell
python scripts/prepare_sft_data.py `
  --pair data/private/DISC-Law-SFT-Pair.jsonl `
  --triplet data/private/DISC-Law-SFT-Triplet-released.jsonl `
  --output-dir data/processed
```

处理流程包括：

1. 统一 Unicode、空白和字段名；
2. 遮蔽手机号、身份证号、邮箱和银行卡号等明显隐私信息；
3. 对规范化后的问答做精确去重；
4. 按案件、问题来源或改写链分组切分，避免同源样本跨训练集和测试集；
5. 输出样本量、来源格式、重复数和各切分数量。

正则只能处理格式明显的隐私信息。真实案件数据还要人工抽查人名、地址、案号和事实描述里的间接身份信息。

## 6. 开始训练

把生成的 `train.jsonl` 和 `dev.jsonl` 放进 LLaMA-Factory 的 `data/`，再合并 [`dataset_info.json`](./configs/dataset_info.json)：

```powershell
llamafactory-cli train path/to/law_qwen3_lora_sft.yaml
```

示例配置使用 `Qwen/Qwen3-4B-Instruct-2507`、LoRA 和 `qwen3_nothink` template。训练和推理必须使用同一 chat template。YAML 里的 batch size、学习率、轮数和上下文长度只是实验起点，要根据显存、Token 长度分布、eval loss 和失败样例调整。

每次实验至少记录：基础模型 revision、数据文件 hash、随机种子、完整 YAML、train/eval loss、最终 checkpoint 和运行环境。

## 7. 怎么评估

先用同一批测试样本跑 Base，再固定 system prompt、chat template 和解码参数跑 SFT。

分类任务可以报告 Accuracy 和 Macro-F1。开放式法律问答不要只看 BLEU 或 ROUGE，还要人工检查问题理解、事实使用、虚构内容、结论边界和信息不足时的处理。详细口径见 [`docs/evaluation.md`](./docs/evaluation.md)。

至少保留：

- Base/SFT 总体指标和分任务指标；
- 10 到 20 条典型成功与失败样例；
- 错误类型占比，以及下一轮准备补什么数据；
- 没有改善甚至退化的能力，不能只展示正向样例。

## 8. 简历上怎么写

简历不要写成“收集数据、LoRA 微调、效果显著提升”。面试官真正会追问的是：数据从哪里来、怎么避免泄漏、Base 和 SFT 是否同口径比较，以及失败样例暴露了什么问题。

下面是一组结果口径完整的简历写法，数据规模按公开法律数据集和 4B 模型单机 LoRA 的常见项目量级设置。**数字不是当前仓库跑出的结果**。先在固定评测集上完成实验，再把它们换成自己的实测值。

**项目名称：法律领域大模型监督微调**

**技术栈：Qwen3、LLaMA-Factory、LoRA、PyTorch**

- 面向法律咨询问答，从 DISC-Law-SFT 的 9.3 万条法律问答数据中清洗出 7.8 万条有效样本，按问题来源分组划分 7.2 万条训练集、3000 条验证集和 3000 条内部测试集，完成字段统一、隐私脱敏和精确去重；
- 基于 `Qwen/Qwen3-4B-Instruct-2507` 使用 LoRA 完成 SFT，统一训练与推理 chat template，记录数据 hash、训练配置和 Base/SFT 对照实验；
- 在 LawBench 法律咨询任务的 500 条外部样本上，将 ROUGE-L 从 26.4 提升到 31.8；对其中 200 条回答进行人工盲评，结论正确且边界清楚的比例由 68.0% 提升到 75.5%，虚构法条率由 12.0% 降到 7.5%，并将错误归纳为法条张冠李戴、事实条件遗漏、过度推断和风险边界不清四类。

答辩时把预测文件、评分脚本和人工评审记录带上。单独报一个百分比，无法证明切分方式、评测口径和失败样例经得住复核。

## 9. 面试官可能怎么问

### 9.1 为什么用 SFT，不做 RAG？

回答要点：这个项目的目标是让模型学习法律任务的回答形式和领域行为，验证领域 SFT 本身的收益，所以主动把检索排除在外。SFT 不能保证法条实时更新，也不能给每个结论提供可追溯证据；如果产品要求依据最新法规回答，RAG 应该作为另一个模块加入，不能把两者能力混在一起描述。

### 9.2 法律数据从哪里来，怎么确认能用？

回答要点：说明具体数据集、许可证和用途；公开数据不等于可以任意商用。真实案件或咨询数据还要确认授权、脱敏和保存范围。简历里只写实际使用过的数据，不把“看过”写成“训练过”。

### 9.3 为什么不能随机切分每一条样本？

回答要点：同一案件、同一问题的改写或同一来源模板可能高度相似。逐条随机切分会让近重复内容同时进入训练集和测试集，指标虚高。应该先构造 `group_id`，再按组切分。

### 9.4 LoRA 的 rank 和 target modules 怎么选？

回答要点：rank 决定低秩适配器容量，也影响显存和训练参数量；`lora_target: all` 是起点，不是最优结论。用固定数据与评估集比较不同 rank/target，观察任务指标、eval loss、通用能力退化和资源消耗，再做选择。

### 9.5 为什么设置 `train_on_prompt: false`？

回答要点：监督信号主要来自 assistant 回复，通常不希望 system/user Token 参与回答目标的 loss。要确认所用 template 和数据格式真的生成了预期 label mask，不能只看配置名。

### 9.6 开放式法律问答怎么评估？

回答要点：词面指标只能辅助参考。人工评审至少检查问题理解、引用事实是否来自输入、是否虚构法条或案例、结论是否越界、信息不足时是否追问或保留判断；评审规则要提前固定并保存分歧。

### 9.7 SFT 后通用能力下降怎么办？

回答要点：先用通用保留集确认是否真的退化，再排查学习率、训练轮数、领域数据重复和样本分布。可以降低训练强度、混入适量高质量通用指令数据或做多任务 SFT，但必须用同一评估集验证，不能只凭主观感受。

### 9.8 这个模型可以直接提供法律意见吗？

回答要点：不可以。项目只验证离线任务能力；训练语料可能过时，模型仍会幻觉，也没有事实检索和责任边界控制。真实应用需要最新知识来源、引用校验、风险分级、人工复核和合规评估。

## 10. 可以怎么继续扩展

当前版本到 SFT 和离线评估就结束。后续可以比较基础模型、数据配比、LoRA rank、QLoRA、合成数据筛选或多任务 SFT，但每轮只改少量主要变量，并保留可比较的实验记录。
