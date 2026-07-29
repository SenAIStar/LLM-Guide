# LLM-Guide Projects

这里有七个大模型项目。每个目录都保留了数据处理、训练、评估和答辩时真正会用到的核心代码：

1. [法律大模型（SFT）](./01-legal-qa-sft/)：整理法律语料，完成数据清洗、分组切分、LoRA SFT 和离线评估。
2. [高质量行业对话助手（SFT + GRPO）](./02-industry-dialogue-sft-rl/)：以金融客服为例，完成 SFT、偏好标注、Reward Model、GRPO 和盲评。
3. [企业内部知识问答（RAG + SFT）](./03-enterprise-knowledge-rag-sft/)：完成文档分块、混合检索、重排、引用与拒答 SFT，以及检索/生成分层评估。
4. [政策智能问答系统（RAG + SFT + GRPO）](./04-policy-qa-rag-sft-grpo/)：增加政策版本、地区与适用对象过滤，用 SFT 学习证据使用，再用 GRPO 优化引用、拒答和合规行为。
5. [多功能任务智能体（Agent + GRPO）](./05-multitool-agent-grpo/)：在 mock 工具环境中训练 Agent 选择工具、填写参数并完成目标，用终态而不是模型自述判断成功。
6. [深度研究助手（Agent + SFT + GRPO）](./06-deep-research-agent-sft-grpo/)：保留规划、检索和综合主线，补上证据卡、搜索快照、引用评测和轨迹级强化学习。
7. [智能问答助理（Agent + RAG + SFT + GRPO）](./07-agentic-rag-qa-sft-grpo/)：让 Agent 决定 query rewrite、检索分支、重排、补检和拒答，并对检索、轨迹和答案分层评测。

这些不是开箱即用的产品代码，也没有为了显得完整去搭前后端。重点是把算法链路、实验口径和失败分析写明白，让项目能做、能测，也能在面试里讲清楚。

README 里的简历数字用于展示结果该怎么写，不是仓库实测成绩。真正放进简历前，要换成自己的实验结果，并保留配置、逐题输出和失败样例。

示例数据都是合成的，只用于说明字段和代码流程，不能直接用于真实法律判断、政策决策、企业权限控制或外部系统操作。
