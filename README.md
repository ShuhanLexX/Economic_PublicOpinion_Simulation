# 生成式代理模型驱动的消费券政策效果模拟评估

项目以公共危机事件背景下的消费券发放政策为案例，构建了一个基于大语言模型（LLM）驱动的生成式代理模型（Generative Agent-Based Model, GABM）仿真平台，通过深度模拟了千名社会智能体在舆论子系统与经济子系统中的动态行为，验证了该范式在政策效果预评估中的可行性。

## 方法概要

- **智能体构建**：基于真实社会调查数据（人口学、经济、心理等多维属性）构建千级规模的异质性社会智能体，每个智能体具备短期/长期记忆及定期反思机制。
- **LLM驱动决策**：智能体通过调用大语言模型（兼容 OpenAI API），在给定个人画像、记忆和外部环境信息后，生成符合角色设定的社交媒体发言与经济行为决策。
- **双子系统环境交互**：舆论子系统负责热搜新闻分发与帖子推荐流转；经济子系统管理消费券发放、消费能力核算与政策效果统计，两者通过智能体行为耦合。
- **政策传导模拟**：按月推进仿真，在指定月份注入消费券政策，观测政策对智能体消费率、工作意愿、舆论情感等指标的影响轨迹。

## 项目结构

```
├── src/
│   ├── main.py                  # 主程序入口
│   ├── config/config.py         # 全局配置
│   ├── agents/
│   │   ├── agent.py             # 智能体定义与LLM决策
│   │   └── memory.py            # 记忆模块（FAISS向量检索 + 反思）
│   ├── models/
│   │   ├── llm_models.py        # LLM接口（含负载均衡）
│   │   └── embedding_model.py   # Embedding模型加载
│   ├── systems/
│   │   ├── social_system.py     # 舆论子系统
│   │   └── economic_system.py   # 经济子系统
│   ├── prompts/agent_prompts.py # Prompt模板
│   └── utils/                   # 日志、工具函数
├── data/
│   ├── agent_pool/              # 智能体档案（JSON）
│   └── social_data/             # 热搜新闻与政策信息
├── population/                  # 人口数据模板
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 配置 API

本项目通过 OpenAI 兼容 API 调用大语言模型。请通过环境变量设置：

```bash
export GLM_BASE_URL="http://your-llm-api:port/v1"
export GLM_API_KEY="your-api-key"
export GLM_MODEL_NAME="glm-4-flash"            # 或其他兼容模型
export EMBEDDING_MODEL_PATH="sentence-transformers/all-MiniLM-L6-v2"  # 本地路径或HuggingFace名称
```

### 3. 运行模拟

```bash
cd src
python main.py --agents 100 --months 13
```

- `--agents`：参与模拟的智能体数量（默认1000）
- `--months`：模拟月数（默认13）

模拟结果（经济报告、情感统计、智能体状态等）将保存在 `data/results/` 下按时间戳命名的目录中。
"# Economic_PublicOpinion_Simulation" 
"# Economic_PublicOpinion_Simulation" 
