"""
智能体定义：属性、记忆、经济行为与LLM驱动的决策
"""
import json
import os
import datetime
import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from memory import AgentMemory
from logger import logger
from utils import load_json, save_json
from config import AGENT_POOL_DIR, CURRENT_TIME, BASE_CONSUMPTION_RATE, BASE_WORK_WILLINGNESS
from llm_models import glm_model, call_llm_model, load_balanced_model
from agent_prompts import AGENT_DECISION_PROMPT, SENTIMENT_ANALYSIS_PROMPT


class AgentDecision(BaseModel):
    """单次决策结果"""
    social_post: Dict[str, Any] = Field(
        default_factory=lambda: {"content": "", "sentiment": 0.5})
    economic_decision: Dict[str, Any] = Field(
        default_factory=lambda: {
            "consumption_rate": BASE_CONSUMPTION_RATE,
            "work_willingness": BASE_WORK_WILLINGNESS})
    reasoning: str = ""


class Agent(BaseModel):
    """社会智能体：承载人口学、经济、心理等多维属性，通过LLM完成行为决策"""

    # 多维属性
    agent_id: int
    demographic: Dict[str, Any] = Field(default_factory=dict)
    economic: Dict[str, Any] = Field(default_factory=dict)
    family: Dict[str, Any] = Field(default_factory=dict)
    location: Dict[str, Any] = Field(default_factory=dict)
    psychological: Dict[str, Any] = Field(default_factory=dict)
    social: Dict[str, Any] = Field(default_factory=dict)
    enriched: Dict[str, Any] = Field(default_factory=dict)

    # 状态
    memory: Optional[AgentMemory] = None
    current_month: int = 0
    current_time: str = CURRENT_TIME
    economic_history: List[Dict[str, Any]] = Field(default_factory=list)
    received_vouchers: bool = False
    voucher_amount: int = 0

    class Config:
        arbitrary_types_allowed = True

    # ──────── 构造 ────────

    @classmethod
    def from_file(cls, agent_id: int, embedding_model=None):
        """从JSON文件加载智能体并初始化记忆模块"""
        agent_file = os.path.join(AGENT_POOL_DIR, f"agent_{agent_id}.json")
        agent_data = load_json(agent_file)
        agent = cls(**agent_data)
        if embedding_model:
            agent.memory = AgentMemory(agent_id=agent_id, embedding_model=embedding_model)
        logger.info(f"智能体 {agent_id} 加载完成")
        return agent

    # ──────── 信息摘要 ────────

    def get_info_summary(self) -> str:
        """生成自然语言格式的个体画像摘要，供LLM prompt使用"""
        info = []
        # 人口学
        gender = self.demographic.get("gender", {}).get("label", "未知")
        age = self.demographic.get("age", {}).get("value", 0)
        education = self.demographic.get("w01", {}).get("label", "未知")
        marriage = self.demographic.get("marriage_last", {}).get("label", "未知")
        info.append(f"基本信息: {gender}, {age}岁, 教育水平: {education}, 婚姻状况: {marriage}")

        # 经济
        income = self.economic.get("income", {}).get("value", 0)
        income_sat = self.economic.get("qg401", {}).get("label", "未知")
        job_sat = self.economic.get("qg406", {}).get("label", "未知")
        info.append(f"经济状况: 月收入 {income}元, 收入满意度: {income_sat}, 工作满意度: {job_sat}")

        # 家庭
        fml = self.family.get("fml_count", {}).get("value", 0)
        child = self.family.get("child16n", {}).get("value", 0)
        info.append(f"家庭情况: 家庭成员 {fml}人, 16岁以下子女 {child}人")

        # 地理
        urban = self.location.get("urban20", {}).get("label", "未知")
        province = self.location.get("province", {}).get("label", "未知")
        info.append(f"地理位置: {province}, {urban}")

        # 心理
        happiness = self.psychological.get("qm2016", {}).get("value", 0)
        life_sat = self.psychological.get("qn12012", {}).get("label", "未知")
        info.append(f"心理状态: 幸福感 {happiness}/10, 生活满意度: {life_sat}")

        # 丰富化信息（生活经历）
        if isinstance(self.enriched, dict) and "life_experience" in self.enriched:
            exp = self.enriched["life_experience"]
            if isinstance(exp, dict):
                for key, label in [("education", "教育经历"), ("work", "工作经历"),
                                   ("family_life", "家庭生活")]:
                    val = exp.get(key, "")
                    if val:
                        info.append(f"{label}: {val}")

        return "\n".join(info)

    # ──────── 经济状态 ────────

    def get_economic_status(self) -> Dict[str, Any]:
        """计算当前经济状态：收入、消费能力等"""
        base_income = float(self.economic.get("income", {}).get("value", 0))

        consumption_rate = BASE_CONSUMPTION_RATE
        work_willingness = BASE_WORK_WILLINGNESS
        if self.economic_history:
            latest = self.economic_history[-1]
            consumption_rate = latest.get("consumption_rate", BASE_CONSUMPTION_RATE)
            work_willingness = latest.get("work_willingness", BASE_WORK_WILLINGNESS)

        current_income = base_income * (0.5 + 0.5 * work_willingness)
        consumption_capacity = current_income * consumption_rate
        if self.received_vouchers:
            consumption_capacity += self.voucher_amount

        return {
            "base_income": base_income,
            "current_income": current_income,
            "consumption_rate": consumption_rate,
            "work_willingness": work_willingness,
            "consumption_capacity": consumption_capacity,
            "received_vouchers": self.received_vouchers,
            "voucher_amount": self.voucher_amount if self.received_vouchers else 0
        }

    def update_economic_history(self, decision: Dict[str, Any]):
        """记录经济决策到历史序列，并同步写入短期记忆"""
        try:
            cr = float(decision.get("consumption_rate", BASE_CONSUMPTION_RATE))
        except (TypeError, ValueError):
            cr = BASE_CONSUMPTION_RATE
        try:
            ww = float(decision.get("work_willingness", BASE_WORK_WILLINGNESS))
        except (TypeError, ValueError):
            ww = BASE_WORK_WILLINGNESS

        record = {
            "month": self.current_month,
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "consumption_rate": cr,
            "work_willingness": ww,
            "received_vouchers": self.received_vouchers,
            "voucher_amount": self.voucher_amount if self.received_vouchers else 0
        }
        self.economic_history.append(record)

        if self.memory:
            status = self.get_economic_status()
            content = (f"我的经济状态：基础收入 {status['base_income']}元，"
                       f"当前收入 {status['current_income']}元，"
                       f"消费能力 {status['consumption_capacity']}元")
            if self.received_vouchers:
                content += f"，已领取{self.voucher_amount}元消费券"
            self.memory.add_short_term_memory(
                content, {'type': 'economic_status', 'month': self.current_month})

    # ──────── 消费券 ────────

    def receive_voucher(self, amount: int):
        """接收消费券并写入记忆"""
        self.received_vouchers = True
        self.voucher_amount = amount
        if self.memory:
            text = f"第{self.current_month}月，我收到了{amount}元消费券，可用于增加消费能力"
            meta = {'type': 'economic', 'source': 'policy', 'month': self.current_month}
            self.memory.add_short_term_memory(text, meta)
            self.memory.add_long_term_memory(text, meta)
        logger.info(f"智能体 {self.agent_id} 收到 {amount} 元消费券")

    def reset_voucher(self):
        self.received_vouchers = False
        self.voucher_amount = 0

    # ──────── LLM决策 ────────

    async def make_decision(self, hot_news: str, policy_info: str,
                            use_load_balancer: bool = False) -> AgentDecision:
        """调用LLM生成本月社交帖子与经济决策"""
        agent_info = self.get_info_summary()

        # 决策前触发记忆反思
        if self.memory:
            self.memory.increment_simulation_count()
            await self.memory.try_reflect()

        short_mem = self.memory.get_short_term_memory_text() if self.memory else "没有短期记忆"
        long_mem = await self.memory.get_long_term_memory_text() if self.memory else "没有长期记忆"

        prompt = AGENT_DECISION_PROMPT.format(
            agent_info=agent_info,
            short_term_memory=short_mem,
            long_term_memory=long_mem,
            current_month=self.current_month,
            hot_news=hot_news,
            policy_info=policy_info
        )

        try:
            output = await call_llm_model(
                prompt=prompt, json_response=True, use_load_balancer=use_load_balancer)
        except Exception as e:
            logger.error(f"智能体 {self.agent_id} 决策失败: {e}")
            output = {}

        # 解析
        if isinstance(output, dict) and "error" not in output:
            social_post = output.get("social_post", {})
            if not isinstance(social_post, dict):
                social_post = {"content": str(social_post), "sentiment": 0.5}

            econ = output.get("economic_decision", {})
            if not isinstance(econ, dict):
                econ = {"consumption_rate": BASE_CONSUMPTION_RATE,
                        "work_willingness": BASE_WORK_WILLINGNESS}
            econ.setdefault("consumption_rate", BASE_CONSUMPTION_RATE)
            econ.setdefault("work_willingness", BASE_WORK_WILLINGNESS)

            reasoning = output.get("reasoning", "")
            decision = AgentDecision(social_post=social_post, economic_decision=econ,
                                     reasoning=reasoning)

            # 写入长期记忆
            if self.memory:
                post_text = social_post.get("content", "")
                if post_text:
                    await self.memory.add_long_term_memory(
                        f"第{self.current_month}月，我发表了帖子：{post_text}",
                        {'type': 'social_post', 'month': self.current_month})

                cr = econ.get("consumption_rate", BASE_CONSUMPTION_RATE)
                ww = econ.get("work_willingness", BASE_WORK_WILLINGNESS)
                try:
                    econ_text = (f"第{self.current_month}月，经济决策：消费比例 {float(cr):.2f}，"
                                 f"工作意愿 {float(ww):.2f}")
                except (TypeError, ValueError):
                    econ_text = f"第{self.current_month}月，经济决策：消费比例 {cr}，工作意愿 {ww}"
                await self.memory.add_long_term_memory(
                    econ_text, {'type': 'economic_decision', 'month': self.current_month})

                await self.memory.add_long_term_memory(
                    f"第{self.current_month}月热搜：{hot_news}",
                    {'type': 'hot_news', 'month': self.current_month})
                if policy_info:
                    await self.memory.add_long_term_memory(
                        f"第{self.current_month}月政策：{policy_info}",
                        {'type': 'policy', 'month': self.current_month})

            self.update_economic_history(econ)
            logger.info(f"智能体 {self.agent_id} 决策完成 | "
                        f"情感={social_post.get('sentiment',0.5)} "
                        f"消费率={econ.get('consumption_rate')} "
                        f"工作意愿={econ.get('work_willingness')}")
            return decision

        # 解析失败 -> 默认值
        logger.warning(f"智能体 {self.agent_id} 决策解析失败，使用默认值")
        return AgentDecision(
            social_post={"content": "今天是平常的一天。", "sentiment": 0.5},
            economic_decision={"consumption_rate": BASE_CONSUMPTION_RATE,
                               "work_willingness": BASE_WORK_WILLINGNESS},
            reasoning="决策生成失败，使用默认值")

    # ──────── 情感分析 ────────

    async def analyze_sentiment(self, post_content: str,
                                use_load_balancer: bool = False) -> float:
        """利用LLM对帖子内容做情感打分 (0~1)"""
        prompt = SENTIMENT_ANALYSIS_PROMPT.format(post_content=post_content)
        try:
            result = await call_llm_model(
                prompt=prompt, temperature=0.1, use_load_balancer=use_load_balancer)
        except Exception as e:
            logger.warning(f"情感分析失败: {e}")
            return 0.5

        match = re.search(r"(\d+(\.\d+)?)", str(result))
        if match:
            return max(0.0, min(float(match.group(1)), 1.0))
        return 0.5

    # ──────── 持久化 ────────

    def save(self, output_dir=None):
        """保存智能体当前状态到JSON"""
        output_dir = output_dir or AGENT_POOL_DIR
        os.makedirs(output_dir, exist_ok=True)
        data = {
            "agent_id": self.agent_id,
            "demographic": self.demographic,
            "economic": self.economic,
            "family": self.family,
            "location": self.location,
            "psychological": self.psychological,
            "social": self.social,
            "enriched": self.enriched,
            "current_month": self.current_month,
            "current_time": self.current_time,
            "economic_history": self.economic_history,
            "received_vouchers": self.received_vouchers,
            "voucher_amount": self.voucher_amount
        }
        path = os.path.join(output_dir, f"agent_{self.agent_id}_result.json")
        save_json(data, path)
        logger.info(f"智能体 {self.agent_id} 状态已保存")


# ──────── 批量加载 ────────

def load_agents(agent_ids: List[int], embedding_model=None) -> Dict[int, Agent]:
    """批量加载智能体，跳过加载失败的个体"""
    agents = {}
    for aid in agent_ids:
        try:
            agents[aid] = Agent.from_file(aid, embedding_model)
        except Exception as e:
            logger.error(f"加载智能体 {aid} 失败: {e}")
    logger.info(f"成功加载 {len(agents)} 个智能体")
    return agents
