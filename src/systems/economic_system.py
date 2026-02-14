"""
经济子系统：消费券发放、经济统计与政策效果评估
"""
import os
import numpy as np
from typing import Dict, List, Any
from datetime import datetime

from utils.logger import logger
from utils.utils import load_json, save_json, truncate_float
from config.config import RESULT_DIR, VOUCHER_AMOUNT, VOUCHER_START_MONTH, VOUCHER_END_MONTH


class EconomicSystem:
    """管理经济数据采集、消费券政策执行和统计分析"""

    def __init__(self, result_dir: str = None):
        self.result_dir = result_dir or RESULT_DIR
        os.makedirs(self.result_dir, exist_ok=True)

        self.monthly_data: Dict[int, Dict] = {}
        self.agents_data: Dict[int, List[Dict]] = {}

        self.voucher_amount = VOUCHER_AMOUNT
        self.voucher_start_month = VOUCHER_START_MONTH
        self.voucher_end_month = VOUCHER_END_MONTH
        self.total_policy_investment = 0

        logger.info("经济子系统初始化完成")

    def should_issue_voucher(self, month: int) -> bool:
        return self.voucher_start_month <= month <= self.voucher_end_month

    def update_agent_data(self, agent_id: int, month: int, data: Dict[str, Any]):
        """记录单个智能体的月度经济数据"""
        self.agents_data.setdefault(agent_id, [])
        data.update({"agent_id": agent_id, "month": month,
                     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        self.agents_data[agent_id].append(data)

    def calculate_monthly_stats(self, month: int, agents_status: Dict[int, Dict[str, Any]]):
        """汇总月度经济统计指标"""
        vals = list(agents_status.values())
        self.monthly_data[month] = {
            "month": month,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "agent_count": len(vals),
            "avg_base_income": truncate_float(np.mean([v["base_income"] for v in vals])),
            "avg_current_income": truncate_float(np.mean([v["current_income"] for v in vals])),
            "avg_consumption_rate": truncate_float(np.mean([v["consumption_rate"] for v in vals])),
            "avg_work_willingness": truncate_float(np.mean([v["work_willingness"] for v in vals])),
            "total_consumption_capacity": truncate_float(sum(v["consumption_capacity"] for v in vals)),
            "avg_consumption_capacity": truncate_float(np.mean([v["consumption_capacity"] for v in vals])),
            "voucher_issued": self.should_issue_voucher(month)
        }
        logger.info(f"月份 {month} 经济统计: 平均消费能力 "
                    f"{self.monthly_data[month]['avg_consumption_capacity']}元")

    def issue_vouchers(self, agents_dict: Dict[int, Any]) -> int:
        """向所有智能体发放消费券"""
        for agent in agents_dict.values():
            agent.receive_voucher(self.voucher_amount)
        total = len(agents_dict) * self.voucher_amount
        self.total_policy_investment += total
        logger.info(f"发放消费券: {len(agents_dict)}人, 总额 {total}元")
        return total

    def get_consumption_change_rate(self, month: int) -> float:
        if month <= 1 or month not in self.monthly_data or (month - 1) not in self.monthly_data:
            return 0.0
        cur = self.monthly_data[month]["total_consumption_capacity"]
        prev = self.monthly_data[month - 1]["total_consumption_capacity"]
        return truncate_float((cur - prev) / prev) if prev != 0 else 0.0

    def get_policy_leverage_ratio(self) -> float:
        """计算政策撬动比 = (政策后消费 - 政策前消费) / 政策投入"""
        if self.total_policy_investment == 0:
            return 0.0
        pre = self.voucher_start_month - 1
        post = self.voucher_end_month
        if pre not in self.monthly_data or post not in self.monthly_data:
            return 0.0
        delta = (self.monthly_data[post]["total_consumption_capacity"]
                 - self.monthly_data[pre]["total_consumption_capacity"])
        return truncate_float(delta / self.total_policy_investment)

    def get_urban_rural_difference(self, month: int, agents_dict: Dict[int, Any]) -> Dict[str, Any]:
        """城乡消费差异分析"""
        urban, rural = [], []
        for agent in agents_dict.values():
            cap = agent.get_economic_status()["consumption_capacity"]
            if agent.location.get("urban20", {}).get("value", 0) == 1:
                urban.append(cap)
            else:
                rural.append(cap)
        avg_u = truncate_float(np.mean(urban)) if urban else 0
        avg_r = truncate_float(np.mean(rural)) if rural else 0
        return {
            "month": month,
            "avg_urban_consumption": avg_u,
            "avg_rural_consumption": avg_r,
            "urban_rural_ratio": truncate_float(avg_u / avg_r) if avg_r > 0 else 0,
            "urban_count": len(urban),
            "rural_count": len(rural)
        }

    def get_demographic_analysis(self, month: int, agents_dict: Dict[int, Any]) -> Dict:
        """按年龄段和教育水平的分组统计"""
        age_groups = {"18-30": {"c": [], "w": []}, "31-45": {"c": [], "w": []},
                      "46-60": {"c": [], "w": []}, "60+": {"c": [], "w": []}}
        edu_groups = {"初中及以下": {"c": [], "w": []}, "高中/中专": {"c": [], "w": []},
                      "大专/本科": {"c": [], "w": []}, "研究生及以上": {"c": [], "w": []}}

        for agent in agents_dict.values():
            s = agent.get_economic_status()
            cap, ww = s["consumption_capacity"], s["work_willingness"]

            age = agent.demographic.get("age", {}).get("value", 30)
            ag = "18-30" if age <= 30 else "31-45" if age <= 45 else "46-60" if age <= 60 else "60+"
            age_groups[ag]["c"].append(cap)
            age_groups[ag]["w"].append(ww)

            edu = agent.demographic.get("w01", {}).get("value", 4)
            eg = ("初中及以下" if edu <= 4 else "高中/中专" if edu <= 6
                  else "大专/本科" if edu <= 8 else "研究生及以上")
            edu_groups[eg]["c"].append(cap)
            edu_groups[eg]["w"].append(ww)

        def _summarize(groups):
            return {k: {"avg_consumption": truncate_float(np.mean(v["c"])) if v["c"] else 0,
                         "avg_work_willingness": truncate_float(np.mean(v["w"])) if v["w"] else 0,
                         "count": len(v["c"])} for k, v in groups.items()}

        return {"age_groups": _summarize(age_groups),
                "education_groups": _summarize(edu_groups)}

    def get_income_consumption_correlation(self, agents_dict: Dict[int, Any]) -> List[Dict[str, float]]:
        return [{"income": truncate_float(a.get_economic_status()["base_income"]),
                 "consumption": truncate_float(a.get_economic_status()["consumption_capacity"])}
                for a in agents_dict.values()]

    def save_results(self, custom_result_dir: str = None):
        d = custom_result_dir or self.result_dir
        os.makedirs(d, exist_ok=True)
        save_json(self.monthly_data, os.path.join(d, "monthly_economic_data.json"))
        save_json(self.agents_data, os.path.join(d, "agent_economic_data.json"))
        logger.info(f"经济数据已保存至 {d}")

    def generate_report(self, agents_dict: Dict[int, Any]) -> Dict[str, Any]:
        if not self.monthly_data:
            return {"error": "数据不足"}
        latest = max(self.monthly_data.keys())
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "months_simulated": latest + 1,
            "monthly_stats": self.monthly_data,
            "policy_leverage_ratio": self.get_policy_leverage_ratio(),
            "total_policy_investment": self.total_policy_investment,
            "urban_rural_difference": self.get_urban_rural_difference(latest, agents_dict),
            "demographic_analysis": self.get_demographic_analysis(latest, agents_dict),
            "income_consumption_correlation": self.get_income_consumption_correlation(agents_dict)
        }
        save_json(report, os.path.join(self.result_dir, "economic_report.json"))
        logger.info("经济报告已生成")
        return report
