"""
主程序入口：消费券政策模拟实验的执行流程控制
"""
import os
import asyncio
import random
import argparse
from typing import Dict, Any
from datetime import datetime

from config.config import (
    SIMULATION_MONTHS, VOUCHER_START_MONTH, VOUCHER_END_MONTH,
    AGENT_POOL_DIR, RESULT_DIR, AGENT_COUNT,
    USE_SOCIAL_SYSTEM, USE_INIT_SOCIAL_POSTS, RANDOM_SEED, MAX_CONCURRENT_REQUESTS
)
from utils.logger import logger
from utils.utils import run_concurrently, save_json
from utils.model_utils import cleanup_model_resources
from models.embedding_model import get_embedding_model
from agents.agent import Agent, load_agents
from systems.social_system import SocialSystem
from systems.economic_system import EconomicSystem


async def initialize_systems(embedding_dim, embedding_model, agent_count=None):
    """初始化智能体群体与子系统"""
    logger.info(f"初始化系统 | 月数={SIMULATION_MONTHS}, 消费券月份=[{VOUCHER_START_MONTH},{VOUCHER_END_MONTH}], "
                f"智能体={AGENT_COUNT}, 舆论系统={USE_SOCIAL_SYSTEM}")

    os.makedirs(RESULT_DIR, exist_ok=True)
    n = agent_count or AGENT_COUNT

    # 加载智能体
    agent_files = []
    if os.path.exists(AGENT_POOL_DIR):
        for f in os.listdir(AGENT_POOL_DIR):
            if f.startswith("agent_") and f.endswith(".json") and "result" not in f:
                agent_files.append(int(f.replace("agent_", "").replace(".json", "")))

    if len(agent_files) < n:
        logger.warning(f"可用智能体不足({len(agent_files)}), 使用示例智能体填充")
        agent_files = [101129501] * n
    else:
        random.seed(RANDOM_SEED)
        agent_files = random.sample(agent_files, n)

    agents_dict = load_agents(agent_files, embedding_model)
    logger.info(f"已加载 {len(agents_dict)} 个智能体")

    return agents_dict, SocialSystem(), EconomicSystem()


async def run_monthly_simulation(month: int, agents_dict: Dict[int, Agent],
                                 social_system: SocialSystem,
                                 economic_system: EconomicSystem):
    """单月模拟：信息注入 -> 消费券发放 -> 并发决策 -> 数据汇总"""
    logger.info(f"======== 月份 {month} 开始 ========")

    for agent in agents_dict.values():
        agent.current_month = month

    hot_news = social_system.get_hot_news(month)
    policy_info = social_system.get_policy_info(month)

    # 消费券发放
    if economic_system.should_issue_voucher(month):
        total = economic_system.issue_vouchers(agents_dict)
        logger.info(f"月份 {month} 发放消费券: {total}元")

    # 构建决策任务
    tasks, agents_list = [], []
    for aid, agent in agents_dict.items():
        if USE_SOCIAL_SYSTEM:
            for i, post in enumerate(social_system.get_random_posts(month, 5, exclude_agent_id=aid)):
                if agent.memory:
                    agent.memory.add_short_term_memory(
                        f"第{month}月，看到帖子{i+1}：{post.content}",
                        {'type': 'social_post', 'source': 'recommendation',
                         'post_id': post.post_id, 'month': month})
        tasks.append(agent.make_decision(hot_news, policy_info, use_load_balancer=True))
        agents_list.append((aid, agent))

    decisions = await run_concurrently(tasks, max_concurrency=MAX_CONCURRENT_REQUESTS)

    # 处理决策结果
    eco_status = {}
    failed = []

    for i, (aid, agent) in enumerate(agents_list):
        d = decisions[i] if i < len(decisions) else None
        ok = (d is not None
              and hasattr(d, 'economic_decision') and isinstance(d.economic_decision, dict)
              and hasattr(d, 'social_post') and isinstance(d.social_post, dict))
        if not ok:
            failed.append((aid, agent))
            continue

        content = d.social_post.get("content", "")
        if content and content.strip():
            social_system.add_social_post(aid, content, d.social_post.get("sentiment", 0.5), month)

        status = agent.get_economic_status()
        eco_status[aid] = status
        economic_system.update_agent_data(aid, month, {
            k: status[k] for k in ["base_income", "current_income", "consumption_rate",
                                     "work_willingness", "consumption_capacity",
                                     "received_vouchers", "voucher_amount"]})

    # 重试失败的智能体（最多3轮）
    for retry in range(1, 4):
        if not failed:
            break
        logger.info(f"第{retry}次重试，剩余{len(failed)}个智能体")
        retry_tasks = [a.make_decision(hot_news, policy_info, True) for _, a in failed]
        retry_results = await run_concurrently(retry_tasks, max_concurrency=MAX_CONCURRENT_REQUESTS)
        still_failed = []
        for j, (aid, agent) in enumerate(failed):
            rd = retry_results[j] if j < len(retry_results) else None
            ok = (rd is not None
                  and hasattr(rd, 'economic_decision') and isinstance(rd.economic_decision, dict)
                  and hasattr(rd, 'social_post') and isinstance(rd.social_post, dict))
            if ok:
                c = rd.social_post.get("content", "")
                if c and c.strip():
                    social_system.add_social_post(aid, c, rd.social_post.get("sentiment", 0.5), month)
                status = agent.get_economic_status()
                eco_status[aid] = status
                economic_system.update_agent_data(aid, month, {
                    k: status[k] for k in ["base_income", "current_income", "consumption_rate",
                                             "work_willingness", "consumption_capacity",
                                             "received_vouchers", "voucher_amount"]})
            else:
                still_failed.append((aid, agent))
        failed = still_failed

    # 仍然失败的使用默认状态
    for aid, agent in failed:
        logger.warning(f"智能体 {aid} 多次重试失败，使用默认状态")
        status = agent.get_economic_status()
        eco_status[aid] = status
        economic_system.update_agent_data(aid, month, {
            k: status[k] for k in ["base_income", "current_income", "consumption_rate",
                                     "work_willingness", "consumption_capacity",
                                     "received_vouchers", "voucher_amount"]})

    economic_system.calculate_monthly_stats(month, eco_status)

    if month > VOUCHER_END_MONTH:
        for agent in agents_dict.values():
            agent.reset_voucher()

    return {
        "month": month,
        "sentiment_stats": social_system.get_sentiment_stats(month),
        "consumption_change_rate": economic_system.get_consumption_change_rate(month)
    }


async def run_simulation(agent_count=None, simulation_months=None):
    """执行完整的多月模拟并输出结果"""
    t0 = datetime.now()
    n_agents = agent_count or AGENT_COUNT
    n_months = simulation_months or SIMULATION_MONTHS

    logger.info(f"模拟开始 | 智能体={n_agents}, 月数={n_months}")

    embedding_dim, embedding_model = get_embedding_model()
    agents_dict, social_system, economic_system = await initialize_systems(
        embedding_dim, embedding_model, n_agents)

    monthly_results = []
    sentiment_data = {}

    for month in range(n_months):
        result = await run_monthly_simulation(month, agents_dict, social_system, economic_system)
        monthly_results.append(result)
        sentiment_data[month] = result["sentiment_stats"]

        stats = economic_system.monthly_data.get(month, {})
        logger.info(f"月份 {month} 完成 | 消费变化率={result['consumption_change_rate']:.4f}, "
                    f"情感均值={result['sentiment_stats']['average']:.2f}, "
                    f"平均消费率={stats.get('avg_consumption_rate', 0):.2f}, "
                    f"平均收入={stats.get('avg_current_income', 0):.0f}元")

    # 保存结果
    result_dir = os.path.join(RESULT_DIR, t0.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(result_dir, exist_ok=True)

    # 保存智能体状态
    complete_data = {}
    for aid, agent in agents_dict.items():
        agent.save(result_dir)
        complete_data[aid] = {
            "agent_id": agent.agent_id,
            "demographic": agent.demographic,
            "economic": agent.economic,
            "family": agent.family,
            "location": agent.location,
            "psychological": agent.psychological,
            "social": agent.social,
            "economic_status": agent.get_economic_status(),
            "economic_history": agent.economic_history,
            "current_month": agent.current_month
        }

    report = economic_system.generate_report(agents_dict)
    economic_system.save_results(result_dir)
    social_system.save_social_posts(custom_result_dir=result_dir)
    save_json(complete_data, os.path.join(result_dir, "complete_agent_data.json"))

    sim_results = {
        "agents_count": len(agents_dict),
        "months_simulated": n_months,
        "monthly_results": monthly_results,
        "economic_report": report,
        "sentiment_data": sentiment_data
    }
    save_json(sim_results, os.path.join(result_dir, "simulation_results.json"))

    logger.info(f"模拟完成 | 用时 {(datetime.now()-t0).total_seconds():.1f}s, 结果保存至 {result_dir}")
    return sim_results


async def main():
    parser = argparse.ArgumentParser(description="消费券政策效果模拟评估")
    parser.add_argument("--agents", type=int, default=AGENT_COUNT, help="智能体数量")
    parser.add_argument("--months", type=int, default=SIMULATION_MONTHS, help="模拟月数")
    args = parser.parse_args()

    try:
        results = await run_simulation(agent_count=args.agents, simulation_months=args.months)
        leverage = results["economic_report"].get("policy_leverage_ratio", 0)
        logger.info(f"消费券政策撬动比: {leverage:.2f}")
        return results
    finally:
        await cleanup_model_resources()


if __name__ == "__main__":
    asyncio.run(main())
