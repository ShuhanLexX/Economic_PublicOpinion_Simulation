"""
舆论子系统：热搜新闻管理、社交帖子发布与情感统计
"""
import json
import os
import random
from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel

from logger import logger
from utils import load_json, save_json
from config import DATA_DIR, USE_INIT_SOCIAL_POSTS


class SocialPost(BaseModel):
    """社交媒体帖子"""
    post_id: str
    agent_id: int
    content: str
    sentiment: float
    month: int
    timestamp: str

    @classmethod
    def create(cls, agent_id: int, content: str, sentiment, month: int):
        """创建帖子，自动将文本情感标签转换为数值"""
        # 文本情感标签 -> 数值映射
        _LABEL_MAP = {
            "愤怒": 0.1, "生气": 0.1, "不满": 0.2, "失望": 0.2,
            "担忧": 0.3, "焦虑": 0.3, "紧张": 0.3,
            "中性": 0.5, "平静": 0.5, "一般": 0.5,
            "满意": 0.7, "开心": 0.8, "高兴": 0.8, "快乐": 0.9, "喜悦": 0.9
        }
        try:
            val = _LABEL_MAP[sentiment] if isinstance(sentiment, str) else float(sentiment)
            val = max(0.0, min(1.0, val))
        except (ValueError, TypeError, KeyError):
            val = 0.5

        return cls(
            post_id=f"post_{agent_id}_{month}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            agent_id=agent_id,
            content=content,
            sentiment=val,
            month=month,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )


class SocialSystem:
    """舆论子系统：管理热搜新闻分发和社交帖子生命周期"""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(DATA_DIR, "social_data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.hot_news_file = os.path.join(self.data_dir, "hot_news.json")
        self.social_posts_file = os.path.join(self.data_dir, "social_posts.json")

        self.hot_news = self._load_hot_news()
        self.social_posts: Dict[int, List[SocialPost]] = (
            self._load_social_posts() if USE_INIT_SOCIAL_POSTS else {})
        logger.info("舆论子系统初始化完成")

    # ──── 热搜新闻 ────

    def _load_hot_news(self) -> Dict[int, str]:
        if os.path.exists(self.hot_news_file):
            raw = load_json(self.hot_news_file)
            news = {}
            for k, v in raw.items():
                try:
                    news[int(k)] = v
                except (ValueError, TypeError):
                    pass
            if news:
                logger.info(f"已加载热搜新闻 {len(news)} 条")
                return news

        # 默认新闻序列（模拟疫情-消费券政策周期）
        default = {
            0: "春节旅游热度持续上升，各地旅游景点人气旺盛。",
            1: "发现不明原因肺炎病例，专家正在调查病因。",
            2: "病毒疫情爆发，全国多地启动应急响应机制。",
            3: "全国肺炎确诊病例持续增长，各地实施严格防控措施。",
            4: "国家发布消费券政策，鼓励民众消费促进经济复苏。",
            5: "疫情形势依然严峻，第二批消费券发放开始。",
            6: "肺炎病例出现反弹，消费券政策继续实施。",
            7: "疫情防控进入常态化，消费券政策暂停。",
            8: "全国疫情防控成效显著，经济逐步恢复。",
        }
        save_json(default, self.hot_news_file)
        return default

    def get_hot_news(self, month: int) -> str:
        if month in self.hot_news:
            return self.hot_news[month]
        str_month = str(month)
        if str_month in self.hot_news:
            return self.hot_news[str_month]
        logger.warning(f"月份 {month} 无热搜新闻")
        return "今日无热搜新闻。"

    # ──── 政策信息 ────

    def get_policy_info(self, month: int) -> str:
        path = os.path.join(DATA_DIR, 'social_data', 'policy_news.json')
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get(str(month), "")
        except Exception:
            return ""

    # ──── 社交帖子 ────

    def _load_social_posts(self) -> Dict[int, List[SocialPost]]:
        default: Dict[int, List[SocialPost]] = {i: [] for i in range(9)}
        if os.path.exists(self.social_posts_file):
            try:
                raw = load_json(self.social_posts_file)
                return {int(m): [SocialPost(**p) for p in posts] for m, posts in raw.items()}
            except Exception as e:
                logger.warning(f"加载社交帖子失败: {e}")

        # 初始帖子
        for item in [
            {"agent_id": 0, "content": "新年快乐！祝大家春节假期愉快！", "sentiment": 0.9, "month": 0},
            {"agent_id": 0, "content": "最近天气不错，准备出去旅游一趟。", "sentiment": 0.8, "month": 0},
            {"agent_id": 0, "content": "春节期间，家里来了很多亲戚，忙碌但开心。", "sentiment": 0.7, "month": 0},
        ]:
            default[item["month"]].append(SocialPost.create(**item))
        self.save_social_posts(default)
        return default

    def add_social_post(self, agent_id: int, content: str, sentiment: float, month: int):
        if not content or not content.strip():
            return None
        post = SocialPost.create(agent_id, content, sentiment, month)
        self.social_posts.setdefault(month, []).append(post)
        self.save_social_posts(self.social_posts)
        return post

    def get_random_posts(self, month: int, count: int = 3,
                         exclude_agent_id: Optional[int] = None) -> List[SocialPost]:
        """随机采样帖子供智能体阅读，模拟信息流推荐"""
        posts = self.social_posts.get(month, [])
        if not posts:
            posts = self.social_posts.get(max(0, month - 1), [])
        if exclude_agent_id is not None:
            posts = [p for p in posts if p.agent_id != exclude_agent_id]
        if len(posts) <= count:
            return posts
        return random.sample(posts, count)

    def get_sentiment_stats(self, month: int) -> Dict[str, float]:
        posts = self.social_posts.get(month, [])
        if not posts:
            return {"average": 0.5, "count": 0}
        sentiments = [p.sentiment for p in posts]
        return {"average": sum(sentiments) / len(sentiments), "count": len(posts)}

    def save_social_posts(self, posts: Dict[int, List[SocialPost]] = None,
                          custom_result_dir: str = None):
        posts = posts or self.social_posts
        serializable = {str(m): [p.dict() for p in ps] for m, ps in posts.items()}
        save_json(serializable, self.social_posts_file)
        if custom_result_dir:
            os.makedirs(custom_result_dir, exist_ok=True)
            save_json(serializable, os.path.join(custom_result_dir, "social_posts.json"))
