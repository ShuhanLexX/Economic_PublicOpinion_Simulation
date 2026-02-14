"""
智能体记忆模块：短期记忆 + 基于FAISS向量检索的长期记忆 + 定期反思机制
"""
import asyncio
import datetime
import faiss
from typing import List, Any, Dict, Optional
from pydantic import BaseModel, Field
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from utils.logger import logger
from config.config import MEMORY_REFLECT_INTERVAL, MEMORY_RECENT_LENGTH, MEMORY_INSIGHTS_COUNT
from models.llm_models import glm_model


class TimeWeightedRetriever:
    """基于时间衰减的向量检索器"""

    def __init__(self, vectorstore, decay_rate=0.01, k=5):
        self.vectorstore = vectorstore
        self.decay_rate = decay_rate
        self.k = k
        self.memory_stream: List[Document] = []

    async def aadd_documents(self, documents, current_time=None):
        if current_time is None:
            current_time = datetime.datetime.now()
        for doc in documents:
            doc.metadata.setdefault('last_accessed_at', current_time)
            doc.metadata.setdefault('created_at', current_time)
        self.memory_stream.extend(documents)
        await asyncio.to_thread(self.vectorstore.add_documents, documents)

    async def aget_relevant_documents(self, query, current_time=None):
        return await asyncio.to_thread(self.vectorstore.similarity_search, query, k=self.k)


class AgentMemory(BaseModel):
    """管理智能体的短期记忆列表和FAISS支持的长期记忆"""

    agent_id: int
    long_term_retriever: Any = None
    short_term_memory: List[Document] = Field(default_factory=list)
    embedding_model: Any = None
    simulation_count: int = 0

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        if self.long_term_retriever is None and self.embedding_model is not None:
            self.long_term_retriever = self._create_retriever(self.embedding_model)
            logger.info(f"智能体 {self.agent_id} 记忆模块初始化完成")

    def _create_retriever(self, embedding_model):
        embed_dim = 384
        index = faiss.IndexFlatL2(embed_dim)
        vectorstore = FAISS(
            embedding_model, index, InMemoryDocstore({}), {},
            relevance_score_fn=lambda score: 1.0 - score / 2
        )
        return TimeWeightedRetriever(vectorstore=vectorstore)

    # ──── 长期记忆 ────

    async def add_long_term_memory(self, content: str, metadata: dict = None):
        metadata = metadata or {}
        now = datetime.datetime.now()
        metadata.update({'agent_id': self.agent_id, 'created_at': now})
        doc = Document(page_content=content, metadata=metadata)
        if self.long_term_retriever:
            await self.long_term_retriever.aadd_documents([doc], now)

    async def get_related_memory(self, query: str, k: int = 5) -> List[Document]:
        if not self.long_term_retriever:
            return []
        self.long_term_retriever.k = k
        return await self.long_term_retriever.aget_relevant_documents(query)

    async def get_recent_memory(self, limit: int = 5) -> List[Document]:
        if not self.long_term_retriever or not self.long_term_retriever.memory_stream:
            return []
        sorted_mem = sorted(
            self.long_term_retriever.memory_stream,
            key=lambda x: x.metadata.get('created_at', datetime.datetime.min),
            reverse=True)
        return sorted_mem[:limit]

    async def get_long_term_memory_text(self, limit: int = 8) -> str:
        recent = await self.get_recent_memory(limit)
        if not recent:
            return "没有长期记忆"
        lines = []
        for doc in recent:
            month = doc.metadata.get('month', '')
            prefix = f"[第{month}月]" if month != '' else ""
            lines.append(f"{prefix} {doc.page_content}")
        return "\n".join(lines)

    # ──── 短期记忆 ────

    def add_short_term_memory(self, content: str, metadata: dict = None):
        metadata = metadata or {}
        metadata.update({'agent_id': self.agent_id, 'created_at': datetime.datetime.now()})
        self.short_term_memory.append(Document(page_content=content, metadata=metadata))
        if len(self.short_term_memory) > MEMORY_RECENT_LENGTH:
            self.short_term_memory.pop(0)

    def get_short_term_memory_text(self) -> str:
        entries = []
        for doc in self.short_term_memory:
            if doc.metadata.get('type') == 'agent_info':
                continue
            month = doc.metadata.get('month', '')
            prefix = f"[第{month}月]" if month != '' else ""
            entries.append(f"{prefix} {doc.page_content}")
        return "\n".join(entries) if entries else "没有短期记忆"

    # ──── 反思机制 ────

    def increment_simulation_count(self):
        self.simulation_count += 1

    def should_reflect(self) -> bool:
        return self.simulation_count > 0 and self.simulation_count % MEMORY_REFLECT_INTERVAL == 0

    async def try_reflect(self):
        if self.should_reflect():
            await self.reflect()

    async def reflect(self):
        """基于累积记忆生成高层洞察，写回长期记忆"""
        if not self.long_term_retriever or not self.long_term_retriever.memory_stream:
            return
        logger.info(f"智能体 {self.agent_id} 开始反思 (第{self.simulation_count}次模拟)")

        content = "\n".join(
            doc.page_content for doc in self.long_term_retriever.memory_stream
            if doc.metadata.get('type') != 'reflection')
        if not content.strip():
            return

        insights = await self._generate_insights(content)
        for ins in insights:
            now = datetime.datetime.now()
            meta = {'agent_id': self.agent_id, 'created_at': now,
                    'type': 'reflection', 'simulation_round': self.simulation_count}
            await self.long_term_retriever.aadd_documents(
                [Document(page_content=ins, metadata=meta)], now)
        logger.info(f"智能体 {self.agent_id} 反思完成，生成 {len(insights)} 条洞察")

    async def _generate_insights(self, memory_content: str) -> List[str]:
        if not memory_content.strip():
            return []
        prompt = (f"根据以下历史记忆，生成{MEMORY_INSIGHTS_COUNT}条重要的个人洞察或总结：\n\n"
                  f"{memory_content}\n\n"
                  f"请直接列出洞察内容，每条洞察以\"洞察：\"开头，简洁清晰，不超过30字。")
        try:
            response = await glm_model.generate(prompt)
        except Exception as e:
            logger.warning(f"智能体 {self.agent_id} 洞察生成失败: {e}")
            return []

        insights = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith("洞察:") or line.startswith("洞察："):
                text = line.replace("洞察:", "").replace("洞察：", "").strip()
                if text:
                    insights.append(f"反思洞察：{text}")
        if not insights and response.strip():
            insights = [f"反思洞察：{response.strip()}"]
        return insights[:MEMORY_INSIGHTS_COUNT]
