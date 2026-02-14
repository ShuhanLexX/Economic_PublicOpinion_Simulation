"""
模型资源清理
"""
from utils.logger import logger
from models.llm_models import glm_model, load_balanced_model


async def cleanup_model_resources():
    """关闭所有LLM HTTP会话"""
    try:
        await load_balanced_model.close()
        await glm_model.close()
        logger.info("模型资源已清理")
    except Exception as e:
        logger.error(f"清理模型资源时出错: {e}")
