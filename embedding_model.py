"""
Embedding模型加载与文本向量化
"""
import numpy as np
from typing import List, Union, Tuple

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HUGGINGFACE_EMBEDDINGS_AVAILABLE = True
except ImportError:
    HUGGINGFACE_EMBEDDINGS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from config import DEFAULT_EMBEDDING_DIM, EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_DIMENSION
from logger import logger


def get_embedding_model() -> Tuple[int, object]:
    """加载embedding模型，按优先级尝试 HuggingFaceEmbeddings -> SentenceTransformer -> 哈希回退"""
    dim = EMBEDDING_MODEL_DIMENSION

    if HUGGINGFACE_EMBEDDINGS_AVAILABLE:
        try:
            model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_PATH,
                model_kwargs={'device': 'cpu'}
            )
            logger.info(f"已加载HuggingFaceEmbeddings: {EMBEDDING_MODEL_PATH}")
            return dim, model
        except Exception as e:
            logger.warning(f"HuggingFaceEmbeddings加载失败: {e}")

    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            model = SentenceTransformer(EMBEDDING_MODEL_PATH)
            logger.info(f"已加载SentenceTransformer: {EMBEDDING_MODEL_PATH}")
            return dim, model
        except Exception as e:
            logger.warning(f"SentenceTransformer加载失败: {e}")

    # 回退：基于哈希的伪嵌入（仅供测试）
    logger.warning("使用哈希伪嵌入模型（仅用于开发测试）")

    class SimpleHashEmbedding:
        """基于字符串哈希生成固定维度伪向量，仅在无GPU/模型环境下回退使用"""
        def __init__(self, d):
            self.dim = d

        def encode(self, sentences: Union[str, List[str]], **kwargs) -> np.ndarray:
            if isinstance(sentences, str):
                sentences = [sentences]
            result = []
            for s in sentences:
                np.random.seed(hash(s) % (2**31))
                vec = np.random.uniform(-1, 1, self.dim)
                vec /= np.linalg.norm(vec)
                result.append(vec)
            return np.array(result)

    return dim, SimpleHashEmbedding(dim)


def embed_texts(texts: Union[str, List[str]], model) -> np.ndarray:
    """对文本列表进行向量化，自动适配不同模型接口"""
    if isinstance(texts, str):
        texts = [texts]

    if hasattr(model, 'embed_documents'):
        return np.array(model.embed_documents(texts))
    elif hasattr(model, 'encode'):
        return model.encode(texts)
    else:
        logger.warning("模型接口不兼容，返回随机向量")
        return np.random.rand(len(texts), EMBEDDING_MODEL_DIMENSION)
