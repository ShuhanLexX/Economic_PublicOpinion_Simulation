"""
全局配置：模拟参数、模型设置、路径定义
"""
import os
from pathlib import Path
from datetime import datetime

# ============ 路径配置 ============
BASE_DIR = Path(__file__).parent
DATA_DIR = os.path.join(BASE_DIR, "data")
AGENT_POOL_DIR = os.path.join(DATA_DIR, "agent_pool")
RESULT_DIR = os.path.join(DATA_DIR, "results")
MODEL_DIR = os.path.join(DATA_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

for dir_path in [DATA_DIR, AGENT_POOL_DIR, RESULT_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============ 模拟参数 ============
AGENT_COUNT = 1000              # 智能体数量
SIMULATION_MONTHS = 13          # 模拟月数
VOUCHER_START_MONTH = 4         # 消费券发放起始月
VOUCHER_END_MONTH = 6           # 消费券发放终止月
VOUCHER_AMOUNT = 500            # 消费券面额（元）
RANDOM_SEED = 5

# ============ LLM 与 Embedding 参数 ============
DEFAULT_EMBEDDING_DIM = 1536
FAISS_INDEX_TYPE = "Flat"
MAX_CONCURRENT_REQUESTS = 50
BATCH_DELAY_SECONDS = 1.0
LLM_DEFAULT_TEMPERATURE = 0.9
LLM_MAX_RETRY = 5
LLM_RETRY_DELAY = 1.0
TOKEN_LIMIT = 32768

# ============ 记忆模块参数 ============
MEMORY_RETRIEVAL_COUNT = 8
SHORT_TERM_MEMORY_RETENTION = 45
MAX_MEMORY_IMPORTANCE = 15
MIN_MEMORY_IMPORTANCE = 0
MEMORY_REFLECT_INTERVAL = 4     # 每N次模拟触发一次反思
MEMORY_RECENT_LENGTH = 10       # 短期记忆容量
MEMORY_INSIGHTS_COUNT = 3       # 每次反思生成的洞察数量

# ============ 舆论子系统参数 ============
USE_SOCIAL_SYSTEM = True
USE_INIT_SOCIAL_POSTS = False
DEFAULT_SENTIMENT_SCORE = 0.5
POST_BATCH_SIZE = 10

# ============ 经济子系统参数 ============
BASE_CONSUMPTION_RATE = 0.6
BASE_WORK_WILLINGNESS = 0.7

# ============ API 配置（通过环境变量设置） ============
GLM_BASE_URL = os.environ.get("GLM_BASE_URL", "http://localhost:3000/v1")
GLM_API_KEY = os.environ.get("GLM_API_KEY", "your-api-key")
GLM_MODEL_NAME = os.environ.get("GLM_MODEL_NAME", "glm-4-flash")

LLM_API_URL = os.environ.get("LLM_API_URL", GLM_BASE_URL)
LLM_API_KEY = os.environ.get("LLM_API_KEY", GLM_API_KEY)

EMBEDDING_API_URL = os.environ.get("EMBEDDING_API_URL", "http://localhost:8100/v1")
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "your-api-key")

# 负载均衡多端点配置（可按需扩展）
NEW_API_MODELS = [
    {"name": "default", "path": "", "weight": 2},
]

# ============ 本地 Embedding 模型 ============
EMBEDDING_MODEL_PATH = os.environ.get(
    "EMBEDDING_MODEL_PATH",
    "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_MODEL_DIMENSION = 384

# ============ 智能体文件格式 ============
AGENT_FILE_PREFIX = "agent_"
AGENT_FILE_SUFFIX = ".json"
CURRENT_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ============ 日志配置 ============
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(LOG_DIR, f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
