"""
日志配置
"""
import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logger = logging.getLogger("simulation")
logger.setLevel(logging.INFO)

_fmt = logging.Formatter(
    '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(_fmt)

_file = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
_file.setLevel(logging.INFO)
_file.setFormatter(_fmt)

logger.addHandler(_console)
logger.addHandler(_file)


def get_logger(name=None):
    return logger if name is None else logger.getChild(name)
