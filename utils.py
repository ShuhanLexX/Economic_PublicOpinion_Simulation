"""
通用工具函数
"""
import asyncio
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Awaitable, TypeVar

try:
    from config import BATCH_DELAY_SECONDS
except ImportError:
    BATCH_DELAY_SECONDS = 2.0

T = TypeVar('T')


def load_json(file_path: str, default=None):
    """加载JSON文件，文件不存在或解析失败时返回default"""
    if not os.path.exists(file_path):
        return default if default is not None else {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载JSON失败 {file_path}: {e}")
        return default if default is not None else {}


def save_json(data, file_path: str, ensure_dir: bool = True) -> bool:
    """保存数据为JSON文件"""
    if ensure_dir:
        d = os.path.dirname(file_path)
        if d:
            os.makedirs(d, exist_ok=True)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存JSON失败 {file_path}: {e}")
        return False


def convert_datetime_to_str(dt_obj):
    if isinstance(dt_obj, str):
        return dt_obj
    return dt_obj.strftime('%Y-%m-%d %H:%M:%S')


async def run_concurrently(tasks: List[Awaitable[T]], max_concurrency: int = 10) -> List[T]:
    """带并发上限和超时控制的批量异步执行"""
    if not tasks:
        return []

    semaphore = asyncio.Semaphore(max_concurrency)
    results = []

    async def _run(task):
        async with semaphore:
            try:
                return await asyncio.wait_for(task, timeout=60)
            except (asyncio.TimeoutError, Exception) as e:
                print(f"任务异常: {e}")
                return None

    batch_size = max_concurrency * 2
    total = (len(tasks) + batch_size - 1) // batch_size
    for idx, i in enumerate(range(0, len(tasks), batch_size)):
        batch = tasks[i:i + batch_size]
        batch_results = await asyncio.gather(*[_run(t) for t in batch])
        results.extend(batch_results)
        if idx < total - 1:
            await asyncio.sleep(BATCH_DELAY_SECONDS)
    return results


def truncate_float(value, decimals=2):
    factor = 10 ** decimals
    return int(value * factor) / factor
