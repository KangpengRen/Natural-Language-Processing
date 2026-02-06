"""
超参数等配置信息
"""
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
ROW_DATA_DIR = ROOT_DIR / "data" / "row"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"

SEQ_LEN = 5  # 句子窗口长度

BATCH_SIZE = 64  # 批量大小

EMBEDDING_DIM = 128  # 词向量维度

HIDDEN_SIZE = 256  # 隐藏层维度

LEARNING_RATE = 1e-3  # 学习率

EPOCHS = 10  # 训练轮次
