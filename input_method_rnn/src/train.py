"""
训练脚本
"""
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from input_method_rnn.src.tokenizer import JiebaTokenizer
from model import InputMethodModel
from dataset import get_dataloader
from config import MODELS_DIR, LEARNING_RATE, EPOCHS, LOGS_DIR


def train():
    # 1. 确定训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 获取数据集
    dataloader = get_dataloader(train=True)

    # 3. 加载词表
    tokenizer = JiebaTokenizer.from_vocab(MODELS_DIR / "vocab.txt")

    # 4. 构建模型
    model = InputMethodModel(tokenizer.vocab_size)

    # 5. 定义损失函数
    loss_fn = nn.CrossEntropyLoss()

    # 6. 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # tensorboard writer
    writer = SummaryWriter(log_dir=LOGS_DIR)

    # 开始训练
    best_loss = float("inf")
    for epoch in range(EPOCHS):
        print("=" * 10, f"EPOCH: {epoch + 1}", "=" * 10)
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f"loss: {loss}")

        # 记录训练结果
        writer.add_scalar("loss", loss, epoch)

        # 保存模型
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), MODELS_DIR / "best_model.pth")
            print("模型参数更新成功")

        writer.close()


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    训练一个轮次
    :param model: 模型
    :param dataloader: 数据集
    :param loss_fn: 损失函数
    :param optimizer: 优化器
    :param device: 训练设备
    :return: 当前epoch的平均loss
    """
    model.train()  # 将模型设置为训练模式
    model.to(device)  # 将模型放入训练设备

    total_loss = 0
    for inputs, targets in tqdm(dataloader, desc="训练"):
        inputs = inputs.to(device)  # 输入放入训练设备中     shape: (batch_size, seq_len)
        targets = targets.to(device)  # 输出放入训练设备中    shape: (batch_size)

        # 前向传播
        outputs = model(inputs) # 前向传播结果 shape: (batch_size, vocab_size)

        # 计算损失值
        loss = loss_fn(outputs, targets)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
    return total_loss / len(dataloader)

if __name__ == '__main__':
    train()
