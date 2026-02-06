"""
模型预测
"""
import jieba
import torch

from config import MODELS_DIR
from model import InputMethodModel


def predict(text):
    # 1. 定义预测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 加载词表
    # with open(MODELS_DIR / "vocab.txt", "r", encoding="utf-8") as f:
    #     vocab_list = (line.strip() for line in f.readlines())
    with open(MODELS_DIR / "vocab.txt", "r", encoding="utf-8") as f:
        vocab_list = [line.strip() for line in f.readlines()]
    index2token = {index: token for index, token in enumerate(vocab_list)}  # index -> token 表
    token2index = {token: index for index, token in enumerate(vocab_list)}  # token -> index 表

    # 3. 加载模型
    model = InputMethodModel(vocab_size=len(vocab_list)).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "best_model.pth"))

    # 4. 输入处理
    tokens = jieba.lcut(text)
    indexes = [token2index.get(token, 0) for token in tokens]
    input_tensor = torch.tensor([indexes], dtype=torch.long)  # 转化为二维张量

    # 5. 预测逻辑
    model.eval()  # 开启模型预测模式
    with torch.no_grad():
        output = model(input_tensor)  # (batch_size, vocab_size)
        # 取前5最高预测值
        top5_indexes = torch.topk(input=output, k=5, dim=1)  # (batch_size, k)
        top5_indexes_list = top5_indexes.tolist()
        top5_tokens = [index2token.get(index) for index in top5_indexes_list[0]]
    return top5_tokens


if __name__ == '__main__':
    top5_tokens = predict("我们团队")
    print(top5_tokens)
