"""
模型验证
"""
import torch

from config import MODELS_DIR
from input_method_rnn.src.tokenizer import JiebaTokenizer
from model import InputMethodModel
from dataset import get_dataloader
from predict import predict_batch


def run_evaluate():
    # 1. 定义验证设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 加载验证集
    test_dataloader = get_dataloader(train=False)

    # 3. 加载词表
    tokenize = JiebaTokenizer.from_vocab(MODELS_DIR / "vocab.txt")

    # 4. 创建模型
    model = InputMethodModel(vocab_size=tokenize.vocab_size).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "best_model.pth"))

    top1_acc, top5_acc = evaluate(model, test_dataloader, device)
    print(f"评估结果：前5命中：{top5_acc:.2f}，前1命中：{top1_acc:.2f}")


def evaluate(model, test_dataloader, device):
    top1_acc_count = 0
    top5_acc_count = 0
    total_count = 0

    for inputs, targets in test_dataloader:
        inputs = inputs.to(device)
        targets = targets.tolist()

        top5_indexes_list = predict_batch(model, inputs)

        for target, top5_indexes in zip(targets, top5_indexes_list):
            total_count += 1
            if target in top5_indexes:  # 前5命中
                top5_acc_count += 1
                if target == top5_indexes[0]:  # 前1命中
                    top1_acc_count += 1
    return top1_acc_count / total_count, top5_acc_count / total_count

if __name__ == '__main__':
    run_evaluate()
