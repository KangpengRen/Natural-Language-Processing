"""
处理原始数据
"""
import jieba
import pandas as pd
from sympy.abc import J

import tokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import ROW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, SEQ_LEN
from input_method_rnn.src.tokenizer import JiebaTokenizer


def process():
    # 1. 读取文件
    df = (
        pd.read_json(ROW_DATA_DIR / "synthesized_.jsonl", lines=True, orient="records")
        # .sample(frac=0.1)  # 抽样10000条，加快测试速度
    )

    # 2. 提取句子
    sentences = []
    for dialog in df["dialog"]:
        for sentence in dialog:
            sentences.append(sentence.split("：")[1])
    print(f"提取句子总数：{len(sentences)}")

    # 3. 划分训练数据和测试数据
    print("划分训练集和测试集...")
    train_sentences, test_sentences = train_test_split(sentences, train_size=0.8, shuffle=True)

    # 4. 构建词表
    JiebaTokenizer.build_vocab(train_sentences, MODELS_DIR / "vocab.txt")
    tokenizer = JiebaTokenizer.from_vocab(MODELS_DIR / "vocab.txt")

    # 5. 构建并保存训练集和测试集
    train_dataset = build_dataset(tokenizer, train_sentences, train=True)
    test_dataset = build_dataset(tokenizer, test_sentences, train=False)
    pd.DataFrame(train_dataset).to_json(PROCESSED_DATA_DIR / "train.jsonl", orient="records", lines=True)
    pd.DataFrame(test_dataset).to_json(PROCESSED_DATA_DIR / "test.jsonl", orient="records", lines=True)

    print("数据处理完成！")


def build_dataset(tokenizer, sentences, train=False) -> list:
    indexes_sentences = [tokenizer.encode(sentence) for sentence in sentences]
    dataset = []
    for indexes_sentence in tqdm(indexes_sentences, desc=("构建训练集" if train else "构建测试集")):
        for i in range(len(indexes_sentence) - SEQ_LEN):
            dataset.append({"input": indexes_sentence[i:i + SEQ_LEN],
                                  "target": indexes_sentence[i + SEQ_LEN]})
    return dataset


if __name__ == '__main__':
    process()
