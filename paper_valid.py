# -*- coding: utf-8 -*-
import re

import jieba
import numpy as np


def get_word_vector():
    """
    w = np.ones((3,4))
    q = np.ones((3,4"))
    print(w)
    print(np.sum(w * q))
   """

    patten = r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+"
    s1 = input("句子1：").strip()
    s2 = input("句子2：").strip()
    s1 = re.sub(patten, " ", s1)
    s2 = re.sub(patten, " ", s2)
    cut1 = jieba.cut(s1)
    cut2 = jieba.cut(s2)

    list_word1 = (' '.join(cut1)).split()
    list_word2 = (' '.join(cut2)).split()
    print(list_word1)
    print(list_word2)

    key_word = list(set(list_word1 + list_word2))  # 取并集
    print(key_word)

    word_vector1 = np.zeros(len(key_word))  # 给定形状和类型的用0填充的矩阵存储向量
    word_vector2 = np.zeros(len(key_word))

    for i in range(len(key_word)):  # 依次确定向量的每个位置的值
        for j in range(len(list_word1)):  # 遍历key_word中每个词在句子中的出现次数
            if key_word[i] == list_word1[j]:
                word_vector1[i] += 1
        for k in range(len(list_word2)):
            if key_word[i] == list_word2[k]:
                word_vector2[i] += 1

    print(word_vector1)  # 输出向量
    print(word_vector2)
    return word_vector1, word_vector2


def cosine():
    while True:
        v1, v2 = get_word_vector()
        percent = float(np.sum(v1 * v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        print('相似度%.4f' % percent)


if __name__ == '__main__':
    cosine()
