# -*- coding: utf-8 -*-
import random
import numpy as np
from PIL import Image

from util import doubanutil

alpha = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
         'w', 'x', 'y', 'z']
MAX_LENGTH = doubanutil.get_length_longest_captcha_word()  # 验证码涉及的单词中最长单词的长度
ALPHA_LENGTH = len(alpha)
used_captcha_set = set()


def char2index(c):
    # 获取字符的位置索引（由ASCII码值进行计算）
    n = ord(c) - 97
    if 0 <= n <= 25:
        return n
    else:
        raise ValueError(c + "不在字母列表内")


def word2vec(word):
    # 将单词转换为向量形式

    word_len = len(word)
    if word_len > MAX_LENGTH:
        raise ValueError("超过最大长度")
    vec = np.zeros(MAX_LENGTH * ALPHA_LENGTH)
    for i, c in enumerate(word):
        index = i * ALPHA_LENGTH + char2index(c)
        vec[index] = 1
    return vec


def vec2word(vec):
    # 将向量转换成单词

    index_arr = vec.nonzero()[0]
    word = ""
    for i, n in enumerate(index_arr):
        char_pos = n % 26
        word += chr(97 + char_pos)
    return word


def get_captcha_word_and_image():
    # 从数据集中获取验证码的单词及对应图片

    image = ""
    while True:
        if len(used_captcha_set) == len(doubanutil.image_word_dict):
            break
        image = random.choice(list(doubanutil.image_word_dict))
        if image not in used_captcha_set:
            used_captcha_set.add(image)
            break
    word = doubanutil.get_word_in_image(image, doubanutil.image_word_dict)
    return image, word


def compute_gray(pic_path):
    # 返回图片的灰度值

    captcha_image = Image.open(pic_path)
    np_image = np.array(captcha_image)
    r, g, b = np_image[:, :, 0], np_image[:, :, 1], np_image[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


if __name__ == "__main__":
    # print(vec2word(word2vec("say")))
    doubanutil.init_func()
    print(random.choice(list(doubanutil.image_word_dict)))
