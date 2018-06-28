# -*- coding: utf-8 -*-
from config import filepath

image_word_dict = dict()


def read_image_names_to_dict():
    # 将"image_word.txt" 文件中的图片名称及对应的单词读到dict中

    with open(filepath.iw_txt, "r") as f_in:
        for line in f_in.readlines():
            image = line.split(",")[0].strip()
            word = line.split(",")[1].strip()
            image_word_dict[image] = word
    return image_word_dict


def get_word_in_image(image_name, iw_dict):
    # 输入图片名称，返回图片上的单词
    if image_name in iw_dict:
        return iw_dict[str(image_name).strip()]
    else:
        return "404"
