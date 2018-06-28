# -*- coding: utf-8 -*-
import os

from util import logmodule
from config import filepath

logger = logmodule.LogModule()
captcha_words_set = set()
image_word_dict = dict()
IS_CAPTCHA_WORDS_SET_CACHED = False


def cache_captcha_words():
    # 缓存豆瓣验证码单词集

    if os.path.exists(filepath.awic_txt):
        with open(filepath.awic_txt, "r") as words:
            for word in words.readlines():
                if word is not "":
                    captcha_words_set.add(str(word).strip())
        global IS_CAPTCHA_WORDS_SET_CACHED
        IS_CAPTCHA_WORDS_SET_CACHED = True
        logger.info("captcha_words_set is now cached")
    else:
        logger.warning(filepath.awic_txt + " not exists")


def get_length_longest_captcha_word():
    # 获取验证码包含的单词中最长单词的长度

    if not IS_CAPTCHA_WORDS_SET_CACHED:
        logger.info("captcha_words_set is not cached")
        cache_captcha_words()
    length = 0
    for word in captcha_words_set:
        temp = len(word)
        if temp > length:
            length = temp
    return length


def read_image_names_to_dict():
    # 将"image_word.txt" 文件中的图片名称及对应的单词读到dict中

    if os.path.exists(filepath.iw_txt):
        with open(filepath.iw_txt, "r") as f_in:
            for line in f_in.readlines():
                image = line.split(",")[0].strip()
                word = line.split(",")[1].strip()
                image_word_dict[image] = word
        logger.info("image_word_dict is now cached")
    else:
        logger.warning(filepath.iw_txt + " not exists")


def get_word_in_image(image_name, iw_dict):
    # 输入图片名称，返回图片上的单词

    if image_name in iw_dict:
        return iw_dict[str(image_name).strip()]
    else:
        return "404"


def init_func():
    # 载入缓存的“豆瓣验证码单词集”
    # 心得：有一些操作是需要在初始化时一并完成的，但是如果不能保证初始化一定被执行的情况下，
    # 需要加入判断（这是因为既然不确定是否完成了必要的缓存初始化，如果有多个地方都调用缓存操作，
    # 则有可能会有重复的操作）来避免重复操作。所以，此处设计很糟糕！类似情况下，代码在设计时应该：
    # 1. 保证初始化一定会被执行，并且成功执行！ 2. 如果不能保证，则所有地方都需要加判断（思考这样做的
    # 弊端？或请教别人给出更好的解决方案）
    if IS_CAPTCHA_WORDS_SET_CACHED is False:
        cache_captcha_words()

    # 载入缓存的“图片名称及对应的单词”字典
    read_image_names_to_dict()
