# -*- coding: utf-8 -*-
from config import filepath
from kernel.bdocr import wordrecognition as wr
from test import test_util
from util import doubanutil


def test_bdocr(version):
    # 测试接口详细步骤

    sum = 0
    right = 0
    in_dict = test_util.read_image_names_to_dict()
    for pic_name, real_word in in_dict.items():
        doubanutil.logger.info("real_word is: " + real_word)
        pic_path = filepath.image_path + pic_name
        recog_word = wr.get_word_in_pic(pic_path, version)
        doubanutil.logger.info("recog_word is: " + recog_word)
        if real_word == recog_word:
            right += 1
        sum += 1
        doubanutil.logger.info("right = " + str(right) + ";sum = " + str(sum))
        if sum == 100:
            break
    doubanutil.logger.info(str(version) + " accuracy: " + str(right / sum))


def test_bdocr_v1():
    # 测试百度OCR接口返回的验证码正确率

    test_bdocr("v1")


def test_bdocr_v2():
    # 测试百度OCR接口+单词匹配处理后返回的验证码正确率

    test_bdocr("v2")


if __name__ == "__main__":
    test_bdocr_v1()
    test_bdocr_v2()
