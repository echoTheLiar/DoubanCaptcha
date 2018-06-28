# -*- coding: utf-8 -*-
from util import doubanutil


def get_longest_word(words):
    # 从单词列表中返回最长的单词（由于百度OCR识别接口可能会把豆瓣验证码识别为多个单词）

    return_word = ""
    if len(words):
        for word in words.split(" "):
            if len(word) > len(return_word):
                return_word = word
        doubanutil.logger.info("the longest word returned is: " + return_word)
    return return_word


def match_closest_captcha_word(word):
    # 在验证码字典里寻找与word最匹配的单词（利用编辑距离计算相似程度）

    if len(doubanutil.captcha_words_set) is 0:
        doubanutil.cache_captcha_words()
    min_dis = 99
    close_word = ""
    for cap_word in doubanutil.captcha_words_set:
        dis = get_edit_distance(word, cap_word)
        if dis < min_dis:
            close_word = cap_word
            min_dis = dis
    doubanutil.logger.info("the closest captcha word matched is: " + close_word)
    return close_word


def get_edit_distance(word1, word2):
    # 计算两个词语的“编辑距离”

    len1, len2 = len(word1), len(word2)
    dis = [[0 for _ in range(0, len2 + 1)] for _ in range(0, len1 + 1)]
    for i in range(0, len1 + 1):
        dis[i][0] = i
    for j in range(1, len2 + 1):
        dis[0][j] = j
    for i in range(0, len1):
        for j in range(0, len2):
            # 不区分大小写
            if word1[i] == word2[j] or chr(ord(word1[i])+32) == word2[j] or chr(ord(word1[i])-32) == word2[j]:
                dis[i + 1][j + 1] = dis[i][j]
            else:
                replace = dis[i][j]
                insert = dis[i + 1][j]
                delete = dis[i][j + 1]
                dis[i + 1][j + 1] = ((replace if replace < insert else insert)
                                     if (replace if replace < insert else insert) < delete else delete)
                dis[i + 1][j + 1] += 1
    return dis[len1][len2]


if __name__ == "__main__":
    print(get_edit_distance("vEssel", "verse"))
