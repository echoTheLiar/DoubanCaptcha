# -*- coding: utf-8 -*-
import base64
import json
import urllib

from kernel.bdocr import accesstoken, wordmatch as wm
from util import doubanutil


def get_word_in_pic(pic_path, version):
    # 给定图片地址 pic_path，识别图片当中的文字

    result = accesstoken.get_access_token()
    access_token = result["access_token"]
    url = 'https://aip.baidubce.com/rest/2.0/ocr/v1/webimage?access_token=' + access_token
    # 二进制方式打开图文件
    f = open(pic_path, 'rb')
    # 参数image：图像base64编码
    img = base64.b64encode(f.read())
    params = {"image": img}
    params = urllib.parse.urlencode(params).encode(encoding="utf-8")
    request = urllib.request.Request(url, params)
    request.add_header('Content-Type', 'application/x-www-form-urlencoded')
    response = urllib.request.urlopen(request)
    content = response.read()
    if content:
        doubanutil.logger.info("baidu OCR API returns: " + str(content))
        content = json.loads(content)
        words_result = content["words_result"]
        # 对百度OCR接口返回的识别结果作进一步处理，以提升识别准确率
        if len(words_result):
            words = str(words_result[0]["words"]).strip()
            if version == "v1":
                return wm.get_longest_word(words)
            if version == "v2":
                return wm.match_closest_captcha_word(wm.get_longest_word(words))
        else:
            return ""
    else:
        return ""
