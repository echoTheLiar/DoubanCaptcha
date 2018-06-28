# -*- coding: utf-8 -*-
import urllib.request
import json

from config import baiconfig as conf


def get_access_token():
    # 获取百度AI开放平台的access_token

    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials' \
           '&client_id=' + conf.API_KEY + '&client_secret=' + conf.SECRET_KEY
    request = urllib.request.Request(host)
    request.add_header('Content-Type', 'application/json; charset=UTF-8')
    response = urllib.request.urlopen(request)
    content = response.read()
    if content:
        content = json.loads(content)
    return content
