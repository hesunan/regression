# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 19:30:53 2018

@author: 95304
"""


import urllib.request
'''
# 向指定的url发送请求，并返回服务器响应的类文件对象
response = urllib.request.urlopen("https://hao.360.cn/")
 
# 类文件对象支持 文件对象的操作方法，如read()方法读取文件全部内容，返回字符串
html = response.read()
 
# 打印字符串
print (html)
'''

def getUrl_multiTry(url):
    user_agent ='"Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.122 Safari/537.36"'
    headers = { 'User-Agent' : user_agent }
    maxTryNum=10
    for tries in range(maxTryNum):
        try:
            req = urllib2.Request(url, headers = headers) 
            html=urllib2.urlopen(req).read()
            break
        except:
            if tries <(maxTryNum-1):
                continue
            else:
                logging.error("Has tried %d times to access url %s, all failed!",maxTryNum,url)
                break
            
                
    return html
