#! usr/bin/python
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import time
import re
from database import DbClient
import threading


def get_html_text(url, encoding="utf-8"):
    """
    
    :param url: 要爬取的url
    :param code: 编码
    :return: str,html的字符串表示
    """
    try:
        r = requests.get(url)
        r.raise_for_status()
        r.encoding = encoding
        return r.text
    except:
        return ""


def get_json_data(html, foundationCode):
    """
    
    :param html:要解析的html
    :return: json串
    """
    json_dict = {}
    json_dict['id'] = foundationCode
    # TODO add name
    # client = DbClient()
    # json_dict['name'] = client.
    json_dict['data'] = parse_html(html)
    return json_dict
    # return json.dumps(json_dict)


def parse_html(html):
    """
    
    :param html: 要解析的html
    :return: list
    """
    data_list = []
    try:
        soup = BeautifulSoup(html, 'html.parser')
        for idx, tr in enumerate(soup.find_all('tr')):
            if idx != 0:
                tds = tr.find_all('td')
                data_list.append({
                    'period': tds[0].contents[0],
                    'accumulatedNet': tds[2].contents[0]
                })
    except:
        print('An exception occured, ignored and continue.')
    return data_list


def get_foundation_by_code(code, size=40, interval=3):
    for page in range(1, 20):
        ymd = time.strftime("%Y-%m-%d", time.localtime()).split('-')
        startDate = '%s-%s-%s' % (int(ymd[0]) - interval, ymd[1], ymd[2])
        url = r'http://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code=%s&page=%s&per=%s&sdate=%s' % (
            code, page, size, startDate)
        # startDate = time.strftime("%Y-%m-%d",time.localtime()[0]-3,time.localtime()[1],time.localtime()[2])
        response = get_html_text(url, encoding="utf-8")
        json_data = get_json_data(response, code)

        # json = parse_html(response)
        # print(json_data)
        client = DbClient()
        client.insert(json_data)
    print('saved %s to database.' % (code))


def get_foundation_by_code_online(code, size=100, interval=3):
    # for page in range(1, 20):
    ymd = time.strftime("%Y-%m-%d", time.localtime()).split('-')
    startDate = '%s-%s-%s' % (int(ymd[0]) - interval, ymd[1], ymd[2])
    url = r'http://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code=%s&page=1&per=%s&sdate=%s' % (
        code, size, startDate)
    # startDate = time.strftime("%Y-%m-%d",time.localtime()[0]-3,time.localtime()[1],time.localtime()[2])
    response = get_html_text(url, encoding="utf-8")
    json_data = get_json_data(response, code)

        # json = parse_html(response)
        # print(json_data)
        # client = DbClient()
        # client.insert(json_data)
    return json_data

def get_all_his_data(codes):
    for code in codes:
        get_foundation_by_code(code)


def get_all_foundation_name():
    for pageIdex in range(1, 80):
        list = []
        url = r'http://fund.eastmoney.com/data/rankhandler.aspx?op=ph&dt=kf&ft=all&rs=&gs=0&sc=2nzf&st=desc&sd=2016-07-20&ed=2017-07-20&qdii=&tabSubtype=,,,,,&pi=%s&pn=50&dx=1' % (
            pageIdex)
        response = get_html_text(url, encoding="utf-8")
        regex = r'[0-9]{6},.+?,'
        match = re.findall(regex, response)
        all_name_dict = dict(map(lambda s: s[:-1].split(','), match))
        client = DbClient()
        for k, v in all_name_dict.items():
            list.append({'code': k, 'name': v})
            print('appended to list...')
        try:
            client.insert_name(list)
        # print list
        # if match:
        #     print(match.group(0),match.group(1))
        except:
            return None


# get_all_foundation_name()
# client = DbClient()
# code = client.find_all_code()
# length = len(code)
# q1, q2, q3 = int(length / 4), 2 * int(length / 4), 3 * int(length / 4)
# part1, part2, part3, part4 = code[:q1], code[q1:q2], code[q2:q3], code[q3:]
# for i in range(4):
#     t1 = threading.Thread(target=get_all_his_data, args=(part1,), name='CrawlThread1'+str(i))
#     t1.start()
#     t1.join()

#get_foundation_by_code_online('001371')
