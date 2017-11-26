#! usr/bin/python
# -*- coding: utf-8 -*-

import pymongo
from pymongo import MongoClient

from configs import configs

# client = MongoClient()
class DbClient:

    client = MongoClient(configs['db']['host'], configs['db']['port'])
    db = client['buaa_web_crawler']

    def insert(self,obj):

        DbClient.db.his_data.insert_one(obj)
        # print('inserted:\n'+insert)

    def insert_name(self,obj):

        DbClient.db.fundation_name.insert_many(obj)

    def find_by_code(self,code):
        return DbClient.db.his_data.find_one({'id':code})

    def find_name_by_code(self,code):
        return DbClient.db.fundation_name.find_one({'code':code})

    def find_all_code(self):
        r = DbClient.db.fundation_name.find()
        l = [e['code'] for e in r]
        return l
