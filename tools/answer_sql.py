import pymysql
import logging
import pandas as pd
from .config import *
import time


class AnswerSQL(object):

    logger = logging.getLogger(__name__)

    def __init__(self):
        self.conn = None
        try:
            self.conn = pymysql.connect(
                host=QA_DB_HOST, user=QA_DB_USER, password=QA_DB_PASSWORD, db='qa', charset='utf8')
        except pymysql.err.OperationalError as e:
            print('Error is '+str(e))
            exit()
        self.answer_dict = None
        self.current_max_id = None
        self.update_time = None

    def get_qa(self, forced=False):
        if self.answer_dict is None:
            self.answer_dict = dict()
            self.update_time = time.time()+60*10
            sql = 'select * from answer'
            answer = pd.read_sql(sql, con=self.conn)
            for i in answer.index:
                recorder = answer.iloc[i]
                self.answer_dict[recorder["id"]] = recorder["text"]
                if self.current_max_id is None:
                    self.current_max_id = recorder["id"]
                self.current_max_id = max(recorder["id"], self.current_max_id)
        elif forced is True or self.update_time < time.time():
            self.update_time = self.update_time + 60 * 10
            sql = 'select * from answer where id >= %s' % self.current_max_id
            answer = pd.read_sql(sql, con=self.conn)
            for i in answer.index:
                recorder = answer.iloc[i]
                self.answer_dict[recorder["id"]] = recorder["text"]
                if self.current_max_id is None:
                    self.current_max_id = recorder["id"]
                self.current_max_id = max(recorder["id"], self.current_max_id)
        return self.answer_dict

