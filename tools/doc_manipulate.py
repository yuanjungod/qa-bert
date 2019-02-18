import numpy as np
import pandas as pd

from elasticsearch_dsl import Index, connections, Document, Nested, InnerDoc, Text, Integer


class QAManipulate:
    _PLACEHOLDER = object()

    def __init__(self, index_name, **settings):
        self.index_name = index_name
        self._index = Index(index_name)
        self._index.settings(**settings)
        self.connect()

        class Inner(QADuos):
            class Index:
                name = index_name
        self._inner_cls = Inner

    def create_index(self):
        self._index.create()

    def delete_index(self):
        self._index.delete()

    def insert(self, doc_id, a_id=None, a_content=None, q_list=None):
        data = self._inner_cls()
        found = data.get(id=doc_id, ignore=404)
        if found is None:
            data.meta.id = doc_id
            data.a_id, data.a_content, data.q_list = a_id, a_content, self._question_parser(q_list)
            return data.save()
        else:
            msg = "ID '{}' of Index '{}' already exists."
            raise ValueError(msg.format(doc_id, self._index._name))

    def update(self, doc_id, a_id=_PLACEHOLDER, a_content=_PLACEHOLDER, q_list=_PLACEHOLDER):
        data = self._inner_cls()
        found = data.get(id=doc_id, ignore=404)
        if found is not None:
            param_dict = {}
            if a_id is not QAManipulate._PLACEHOLDER:
                param_dict.update({"a_id": a_id})
            if a_content is not QAManipulate._PLACEHOLDER:
                param_dict.update({"a_content": a_content})
            if q_list is not QAManipulate._PLACEHOLDER:
                param_dict.update({"q_list": self._question_parser(q_list)})
            return found.update(**param_dict)
        else:
            msg = "ID '{}' of Index '{}' does not exist."
            raise ValueError(msg.format(doc_id, self._index._name))

    def delete(self, doc_id):
        data = self._inner_cls()
        found = data.get(id=doc_id, ignore=404)
        if found is not None:
            return found.delete()
        else:
            msg = "ID '{}' of Index '{}' does not exist."
            raise ValueError(msg.format(doc_id, self._index._name))

    def query(self, query, question_only=True, max_rec_cnt=5, boost_question=1, **kwargs):
        if question_only:
            result = self._search_by_question(query, max_rec_cnt=max_rec_cnt, **kwargs)
        else:
            result = self._search_by_qa(query, boost_question, max_rec_cnt=max_rec_cnt, **kwargs)

        df = pd.DataFrame(self._parse_result(result))
        df_temp = df[['doc_id', 'score']].copy().drop_duplicates()
        df_temp['score_softmax'] = self._softmax(df_temp['score'])

        return df.merge(df_temp, on=['doc_id', 'score'], how='left')

    def _search_by_question(self, question, max_rec_cnt, **kwargs):
        s = self._inner_cls().search(**kwargs)

        result = s.query("match", q_list__q_content=question).execute(ignore_cache=True)
        return result[:max_rec_cnt]

    def _search_by_qa(self, qa_string, boost, max_rec_cnt, **kwargs):
        s = self._inner_cls().search(**kwargs)

        boosted_question = "q_list.q_content^{}".format(boost)

        result = s.query("multi_match", query=qa_string, fields=['a_content', boosted_question])\
                  .execute(ignore_cache=True)
        return result[:max_rec_cnt]

    def _raw_search(self, **kwargs):
        s = self._inner_cls().search(**kwargs)
        return s

    @staticmethod
    def connect(hosts=['http://root:cdslyk912@192.168.10.49:9200/'], timeout=80):
        connections.create_connection(hosts=hosts, timeout=timeout)

    @staticmethod
    def _parse_result(result):
        for hit in result:
            a_id, a_content, score, doc_id = hit.a_id, hit.a_content, hit.meta.score, hit.meta.id
            for question in hit.q_list:
                q_id = question.q_id
                q_content = question.q_content

                yield {'a_id': a_id, 'a_content': a_content, 'score': score,
                       'q_id': q_id, 'q_content': q_content, 'doc_id': doc_id}

    @staticmethod
    def _question_parser(answer_list):
        result = []
        for answer_data in answer_list:
            result.append(
                Questions(q_id=answer_data['q_id'], q_content=answer_data['q_content'])
            )
        return result

    @staticmethod
    def _softmax(score_list):
        score_array = np.sqrt(score_list)
        exp_array = np.exp(score_array)
        factor = score_list[0] / (1 + score_list[0])
        softmax_array = exp_array / exp_array.sum() * factor
        return softmax_array


class Questions(InnerDoc):
    q_id = Integer()
    q_content = Text(analyzer="ik_smart")


class QADuos(Document):
    a_id = Integer()
    a_content = Text(analyzer="ik_smart")

    q_list = Nested(Questions)

    def save(self, **kwargs):
        return super().save(**kwargs)


if __name__ == "__main__":
    pass








