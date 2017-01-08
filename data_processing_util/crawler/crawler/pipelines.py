# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
import io


class CrawlerPipeline(object):
    def process_item(self, item, spider):
        # print('process_item')
        with io.open('/home/jdwang/PycharmProjects/nlp_util/data_processing_util/crawler/related_words/%s.txt' % item[
            'query_word'], 'w', encoding='utf8') as fout:
            fout.write(u'\n'.join(item['related_words']))
        # print(item)
        return item
