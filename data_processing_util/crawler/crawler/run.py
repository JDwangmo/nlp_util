# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2017-01-06'; 'last updated date: 2017-01-06'
    Email:   '383287471@qq.com'
    Describe: 在 程序中 运行 爬虫
"""
from __future__ import print_function
from data_processing_util.crawler.crawler.spiders.xinhua_dictionary_spider import XinhuaDictionarySpider
from scrapy.crawler import CrawlerProcess

# 导入本项目中 setting.py 的配置
from scrapy.utils.project import get_project_settings

process = CrawlerProcess(get_project_settings())
# 可以传参数
process.crawl(XinhuaDictionarySpider, words=u'哀')
process.start()
