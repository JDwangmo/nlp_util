# -*- coding: utf-8 -*-
import scrapy
from scrapy.http import Request

from data_processing_util.crawler.crawler.configures import RESULT_DATA_PATH
import os
import io
from data_processing_util.crawler.crawler.items import CrawlerItem


class XinhuaDictionarySpider(scrapy.Spider):
    """
    新华字典的爬虫

    """
    name = "XinhuaDictionarySpider"
    root_url = 'http://xh.5156edu.com'
    allowed_domains = ["5156edu.com"]
    start_urls = [
        "http://xh.5156edu.com/index.php",
    ]

    def __init__(self, words, name=None, **kwargs):
        super(XinhuaDictionarySpider, self).__init__(name, **kwargs)
        self.query_words = words

    def start_requests(self):
        for url in self.start_urls:
            # encode_word = quote(self.query_words.encode('gb18030'))  # 编码后的单词
            encode_word = self.query_words.encode('gb2312')  # 编码后的单词
            yield Request(
                url=url,
                method='POST',
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36',
                    # noqa
                    'Referer': 'http://xh.5156edu.com',
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body='f_key={0}&f_type=zi&SearchString.x=0&SearchString.y=0'.format(encode_word),
                meta={'title': self.query_words},
                callback=self.parse,
                dont_filter=True
            )

    def parse(self, response):
        # print(response, type(response))

        # current_url = response.url  # 爬取时请求的url
        # body = response.body  # 返回的html
        # unicode_body = response.body_as_unicode()  # 返回的html unicode编码
        # print(response.url)
        # 基本解释 的 xpath

        # print(response.selector.xpath('//td[contains(@class,"font_18")]').re('.*?<br>.*?<br>(.*?)<br>.*')[0])
        # 更多相关 链接
        extract_url = response.selector.xpath('//a[contains(@href,"ciyu")]/@href').extract()

        if extract_url is not None and not extract_url[0].__contains__('end'):
            for url in extract_url:
                if response.meta['depth'] > 1 or url.__contains__('end'):
                    pass
                else:
                    # self.query_words = self.query_words + '_related'
                    url = self.root_url + url
                    print(url)
                    yield self.make_requests_from_url(url)

        if response.meta['depth'] == 2:
            extract_releated_words = response.xpath('//td[@width="25%"]').xpath('.//a/text()').extract()
            yield CrawlerItem(query_word=self.query_words, related_words=extract_releated_words)
            # filename = os.path.join(RESULT_DATA_PATH, self.query_words + '.html')
            # with io.open(filename, 'wb') as f:
            #     f.write(response.body)
