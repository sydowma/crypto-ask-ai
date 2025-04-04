import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

class AllLinksSpider(CrawlSpider):
    name = 'all_links_spider'  
    allowed_domains = ['okx.com', 'www.okx.com'] 
    start_urls = ['https://okx.com'] 

    # TODO exclude some region, like fr, ar, etc.
    rules = (
        # 提取页面中的所有链接，并递归访问
        Rule(LinkExtractor(allow=()), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        # response.url 是当前页面的链接
        yield {
            'url': response.url,
            'title': response.css('title::text').get(),  # 提取标题
            'content': response.css('body::text').getall()  # 提取正文
        }
