import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor

class AllLinksSpider(CrawlSpider):
    name = 'all_links_spider'  # 爬虫名称
    allowed_domains = ['example.com']  # 限制爬取的域名，防止爬到站外
    start_urls = ['https://www.example.com']  # 起始 URL

    # 定义爬取规则
    rules = (
        # 提取页面中的所有链接，并递归访问
        Rule(LinkExtractor(allow=()), callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        # response.url 是当前页面的链接
        yield {
            'url': response.url
        }

if __name__ == "__main__":
    process = CrawlerProcess()
    process.crawl(AllLinksSpider)
    process.start()
