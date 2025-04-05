import scrapy
from scrapy.spiders import SitemapSpider
from scrapy.http import Request

class SitemapLinksSpider(SitemapSpider):
    name = 'sitemap_links_spider'
    allowed_domains = ['okx.com', 'www.okx.com']
    
    sitemap_urls = [
        'https://www.okx.com/default-index.xml',
        'https://www.okx.com/learn-index.xml',
        'https://www.okx.com/convert-index.xml',
        'https://www.okx.com/help-center-index.xml',
        'https://www.okx.com/landingpage-index.xml',
        'https://www.okx.com/markets-index.xml',
    ]

    # # 定义 sitemap 的过滤规则
    # sitemap_rules = [
    #     # 只爬取 zh-hans 和 en 的页面，其他语言路径被排除
    #     (r'/(zh-hans/|en/|^https://www.okx.com/[^/]+)', 'parse_item'),
    # ]

    deny_paths = (
        r'/cn/', r'/hk/', r'/id/', r'/vi/', r'/fr/', r'/ru/', r'/pt-br/', 
        r'/en-br/', r'/es-la/', r'/pt-pt/', r'/it/', r'/de/', r'/ro/', 
        r'/pl/', r'/es-es/', r'/ua/', r'/cs/', r'/nl/', r'/ar/', 
        r'/zh-hant/', r'/en-sg/', r'/zh-hans-sg/', 
        r'/en-au/', r'/zh-hans-au/', r'/en-us/',
        r'/fr-fe/', r'/sv/', r'/fi/', r'/ua-eu/', r'/ru-eu/',
        r'/en-ae/', r'/ar-ae/', r'/ru-ae/', r'/fr-ae/', r'/zh-hans-ae/', r'/en-eu/',
        r'/explorer/', r'explorer/', r'/zh-hans/explorer/', r'convert/', r'/zh-hans/convert/',
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.url_list = []
        
        
    def sitemap_filter(self, entries):
        for entry in entries:
            url = entry['loc']
            if not any(deny_path in url for deny_path in self.deny_paths):
                yield entry

    def parse(self, response):
        # 解析 sitemap 文件，提取 <loc> 中的 URL
        pass

    def closed(self, reason):
        # 爬虫结束时打印总 URL 数量（可选）
        self.logger.info(f"Total URLs extracted: {len(self.url_list)}")
 