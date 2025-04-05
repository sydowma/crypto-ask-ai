# import scrapy
# from scrapy.spiders import CrawlSpider, Rule
# from scrapy.linkextractors import LinkExtractor

# class AllLinksSpider(CrawlSpider):
#     name = 'all_links_spider'  
#     allowed_domains = ['okx.com', 'www.okx.com'] 
#     start_urls = ['https://okx.com'] 


#     # only allow zh-hans and en, en will be default
#     deny_paths = (
#         r'/cn/', r'/hk/', r'/id/', r'/vi/', r'/fr/', r'/ru/', r'/pt-br/', 
#         r'/en-br/', r'/es-la/', r'/pt-pt/', r'/it/', r'/de/', r'/ro/', 
#         r'/pl/', r'/es-es/', r'/ua/', r'/cs/', r'/nl/', r'/ar/', 
#         r'/zh-hant/', r'/en-sg/', r'/zh-hans-sg/', 
#         r'/en-au/', r'/zh-hans-au/', r'/en-us/',
#         r'/fr-fe/', r'/sv/', r'/fi/', r'/ua-eu/', r'/ru-eu/',
#         r'/en-ae/', r'/ar-ae/', r'/ru-ae/', r'/fr-ae/', r'/zh-hans-ae/', r'/en-eu/',
#         r'/explorer/', r'explorer/', r'/zh-hans/explorer/', r'convert/', r'/zh-hans/convert/',
#     )
#     rules = (
#         Rule(LinkExtractor(allow=(), deny=deny_paths, allow_domains=allowed_domains), callback='parse_item', follow=True),
#     )

#     def parse_item(self, response):
#         # response.url 是当前页面的链接
#         yield {
#             'url': response.url,
#             'title': response.css('title::text').get(),  # 提取标题
#             'content': response.css('body::text').getall()  # 提取正文
#         }
