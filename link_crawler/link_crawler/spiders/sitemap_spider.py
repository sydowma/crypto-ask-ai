import scrapy
from scrapy.spiders import SitemapSpider
from scrapy.http import Request
from urllib.parse import urlparse

class SitemapLinksSpider(SitemapSpider):
    name = 'sitemap_links_spider'
    allowed_domains = ['okx.com', 'www.okx.com']
    
    sitemap_urls = [
        'https://www.okx.com/default-index.xml',
        'https://www.okx.com/learn-index.xml',
        # 'https://www.okx.com/convert-index.xml',
        'https://www.okx.com/help-center-index.xml',
        'https://www.okx.com/landingpage-index.xml',
        # 'https://www.okx.com/markets-index.xml',
    ]

    # Define sitemap rules to link URLs to the parsing method
    sitemap_rules = [
        # Match URLs containing /zh-hans/ or /en/, or the exact root path
        (r'/(zh-hans|en)/.*', 'parse_item'),
        (r'^https://www.okx.com/?$', 'parse_item'), 
    ]
  
    deny_paths = (
        r'/cn/', r'/hk/', r'/id/', r'/vi/', r'/fr/', r'/ru/', r'/pt-br/', 
        r'/en-br/', r'/es-la/', r'/pt-pt/', r'/it/', r'/de/', r'/ro/', 
        r'/pl/', r'/es-es/', r'/ua/', r'/cs/', r'/nl/', r'/ar/', 
        r'/zh-hant/', r'/en-sg/', r'/zh-hans-sg/', 
        r'/en-au/', r'/zh-hans-au/', r'/en-us/',
        r'/fr-fe/', r'/sv/', r'/fi/', r'/ua-eu/', r'/ru-eu/',
        r'/en-ae/', r'/ar-ae/', r'/ru-ae/', r'/fr-ae/', r'/zh-hans-ae/', r'/en-eu/',
        r'/explorer/', r'explorer/', r'/zh-hans/explorer/', r'convert/', r'/zh-hans/convert/',
        r'how-to-buy/', r'price/',
        r'markets/'
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Removed url_list and seen_urls as results are yielded by parse_item
        
    def sitemap_filter(self, entries):
        # Filter entries based on deny_paths before rules are applied
        for entry in entries:
            url = entry['loc']
            # Skip if URL contains any denied paths
            if any(deny_path in url for deny_path in self.deny_paths):
                # print(f"Skipping denied URL: {url}") # Keep for debugging if needed
                continue
            yield entry # Pass allowed entries to sitemap_rules

    # The default parse method is used by SitemapSpider for sitemap URLs
    # We don't need custom logic here unless parsing the sitemap itself differently
    def parse(self, response):
       pass # Let the base class handle sitemap parsing

    def closed(self, reason):
        # Log the reason for closing
        self.logger.info(f"Spider closed: {reason}")
     
    def parse_item(self, response):
        self.logger.info(f"Parsing page: {response.url}")
        # response.url is the current page URL
        yield {
            'url': response.url,
            'title': response.css('title::text').get(),  # Extract title
            # Extract meaningful text content, avoiding scripts/styles
            'content': ' '.join(response.css('body *:not(script):not(style)::text').getall()).strip() 
        }