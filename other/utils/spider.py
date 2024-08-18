# pip install scrapy html2text bs4

import scrapy
from scrapy.crawler import CrawlerProcess
from bs4 import BeautifulSoup
import html2text
import os
import json
from urllib.parse import urlparse


# 功能：爬取网页并将内容转换为 Markdown 格式
# 自定义的Scrapy爬虫类ContentFocusedSpider，继承自Scrapy的Spider基类
class ContentFocusedSpider(scrapy.Spider):
    # name属性定义了这个爬虫的名称，用于标识爬虫
    name = 'content_focused_spider'
    # start_urls是爬虫开始抓取的初始URL列表。这里设置了一个初始URL
    start_urls = ['https://crawl4ai.com/mkdocs/']
    # allowed_domains指定了爬虫可以抓取的域名。此设置可以防止爬虫越界抓取其他域名的网站内容
    allowed_domains = ['crawl4ai.com']

    # 初始化方法，用于设置爬虫在启动时的各种配置
    def __init__(self, *args, **kwargs):
        super(ContentFocusedSpider, self).__init__(*args, **kwargs)
        # 创建了一个HTML2Text对象，html2text是一个将HTML转换为Markdown的库
        self.h = html2text.HTML2Text()
        # 设置HTML2Text的选项，如忽略链接、忽略图像、忽略强调（如斜体字）等
        self.h.ignore_links = True
        self.h.ignore_images = True
        self.h.ignore_emphasis = True
        self.h.body_width = 0
        # 初始化一个空列表self.results，用于存储每个页面的爬取结果）
        self.results = []
        # 创建保存数据的目录，如果目录不存在则创建
        os.makedirs('.data', exist_ok=True)
        os.makedirs('.data/markdown_files', exist_ok=True)

    # Scrapy中的一个默认解析方法，用于处理每个响应（网页内容）
    def parse(self, response):
        # 使用BeautifulSoup解析网页内容为一个soup对象，以便于进一步的HTML处理
        soup = BeautifulSoup(response.text, 'html.parser')
        # 移除导航栏、侧边栏、页脚等元素，专注于页面的主要内容
        for elem in soup(['nav', 'header', 'footer', 'aside']):
            elem.decompose()

        # 尝试找到主要内容区域，优先查找 <main>、 <article> 或带有content类的 <div> 标签
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')

        if main_content:
            content = str(main_content)
        else:
            content = str(soup.body)  # 如果找不到明确的主要内容，使用整个 body

        # 使用HTML2Text将提取的HTML内容转换为Markdown格式
        markdown_content = self.h.handle(content)

        # 保存生成文件名，用于保存Markdown文件。将URL的路径部分转换为文件名，若路径为空则使用'index'
        parsed_url = urlparse(response.url)
        file_path = parsed_url.path.strip('/').replace('/', '_') or 'index'
        # 生成完整的Markdown文件路径
        markdown_filename = f'.data/markdown_files/{file_path}.md'

        # 打开文件并写入转换后的Markdown内容
        with open(markdown_filename, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        # 创建一个字典对象result，包含当前页面的URL和保存的Markdown文件路径
        result = {
            'url': response.url,
            'markdown_file': markdown_filename,
        }
        # 将当前页面的结果添加到self.results列表中
        self.results.append(result)

        # 遍历页面中所有的链接并继续爬取这些链接指向的页面
        # 对每个找到的链接，继续递归爬取，调用self.parse方法处理新的响应
        for link in response.css('a::attr(href)').getall():
            yield response.follow(link, self.parse)

    # closed方法在爬虫结束时调用，用于执行清理或保存最终结果的操作
    def closed(self, reason):
        # 将所有页面的爬取结果（URL和文件路径）保存到一个JSON文件中
        with open('.data/markdown_results.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        print(f"爬取完成。总共爬取了 {len(self.results)} 个页面")
        print("结果元数据保存在 .data/markdown_results.json")
        print("Markdown 文件保存在 .data/markdown_files/ 目录下")



# CrawlerProcess 是 Scrapy 提供的一个类，用于在脚本中创建并启动爬虫。settings 参数接受一个字典，用于配置爬虫的各种设置
# USER_AGENT，这个参数指定了爬虫在向服务器发送请求时使用的浏览器标识（即用户代理）。这里的值模仿了一个真实的浏览器，以避免被服务器拒绝访问
# ROBOTSTXT_OBEY,是否遵守 robots.txt 文件中的爬取规则。True 表示爬虫会遵守这些规则，以避免访问网站管理员不希望爬取的部分
# CONCURRENT_REQUESTS，定义了同时进行的请求数。在这里被设置为 1，表示一次只发送一个请求，这对于避免服务器过载以及防止被封禁非常重要
# DOWNLOAD_DELAY，设置了每个请求之间的延迟时间（以秒为单位）。在此处设置为 2 秒，用于减缓爬取速度，避免给目标服务器造成过大的压力
process = CrawlerProcess(settings={
    'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'ROBOTSTXT_OBEY': True,
    'CONCURRENT_REQUESTS': 1,
    'DOWNLOAD_DELAY': 2,
})


# process.crawl 方法启动指定的爬虫类
# ContentFocusedSpider是定义的一个爬虫类。它包含了爬取逻辑，包括如何处理响应以及从页面提取数据
process.crawl(ContentFocusedSpider)
# 启动并开始执行爬取操作。这个方法会阻塞当前脚本，直到所有爬虫任务完成
process.start()