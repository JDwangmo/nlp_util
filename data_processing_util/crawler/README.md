## crawler 数据爬虫工具
### Describe
- 参考： [教你分分钟学会用python爬虫框架Scrapy爬取心目中的女神](http://www.cnblogs.com/wanghzh/p/5824181.html):
    - [备用地址](https://app.yinxiang.com/shard/s8/nl/1386165/1ccc5940-9302-48dc-b9c8-d13e999186a4)
- 本项目更多的只是作为一个例子,以爬取 [汉语字典](http://xh.5156edu.com/ciyu/83130begin38794.html) 上某个词的相关词为例子;


### Project Structure
- 这里的结构是直接使用 scrapy 提供的命令创建的默认架构：
    - 进入项目目录，在终端运行 `scrapy startproject crawler` 即可自动创建项目
    - `cd crawler`
    - 创建一个蜘蛛： `scrapy genspider example example.com`
    - 最终结果架构如下：
        - scrapy.cfg  项目的配置信息，主要为Scrapy命令行工具提供一个基础的配置信息。（真正爬虫相关的配置信息在settings.py文件中）
        - items.py    设置数据存储模板，用于结构化数据，如：Django的Model
        - pipelines    数据处理行为，如：一般结构化的数据持久化
        - settings.py 配置文件，如：递归的层数、并发数，延迟下载等
        - spiders      爬虫目录，如：创建文件，编写爬虫规则

### Dependence lib
- Scrapy==1.3.0

### User Manual
- 1 可以直接在 终端中输入命令：
    - `cd crawler`
    - `scrapy crawl xiaohuar --nolog` 即可运行，格式：scrapy crawl+爬虫名(name属性)  --nolog即不显示日志
- 2 运行 run.py ，通过脚本运行爬虫
- 3 如果需要修改爬虫的速度，修改"settings.py"里的下载延迟和最大并发数即可
    - 该爬虫没有提供代理池中间件，为了防止网站ban IP地址，下载延迟应该尽量调大一点，最大并发数尽量调小一点。
