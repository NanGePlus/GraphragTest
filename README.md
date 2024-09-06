# 0、相关视频教程       
(1)基础开发环境搭建                   
https://www.bilibili.com/video/BV1tQWje1ErT/?vd_source=30acb5331e4f5739ebbad50f7cc6b949                  
https://youtu.be/myVgyitFzrA             

(2)【GraphRAG+阿里通义千问大模型】构建+检索全流程实操，打造基于知识图谱的本地知识库，本地搜索、全局搜索二合一                                            
https://www.bilibili.com/video/BV1yzHxeZEG5/?vd_source=30acb5331e4f5739ebbad50f7cc6b949                               
https://youtu.be/w9CRDbafhPI                                      

(3)【GraphRAG+知识图谱可视化】知识图谱neo4j可视化呈现，构建近2万字文本知识图谱，打造基于知识图谱的本地知识库，本地搜索、全局搜索二合一                    
https://www.bilibili.com/video/BV1prHxe4E9J/?vd_source=30acb5331e4f5739ebbad50f7cc6b949                   
https://youtu.be/IqE-FP3YLj4                     

(4)【GraphRAG最新版本0.3.0对比实战评测】使用gpt-4o-mini和qwen-plus分别构建近2万字文本知识索引+本地/全局检索对比测试                    
https://www.bilibili.com/video/BV1maHxeYEB1/?vd_source=30acb5331e4f5739ebbad50f7cc6b949                    
https://youtu.be/iXfsJrXCEwA                      

(4)【GraphRAG+智谱AI大模型】构建+检索全流程实操，打造基于知识图谱的本地知识库，本地搜索、全局搜索二合一                                  
https://www.bilibili.com/video/BV11vHxehEiU/?vd_source=30acb5331e4f5739ebbad50f7cc6b949                   
https://youtu.be/0l4OYQ81GEo                

(5)【GraphRAG+讯飞星火大模型】构建+检索全流程实操，打造基于知识图谱的本地知识库，本地搜索、全局搜索二合一                                  
https://www.bilibili.com/video/BV17QHxeSEjy/?vd_source=30acb5331e4f5739ebbad50f7cc6b949                     
https://youtu.be/42oQAp2Gf1g                       


# 1、准备工作
## 1.1 graphrag基础概念
### （1）定义
GraphRAG是微软研究院开发的一种创新型检索增强生成（RAG）方法，旨在提高大语言模型LLM在处理复杂信息和私有数据集时的推理能力    

### (2)索引 indexing 
是一个数据管道和转换套件，旨在利用LLM从非结构化文本中提取有意义的结构化数据    
核心功能：    
**[1]** 从原始文本中提取实体、关系和声明    
**[2]** 在实体中执行社区检测    
**[3]** 生成多级粒度的社群摘要和报告     
**[4]** 将实体嵌入图向量空间     
**[5]** 将文本块嵌入文本向量空间     
**[6]** 管道的输出可以json和parquet等多种格式存储      
提供的实体类型：   
**[1]** Document：输入的文档，csv或txt中的单个行     
**[2]** TextUnit：要分析的文本块，Document与TextUnit关系为1:N关系         
**[3]** Entity：从TextUnit中提取的实体         
**[4]** Relationship：两个实体之间的关系，关系由Covariate（协变量）生成        
**[5]** Covariate：协变量，提取声明信息，包含实体的声明       
**[6]** CommunityReport：社群报告，实体生成后对其执行分层社群检测，并为该分层中的每个社群生成报告        
**[7]** Node：节点，实体和文档渲染图的布局信息     

### (3)提示微调 prompt tuning
为生成的知识图谱创建领域自适应prompt模版的功能，提供两种方式进行调整：      
**自动调整：** 通过加载输入，将输入分割成文本块，然后运行一系列LLM调用和prompt模版替换来生成最终的prompt模版     
**手动调整：** 手动调整prompt模版             
**具体用法如下：**      
python -m graphrag.prompt_tune --config ./settings.yaml --root ./ --no-entity-types --language Chinese --output ./prompts       
根据实际情况选择相关参数：           
**--config** :(必选) 所使用的配置文件，这里选择setting.yaml文件     
**--root** :(可选)数据项目根目录，包括配置文件（YML、JSON 或 .env）。默认为当前目录       
**--domain** :(可选)与输入数据相关的域，如 “空间科学”、“微生物学 ”或 “环境新闻”。如果留空，域将从输入数据中推断出来     
**--method** :(可选)选择文档的方法。选项包括全部(all)、随机(random)或顶部(top)。默认为随机       
**--limit** :(可选)使用随机或顶部选择时加载文本单位的限制。默认为 15     
**--language** :(可选)用于处理输入的语言。如果与输入语言不同，LLM 将进行翻译。默认值为“”，表示将从输入中自动检测     
**--max-tokens** :(可选)生成提示符的最大token数。默认值为 2000     
**--chunk-size** :(可选)从输入文档生成文本单元时使用的标记大小。默认值为 20     
**--no-entity-types**（无实体类型） :(可选)使用无类型实体提取生成。建议在数据涵盖大量主题或高度随机化时使用        
**--output** :(可选)保存生成的提示信息的文件夹。默认为 “prompts”    

### (4)检索 query
本地搜索（Local Search）、全局搜索（Global Search）、问题生成（Question Generation）       
**本地搜索（Local Search）：基于实体的推理**     
本地搜索方法将知识图谱中的结构化数据与输入文档中的非结构化数据结合起来，在查询时用相关实体信息增强 LLM 上下文    
这种方法非常适合回答需要了解输入文档中提到的特定实体的问题（例如，"洋甘菊有哪些治疗功效？）        
**全局搜索（Global Search）： 基于全数据集推理**       
根据LLM生成的知识图谱结构能知道整个数据集的结构（以及主题）     
这样就可以将私有数据集组织成有意义的语义集群，并预先加以总结。LLM在响应用户查询时会使用这些聚类来总结这些主题    
**问题生成（Question Generation）：基于实体的问题生成**        
将知识图谱中的结构化数据与输入文档中的非结构化数据相结合，生成与特定实体相关的候选问题       

## 1.2 oneapi安装和部署    
### （1）OneAPI是什么
官方介绍：是OpenAI接口的管理、分发系统       
支持 Azure、Anthropic Claude、Google PaLM 2 & Gemini、智谱 ChatGLM、百度文心一言、讯飞星火认知、阿里通义千问、360 智脑以及腾讯混元     
### (2)安装、部署  
使用官方提供的release软件包进行安装部署 ，详情参考如下链接中的手动部署：         
https://github.com/songquanpeng/one-api          
下载OneAPI可执行文件one-api并上传到服务器中然后，执行如下命令后台运行                
nohup ./one-api --port 3000 --log-dir ./logs > output.log 2>&1 &                 
运行成功后，浏览器打开如下地址进入one-api页面，默认账号密码为：root 123456               
http://IP:3000/           
### (3)创建渠道和令牌       
创建渠道：大模型类型(通义千问)、APIKey(通义千问申请的真实有效的APIKey)             
创建令牌：创建OneAPI的APIKey，后续代码中直接调用此APIKey                       

## 1.3 openai使用方案
国内无法直接访问，可以使用代理的方式，具体代理方案自己选择               
可以参考这期视频:                
【GraphRAG最新版本0.3.0对比实战评测】使用gpt-4o-mini和qwen-plus分别构建近2万字文本知识索引+本地/全局检索对比测试                                        
https://www.bilibili.com/video/BV1maHxeYEB1/?vd_source=30acb5331e4f5739ebbad50f7cc6b949                                        
https://youtu.be/iXfsJrXCEwA                        

## 1.4 anaconda、pycharm 安装       
anaconda:提供python虚拟环境，官网下载对应系统版本的安装包安装即可              
pycharm:提供集成开发环境，官网下载社区版本安装包安装即可    
可参考如下视频进行安装，基础开发环境搭建                                                    
https://www.bilibili.com/video/BV1tQWje1ErT/?vd_source=30acb5331e4f5739ebbad50f7cc6b949                                     
https://youtu.be/myVgyitFzrA                 


# 2、构建graphrag  
## 2.1 下载源码   
GitHub中下载工程文件到本地，下载地址如下：          
https://github.com/NanGePlus/GraphragTest       

## 2.2 构建项目  
 使用pycharm构建一个项目，为项目配置虚拟python环境         
项目名称：GraphragTest          

## 2.3 将相关代码拷贝到项目工程中            
直接将下载的文件夹中的文件拷贝到新建的项目目录中           

## 2.4 安装项目依赖        
pip install -r requirements.txt              
每个软件包后面都指定了本次视频测试中固定的版本号        
**注意:** 本视频使用是截止现在最新graphrag版本0.3.0            

## 2.5 创建graphrag所需文件夹          
在当前项目下创建个文件夹，**注意:** 这里的ragtest文件夹为自定义文件夹，下面所有操作均在该文件夹目录下进行操作         
mkdir -p ./ragtest          
cd ragtest     
mkdir -p ./input         
mkdir -p ./inputs         
mkdir -p ./cache        

## 2.6 准备测试文档
**注意:** 这里以西游记白话文前九回内容为例，将other/text/下的1-9.txt文件直接放入ragtest/input文件夹下       

## 2.7 初始化   
python -m graphrag.index --init  --root ./        

## 2.8 设置参数
设置.env和settings.yaml       
**注意1:** 针对阿里通义千问大模型具体参考提供的other/temp下的.env和settings.yaml文件内容，直接拷贝即可      
**注意2:** 针对智谱大模型本身的参数限制，将other/temp下的.env和settings.yaml文件内容拷贝后，需要对settings.yaml文件做如下修改          
llm:              
  temperature: 0.95 # temperature for sampling                
  top_p: 0.7 # top-p sampling                        
embeddings:                
  batch_size: 1 # the number of documents to send in a single request                       
  batch_max_tokens: 8000 # the maximum number of tokens to send in a single request        
**注意3:** 针对讯飞星火大模型本身的参数限制，将other/temp下的.env和settings.yaml文件内容拷贝后         
需要对settings.yaml文件做如下修改：                  
llm:              
  temperature: 0.5 # temperature for sampling                     
  top_p: 1 # top-p sampling                  
需要对.env文件做如下调整：          
GRAPHRAG_CHAT_MODEL=SparkDesk-v4.0（使用讯飞的chat模型）             
GRAPHRAG_EMBEDDING_MODEL=text-embedding-v1（使用阿里通义千问的embedding模型）   

## 2.9 优化提示词，选择一条适合的运行即可      
python -m graphrag.prompt_tune --config ./settings.yaml --root ./ --no-entity-types --language Chinese --output ./prompts             

## 2.10 构建索引     
python -m graphrag.index --root ./             


# 3、测试graphrag     
测试代码在utils文件夹，将other/utils文件夹直接拷贝到ragtest文件夹下             
## 3.1 运行main.py脚本   
**注意1:** 需要将代码中的如下代码中的文件路径，替换为你的对应工程的文件路径                  
INPUT_DIR = "/Users/janetjiang/Desktop/agi_code/GraphragTest/ragtest/inputs/artifacts"             
**注意2:** 大模型配置           
**注意3:** 指定向量数据库的集合名称entity_description_embeddings，根据实际情况自定义调整                             
description_embedding_store = LanceDBVectorStore(collection_name="entity_description_embeddings")

## 3.2 运行apiTest.py进行测试
main.py脚本运行成功后，新开一个终端命令行，运行apiTest.py进行测试           
**注意:** 根据需求修改messages中的query的问题      

**通义千问测试结论仅供参考:** 根据各自情况的不同，tokens数量和金额可能会有出入，但相差不会太大        
基础测试数据：西游记白话文前九回，共计19485字，千问测算15743tokens           
阿里云百炼：https://bailian.console.aliyun.com/#/data-analysis          
（1）构建索引使用的LLM模型是qwen-turbo模型、Embedding模型是text-embedding-v1(费用忽略不计)                 
**共调用LLM次数：284次，共消耗575086tokens，共消费1.421元，共耗时5-10mīn**，拆解如下：           
输入417476tokens，消费0.835元          
输出97640tokens，消费0.586元         
（2）测试搜索使用的LLM模型是qwen-plus模型、Embedding模型是text-embedding-v1(费用忽略不计)          
**共测试3次(共计2次本地搜索、2次全局搜索)，调用LLM次数：20次，共消耗61440tokens，共消费0.272元**，拆解如下：          
输入58084tokens，消费0.232元            
输出3356tokens，消费0.040元              
（3）结论，通过上面的计算，整个测试共消费1.693元，我的订单页则显示消费1.69元           

## 3.3 知识图谱使用neo4j图数据库进行可视化
首先需要进入neo4j数据库网站，使用云服务版本，这里直接打开neo4j平台，注册成功后创建实例即可       
https://workspace-preview.neo4j.io/workspace/query    
**注意1:** 需要将代码中的如下代码中的文件路径，替换为你的对应工程的文件路径,然后运行utils文件下的neoo4jTest.py脚本              
GRAPHRAG_FOLDER="/Users/janetjiang/Desktop/agi_code/GraphragTest/ragtest/inputs/artifacts"      
**注意2:** 配置自己的neo4j的数据库连接信息            

## 3.4 知识图谱使用3D图进行可视化 
**注意:** 需要将代码中的如下代码中的文件路径，替换为你的对应工程的文件路径，然后运行utils文件下的graphrag3dknowledge.py脚本            
directory = '/Users/janetjiang/Desktop/agi_code/GraphragTest/ragtest/inputs/artifacts'        
