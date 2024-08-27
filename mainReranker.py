# 功能说明：实现使用langchain框架，使用LCEL构建一个完整的LLM应用程序用于RAG知识库的查询，并使用fastapi进行发布
# 包含：langchain框架的使用，langsmith跟踪检测

# 相关依赖库
# pip install langchain langchain-openai langchain-chroma

import os
import re
import json
import asyncio
import uuid
import time
import logging
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from langchain_openai import ChatOpenAI
# prompt模版
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
# 部署REST API相关
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
# 向量数据库chroma相关
from langchain_chroma import Chroma
# openai的向量模型
from langchain_openai import OpenAIEmbeddings
# RAG相关
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import CrossEncoder


# 设置langsmith环境变量
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"

# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 向量数据库chromaDB设置相关 根据自己的实际情况进行调整
CHROMADB_DIRECTORY = "chromaDB"  # chromaDB向量数据库的持久化路径
CHROMADB_COLLECTION_NAME = "demo001"  # 待查询的chromaDB向量数据库的集合名称

# re-rank模型设置相关 根据自己的实际情况进行调整
RERANK_MODEL = 'other/models/bge-reranker-large'  # re-rank模型,也可使用ms-marco-MiniLM-L-6-v2模型

# prompt模版设置相关 根据自己的实际情况进行调整
PROMPT_TEMPLATE_TXT = "prompt_template.txt"

# 模型设置相关  根据自己的实际情况进行调整
API_TYPE = "oneapi"  # openai:调用gpt模型；oneapi:调用oneapi方案支持的模型(这里调用通义千问)
# openai模型相关配置 根据自己的实际情况进行调整
OPENAI_API_BASE = "https://api.wlai.vip/v1"
OPENAI_CHAT_API_KEY = "sk-t9pOWmiGVE02RBH88e87Eb8aE282471291F34640E787C2C6"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
OPENAI_EMBEDDING_API_KEY = "sk-t9pOWmiGVE02RBH88e87Eb8aE282471291F34640E787C2C6"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
# oneapi相关配置(通义千问为例) 根据自己的实际情况进行调整
ONEAPI_API_BASE = "http://139.224.72.218:3000/v1"
ONEAPI_CHAT_API_KEY = "sk-aFXNklkEVj0McTUf6bA4EcC2F1B84c9aB2DeB494A1EbF87c"
ONEAPI_CHAT_MODEL = "qwen-plus"
ONEAPI_EMBEDDING_API_KEY = "sk-aFXNklkEVj0McTUf6bA4EcC2F1B84c9aB2DeB494A1EbF87c"
ONEAPI_EMBEDDING_MODEL = "text-embedding-v1"

# API服务设置相关  根据自己的实际情况进行调整
PORT = 8012  # 服务访问的端口

# 申明全局变量 全局调用
query_content = ''   # 将chain中传递的用户输入的信息赋值到query_content
model = None  # 使用的LLM模型
embeddings = None  # 使用的Embedding模型
vectorstore = None  # 向量数据库实例
prompt = None  # prompt内容
chain = None  # 定义的chain



# 定义Message类
class Message(BaseModel):
    role: str
    content: str

# 定义ChatCompletionRequest类
class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False

# 定义ChatCompletionResponseChoice类
class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None

# 定义ChatCompletionResponse类
class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseChoice]
    system_fingerprint: Optional[str] = None


# 获取用户输入的内容，重置全局变量query_content，方便format_docs()函数中排序算法调用
def getQueryContent(query):
    # 申明使用全局变量query_content
    global query_content
    # 将chain中传递的用户输入的信息赋值到query_content
    query_content = query
    logger.info(f"chain中传递的用户输入的信息为: {query_content}\n")
    # 返回query_content
    return query_content


# 对retriever检索器返回的数据进行处理后返回
# 调用re-rank模型对获取的结果再进行一次相似度计算及排序
def format_docs(docs):
    # 申明使用全局变量query_content(用户输入的消息)、RERANK_MODEL(排序模型)
    global RERANK_MODEL,query_content
    # 初始化模型
    model = CrossEncoder(RERANK_MODEL, max_length=512)
    logger.info(f"chain中retriever检索器返回的数据:")
    for doc in docs:
        logger.info(f"{doc}\n\n")
    # 将检索出的5个近似的结果，再进行重新打分进行排序
    scores = model.predict([(query_content, doc.page_content) for doc in docs])
    # zip()将相似度分数与对应的文档组合成元组，并创建一个新的列表，并根据scores进行降序排序
    sorted_list = sorted(
        zip(scores, docs), key=lambda x: x[0], reverse=True)
    # 打印出前5条排序结果
    for score, doc in sorted_list:
        logger.info(f"重新排序的数据: {score}\t{doc}\n")

    # 取出排序后打分最高的前两条文本进行拼接，给到prompt模版
    documents = '\n\n'.join([item[1].page_content for item in sorted_list[:2]])
    # 打印出拼接处理后的文本
    logger.info(f"返回排序靠前的两条: {documents}")
    return documents


# 获取prompt在chain中传递的prompt最终的内容
def getPrompt(prompt):
    logger.info(f"最后给到LLM的prompt的内容: {prompt}")
    return prompt


# 格式化响应，对输入的文本进行段落分隔、添加适当的换行符，以及在代码块中增加标记，以便生成更具可读性的输出
def format_response(response):
    # 使用正则表达式 \n{2, }将输入的response按照两个或更多的连续换行符进行分割。这样可以将文本分割成多个段落，每个段落由连续的非空行组成
    paragraphs = re.split(r'\n{2,}', response)
    # 空列表，用于存储格式化后的段落
    formatted_paragraphs = []
    # 遍历每个段落进行处理
    for para in paragraphs:
        # 检查段落中是否包含代码块标记
        if '```' in para:
            # 将段落按照```分割成多个部分，代码块和普通文本交替出现
            parts = para.split('```')
            for i, part in enumerate(parts):
                # 检查当前部分的索引是否为奇数，奇数部分代表代码块
                if i % 2 == 1:  # 这是代码块
                    # 将代码块部分用换行符和```包围，并去除多余的空白字符
                    parts[i] = f"\n```\n{part.strip()}\n```\n"
            # 将分割后的部分重新组合成一个字符串
            para = ''.join(parts)
        else:
            # 否则，将句子中的句点后面的空格替换为换行符，以便句子之间有明确的分隔
            para = para.replace('. ', '.\n')
        # 将格式化后的段落添加到formatted_paragraphs列表
        # strip()方法用于移除字符串开头和结尾的空白字符（包括空格、制表符 \t、换行符 \n等）
        formatted_paragraphs.append(para.strip())
    # 将所有格式化后的段落用两个换行符连接起来，以形成一个具有清晰段落分隔的文本
    return '\n\n'.join(formatted_paragraphs)


# 定义了一个异步函数 lifespan，它接收一个FastAPI应用实例app作为参数。这个函数将管理应用的生命周期，包括启动和关闭时的操作
# 函数在应用启动时执行一些初始化操作，如设置搜索引擎、加载上下文数据、以及初始化问题生成器
# 函数在应用关闭时执行一些清理操作
# @asynccontextmanager 装饰器用于创建一个异步上下文管理器，它允许你在 yield 之前和之后执行特定的代码块，分别表示启动和关闭时的操作
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    # 申明引用全局变量，在函数中被初始化，并在整个应用中使用
    global model, embeddings, vectorstore, prompt, chain, API_TYPE, CHROMADB_DIRECTORY, CHROMADB_COLLECTION_NAME, PROMPT_TEMPLATE_TXT
    global ONEAPI_API_BASE, ONEAPI_CHAT_API_KEY, ONEAPI_CHAT_MODEL, ONEAPI_EMBEDDING_API_KEY, ONEAPI_EMBEDDING_MODEL
    global OPENAI_API_BASE, OPENAI_CHAT_API_KEY, OPENAI_CHAT_MODEL, OPENAI_EMBEDDING_API_KEY, OPENAI_EMBEDDING_MODEL
    # 根据自己实际情况选择调用model和embedding模型类型
    try:
        logger.info("正在初始化模型、实例化Chroma对象、提取prompt模版、定义chain...")
        # （1）根据API_TYPE选择初始化对应的模型
        if API_TYPE == "oneapi":
            # 实例化一个oneapi客户端对象
            model = ChatOpenAI(
                base_url=ONEAPI_API_BASE,
                api_key=ONEAPI_CHAT_API_KEY,
                model=ONEAPI_CHAT_MODEL,  # 本次使用的模型
                # temperature=0,# 发散的程度，一般为0
                # timeout=None,# 服务请求超时
                # max_retries=2,# 失败重试最大次数
            )
            # 实例化embeddings处理模型
            embeddings = OpenAIEmbeddings(
                base_url=ONEAPI_API_BASE,
                api_key=ONEAPI_EMBEDDING_API_KEY,
                model=ONEAPI_EMBEDDING_MODEL,
                deployment=ONEAPI_EMBEDDING_MODEL
            )
        elif API_TYPE == "openai":
            # 实例化一个ChatOpenAI客户端对象
            model = ChatOpenAI(
                base_url=OPENAI_API_BASE,# 请求的API服务地址
                api_key=OPENAI_CHAT_API_KEY,# API Key
                model=OPENAI_CHAT_MODEL,# 本次使用的模型
                # temperature=0,# 发散的程度，一般为0
                # timeout=None,# 服务请求超时
                # max_retries=2,# 失败重试最大次数
            )
            # 实例化embeddings处理模型
            embeddings = OpenAIEmbeddings(
                base_url=OPENAI_API_BASE,# 请求的API服务地址
                api_key=OPENAI_EMBEDDING_API_KEY,# API Key
                model=OPENAI_EMBEDDING_MODEL,
                )

        # （2）实例化Chroma对象
        # 根据自己的实际情况调整persist_directory和collection_name
        vectorstore = Chroma(persist_directory=CHROMADB_DIRECTORY,
                             collection_name=CHROMADB_COLLECTION_NAME,
                             embedding_function=embeddings,
                             )
        # （3）提取prompt模版
        prompt_template = PromptTemplate.from_file(PROMPT_TEMPLATE_TXT)
        # 测试返回的prompt_template对象中提取template的内容
        # logger.info(f"prompt_template的内容: {prompt_template.template}\n")
        prompt = ChatPromptTemplate.from_messages([("human", str(prompt_template.template))])

        # （4）定义chain
        # 将RAG检索放到LangChain的LCEL的chain中执行
        # 这段代码是使用Langchain框架中的`as_retriever`方法创建一个检索器对象
        # LangChain VectorStore对象不是 Runnable 的子类，因此无法集成到LangChain的LCEL的chain中
        # LangChain Retrievers是Runnable，实现了一组标准方法可集成到LCEL的chain中
        # `vectorstore`是一个向量存储对象，用于存储和检索文本数据
        # `as_retriever`方法将向量存储对象转换为一个检索器对象，该对象可以用于搜索与给定查询最相似的文本
        # `search_type`参数设置为"similarity"，表示使用相似度搜索算法
        # `search_kwargs`参数是一个字典，包含搜索算法的参数，这里的`k`参数设置为5，表示只返回与查询最相似的5个结果
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )
        # 定义chain
        # 先构建prompt模版，将用户输入消息直接赋值给prompt模版中的{query}
        # 将根据用户的输入消息调用进行数据检索并将检索结果传递给format_docs函数进行处理，返回值赋值给prompt模版中的{context}
        # 将完整的prompt给到model执行
        chain = {
                    "query": RunnablePassthrough() | getQueryContent,
                    "context": retriever | format_docs
                } | prompt | getPrompt | model

        logger.info("初始化完成")

    except Exception as e:
        logger.error(f"初始化过程中出错: {str(e)}")
        # raise 关键字重新抛出异常，以确保程序不会在错误状态下继续运行
        raise

    # yield 关键字将控制权交还给FastAPI框架，使应用开始运行
    # 分隔了启动和关闭的逻辑。在yield 之前的代码在应用启动时运行，yield 之后的代码在应用关闭时运行
    yield
    # 关闭时执行
    logger.info("正在关闭...")


# lifespan 参数用于在应用程序生命周期的开始和结束时执行一些初始化或清理工作
app = FastAPI(lifespan=lifespan)


# POST请求接口，与大模型进行知识问答
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # 申明引用全局变量，在函数中被初始化，并在整个应用中使用
    if not model or not embeddings or not vectorstore or not prompt or not chain:
        logger.error("服务未初始化")
        raise HTTPException(status_code=500, detail="服务未初始化")

    try:
        logger.info(f"收到聊天完成请求: {request}")
        query_prompt = request.messages[-1].content
        logger.info(f"用户问题是: {query_prompt}")
        # 调用chain进行查询
        result = chain.invoke(query_prompt)
        formatted_response = str(format_response(result.content))
        logger.info(f"格式化的搜索结果: {formatted_response}")

        # 处理流式响应
        if request.stream:
            # 定义一个异步生成器函数，用于生成流式数据
            async def generate_stream():
                # 为每个流式数据片段生成一个唯一的chunk_id
                chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
                # 将格式化后的响应按行分割
                lines = formatted_response.split('\n')
                # 历每一行，并构建响应片段
                for i, line in enumerate(lines):
                    # 创建一个字典，表示流式数据的一个片段
                    chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        # "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": line + '\n'}, # if i > 0 else {"role": "assistant", "content": ""},
                                "finish_reason": None
                            }
                        ]
                    }
                    # 将片段转换为JSON格式并生成
                    yield f"{json.dumps(chunk)}\n"
                    # 每次生成数据后，异步等待0.5秒
                    await asyncio.sleep(0.5)
                # 生成最后一个片段，表示流式响应的结束
                final_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"{json.dumps(final_chunk)}\n"

            # 返回fastapi.responses中StreamingResponse对象，流式传输数据
            # media_type设置为text/event-stream以符合SSE(Server-SentEvents) 格式
            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        # 处理非流式响应处理
        else:
            response = ChatCompletionResponse(
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=Message(role="assistant", content=formatted_response),
                        finish_reason="stop"
                    )
                ]
            )
            logger.info(f"发送响应内容: \n{response}")
            # 返回fastapi.responses中JSONResponse对象
            # model_dump()方法通常用于将Pydantic模型实例的内容转换为一个标准的Python字典，以便进行序列化
            return JSONResponse(content=response.model_dump())

    except Exception as e:
        logger.error(f"处理聊天完成时出错:\n\n {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))




if __name__ == "__main__":
    logger.info(f"在端口 {PORT} 上启动服务器")
    # uvicorn是一个用于运行ASGI应用的轻量级、超快速的ASGI服务器实现
    # 用于部署基于FastAPI框架的异步PythonWeb应用程序
    uvicorn.run(app, host="0.0.0.0", port=PORT)


