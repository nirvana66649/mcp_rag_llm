# rag_chain.py

import os
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA


def query_hospital_with_rag(question: str, max_results: int = 5) -> dict:
    """
    使用本地向量数据库和大语言模型结合的RAG方式回答问题。

    参数:
        question (str): 用户问题
        max_results (int): 召回的文档数量

    返回:
        dict: 包含 answer（str）和 source_documents（List[Document]）
    """
    # 获取环境变量
    openai_api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL")
    model = os.getenv("MODEL")

    if not openai_api_key:
        raise ValueError("未配置 OPENAI_API_KEY，无法进行RAG查询")

    # 初始化 Embedding 和 Chroma 向量数据库
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=openai_api_key
    )

    persist_directory = "./chroma_db"
    index_name = "mcp_medical"

    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"知识库不存在，请先运行rag.py创建。路径：{persist_directory}")
    # 你必须先创建并配置一个连接到本地 Chroma 向量数据库的客户端 (PersistentClient)，然后才能通过它去操作 collection 实例（Chroma），否则系统不知道去哪里加载数据，也不知道用什么配置。
    chroma_client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(anonymized_telemetry=False, allow_reset=False)
    )

    vectorstore = Chroma(
        client=chroma_client,
        collection_name=index_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": max_results})

    # 构建问答链
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        base_url=base_url,
        model_name=model,
        temperature=0.2
    )
    # 构建一个完整的问答链，包含检索和生成答案的步骤。
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff"
    )

    result = qa_chain.invoke({"query": question})

    return {
        "answer": result["result"].strip(),
        "source_documents": result.get("source_documents", [])
    }
