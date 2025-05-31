import os
import glob
from typing import List
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import logging
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalKnowledgeVectorStore:
    def __init__(
            self,
            knowledge_path: str = r"D:\Download\knowledge\knowledge",
            index_name: str = "mcp_medical",
            persist_directory: str = "./chroma_db"
    ):
        """
        初始化医疗知识向量存储系统

        Args:
            knowledge_path: MD文件所在路径
            index_name: 索引名称
            persist_directory: ChromaDB持久化目录
        """
        self.knowledge_path = knowledge_path
        self.index_name = index_name
        self.persist_directory = persist_directory

        # 从.env文件读取OpenAI API密钥
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("未找到OPENAI_API_KEY，请在.env文件中设置")

        logger.info("已从.env文件加载OpenAI API密钥")

        # 初始化OpenAI Embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=openai_api_key
        )

        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # 初始化ChromaDB客户端
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        self.vectorstore = None

    def load_markdown_files(self) -> List[Document]:
        """
        加载指定路径下的所有MD文件

        Returns:
            List[Document]: 文档列表
        """
        documents = []

        # 检查路径是否存在
        if not os.path.exists(self.knowledge_path):
            raise FileNotFoundError(f"知识库路径不存在: {self.knowledge_path}")

        # 获取所有MD文件
        md_files = glob.glob(os.path.join(self.knowledge_path, "**/*.md"), recursive=True)

        if not md_files:
            logger.warning(f"在路径 {self.knowledge_path} 中未找到MD文件")
            return documents

        logger.info(f"找到 {len(md_files)} 个MD文件")

        # 加载每个MD文件
        for file_path in md_files:
            try:
                # 使用TextLoader加载文件
                loader = TextLoader(file_path, encoding='utf-8')
                file_docs = loader.load()

                # 为每个文档添加元数据
                for doc in file_docs:
                    doc.metadata.update({
                        'source': file_path,
                        'filename': os.path.basename(file_path),
                        'file_type': 'markdown',
                        'collection': self.index_name
                    })

                documents.extend(file_docs)
                logger.info(f"成功加载文件: {os.path.basename(file_path)}")

            except Exception as e:
                logger.error(f"加载文件失败 {file_path}: {str(e)}")
                continue

        logger.info(f"总共加载了 {len(documents)} 个文档")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分割文档为小块

        Args:
            documents: 原始文档列表

        Returns:
            List[Document]: 分割后的文档块列表
        """
        logger.info("开始分割文档...")
        split_docs = self.text_splitter.split_documents(documents)
        logger.info(f"文档分割完成，共生成 {len(split_docs)} 个文档块")
        return split_docs

    def create_vectorstore(self, documents: List[Document]) -> None:
        """
        创建向量存储

        Args:
            documents: 文档列表
        """
        logger.info("开始创建向量存储...")

        try:
            # 检查集合是否已存在，如果存在则删除
            try:
                existing_collection = self.chroma_client.get_collection(self.index_name)
                self.chroma_client.delete_collection(self.index_name)
                logger.info(f"删除已存在的集合: {self.index_name}")
            except:
                pass

            # 创建向量存储
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                client=self.chroma_client,
                collection_name=self.index_name,
                persist_directory=self.persist_directory
            )

            logger.info(f"向量存储创建成功，集合名称: {self.index_name}")

        except Exception as e:
            logger.error(f"创建向量存储失败: {str(e)}")
            raise

    def load_existing_vectorstore(self) -> bool:
        """
        加载已存在的向量存储

        Returns:
            bool: 是否成功加载
        """
        try:
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=self.index_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            logger.info(f"成功加载已存在的向量存储: {self.index_name}")
            return True
        except Exception as e:
            logger.error(f"加载向量存储失败: {str(e)}")
            return False

    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        搜索相似文档

        Args:
            query: 搜索查询
            k: 返回结果数量

        Returns:
            List[Document]: 相似文档列表
        """
        if not self.vectorstore:
            raise ValueError("向量存储未初始化，请先创建或加载向量存储")

        results = self.vectorstore.similarity_search(query, k=k)
        return results

    def get_collection_info(self) -> dict:
        """
        获取集合信息

        Returns:
            dict: 集合信息
        """
        if not self.vectorstore:
            return {"error": "向量存储未初始化"}

        try:
            collection = self.chroma_client.get_collection(self.index_name)
            count = collection.count()
            return {
                "collection_name": self.index_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            return {"error": str(e)}

    def build_knowledge_base(self, force_rebuild: bool = False) -> None:
        """
        构建知识库

        Args:
            force_rebuild: 是否强制重建
        """
        logger.info("开始构建医疗知识库...")

        # 如果不强制重建，尝试加载已存在的向量存储
        if not force_rebuild and self.load_existing_vectorstore():
            logger.info("使用已存在的向量存储")
            return
        # 不然就进行重建：加载文档、分割文档、创建向量存储
        # 加载MD文件
        documents = self.load_markdown_files()

        if not documents:
            logger.error("没有找到可用的文档")
            return

        # 分割文档
        split_docs = self.split_documents(documents)

        # 创建向量存储
        self.create_vectorstore(split_docs)

        logger.info("知识库构建完成！")


def main():
    """
    主函数 - 演示如何使用
    """
    # 配置参数
    KNOWLEDGE_PATH = r"D:\Download\knowledge\knowledge"
    INDEX_NAME = "mcp_medical"

    try:
        # 创建向量存储实例
        vector_store = MedicalKnowledgeVectorStore(
            knowledge_path=KNOWLEDGE_PATH,
            index_name=INDEX_NAME
        )

        # 构建知识库
        vector_store.build_knowledge_base(force_rebuild=True)

        # 获取集合信息
        info = vector_store.get_collection_info()
        print(f"集合信息: {info}")

        # 演示搜索功能
        query = "医疗诊断"
        results = vector_store.search(query, k=3)

        print(f"\n搜索查询: {query}")
        print(f"找到 {len(results)} 个相关文档:")
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. 文件: {doc.metadata.get('filename', 'Unknown')}")
            print(f"   内容预览: {doc.page_content[:200]}...")

    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")


if __name__ == "__main__":
    main()
