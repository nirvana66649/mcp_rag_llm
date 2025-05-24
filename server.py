import os
import json
import smtplib
import re
from datetime import datetime
from email.message import EmailMessage
import httpx
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from openai import OpenAI
import pymysql
from tabulate import tabulate
import chromadb
from chromadb.config import Settings
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import logging

# 加载环境变量
load_dotenv()

# 初始化 MCP 服务器
mcp = FastMCP("MyMCPServer")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@mcp.tool()
async def search_google_news(keyword: str) -> str:
    """
    使用 Serper API 根据关键词搜索新闻内容，返回前5条标题、描述和链接。

    参数:
        keyword (str): 搜索关键词，如 "小米汽车"

    返回:
        str: 新闻搜索结果的JSON格式字符串
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "[错误] 未配置 SERPER_API_KEY，请在 .env 文件中设置"

    url = "https://google.serper.dev/news"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    payload = {"q": keyword}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            data = response.json()

        if "news" not in data:
            return "[错误] 未获取到搜索结果"

        articles = [
            {
                "title": item.get("title", ""),
                "desc": item.get("snippet", ""),
                "url": item.get("link", "")
            } for item in data["news"][:5]
        ]

        # 保存结果到文件
        output_dir = "./google_news"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"google_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file_path = os.path.join(output_dir, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)

        result_text = f"[成功] 已获取与 [{keyword}] 相关的前5条 Google 新闻：\n"
        for i, article in enumerate(articles, 1):
            result_text += f"\n{i}. {article['title']}\n   {article['desc']}\n   链接: {article['url']}\n"

        result_text += f"\n[保存] 详细结果已保存到：{file_path}"
        return result_text

    except Exception as e:
        return f"[错误] 搜索新闻时发生错误: {str(e)}"


@mcp.tool()
async def analyze_sentiment(text: str, filename: str = None) -> str:
    """
    对文本内容进行情感分析，并保存为 Markdown 文件。

    参数:
        text (str): 要分析的文本内容
        filename (str): 可选，保存的文件名（不含路径），如不提供则自动生成

    返回:
        str: 分析结果和文件保存路径
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("MODEL")
    base_url = os.getenv("BASE_URL")

    if not openai_key:
        return "[错误] 未配置 OPENAI_API_KEY，无法进行情感分析"

    try:
        client = OpenAI(api_key=openai_key, base_url=base_url)

        prompt = f"""请对以下文本进行详细的情感分析，包括：
1. 整体情感倾向（正面/负面/中性）
2. 情感强度（强/中/弱）
3. 具体情感类型（喜悦、愤怒、担忧等）
4. 分析依据和关键词

文本内容：
{text}"""

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content.strip()

        # 生成文件名
        if not filename:
            # 提取文本关键词作为文件名
            clean_text = re.sub(r'[^\w\s]', '', text)
            words = clean_text.split()[:3]  # 取前3个词
            keyword = '_'.join(words) if words else 'sentiment'
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"sentiment_{keyword}_{timestamp}.md"

        # 生成报告
        markdown = f"""# 情感分析报告

**分析时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📥 原始文本

```
{text}
```

---

## 📊 分析结果

{result}

---

*本报告由 AI 自动生成*
"""

        # 保存文件
        output_dir = "./sentiment_reports"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown)

        return f"[成功] 情感分析完成！\n\n{result}\n\n[保存] 详细报告已保存到：{file_path}"

    except Exception as e:
        return f"[错误] 情感分析失败: {str(e)}"


@mcp.tool()
async def send_email_with_attachment(to: str, subject: str, body: str = None, attachment_path: str = None) -> str:
    """
    发送带附件的邮件。

    参数:
        to (str): 收件人邮箱地址
        subject (str): 邮件标题
        body (str): 可选，邮件正文内容
        attachment_path (str): 可选，附件文件路径或文件名

    返回:
        str: 邮件发送结果
    """
    # 获取邮件配置并验证
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = os.getenv("SMTP_PORT", "465")
    sender_email = os.getenv("EMAIL_USER")
    sender_pass = os.getenv("EMAIL_PASS")

    # 严格验证配置
    if not smtp_server:
        return "❌ 未配置 SMTP_SERVER，请在 .env 文件中设置"
    if not sender_email:
        return "❌ 未配置 EMAIL_USER，请在 .env 文件中设置"
    if not sender_pass:
        return "❌ 未配置 EMAIL_PASS，请在 .env 文件中设置"

    # 验证端口号
    try:
        smtp_port = int(smtp_port)
    except ValueError:
        return "❌ SMTP_PORT 配置错误，必须是有效的端口号"

    # 验证邮箱地址格式
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, to):
        return f"❌ 收件人邮箱地址格式不正确: {to}"
    if not re.match(email_pattern, sender_email):
        return f"❌ 发件人邮箱地址格式不正确: {sender_email}"

    try:
        # 创建邮件对象
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = to

        # 设置邮件正文
        if not body:
            body = f"这是一封由 AI 助手自动发送的邮件。\n\n发送时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        msg.set_content(body)

        # 处理附件
        if attachment_path:
            full_path = None

            # 如果是绝对路径，直接使用
            if os.path.isabs(attachment_path):
                if os.path.exists(attachment_path):
                    full_path = attachment_path
                else:
                    return f"❌ 附件文件不存在: {attachment_path}"
            else:
                # 相对路径，在多个目录中查找文件
                possible_paths = [
                    os.path.abspath(os.path.join("./sentiment_reports", attachment_path)),
                    os.path.abspath(os.path.join("./google_news", attachment_path)),
                    os.path.abspath(os.path.join("./llm_outputs", attachment_path)),
                    os.path.abspath(attachment_path)
                ]

                for path in possible_paths:
                    if os.path.exists(path):
                        full_path = path
                        break

                if not full_path:
                    searched_paths = '\n'.join(possible_paths)
                    return f"❌ 未找到附件文件: {attachment_path}\n已搜索路径:\n{searched_paths}"

            # 读取并添加附件
            try:
                with open(full_path, "rb") as f:
                    file_data = f.read()
                    file_name = os.path.basename(full_path)

                    # 根据文件扩展名设置正确的 MIME 类型
                    file_ext = os.path.splitext(file_name)[1].lower()
                    if file_ext in ['.txt', '.md']:
                        msg.add_attachment(file_data, maintype="text", subtype="plain", filename=file_name)
                    elif file_ext == '.json':
                        msg.add_attachment(file_data, maintype="application", subtype="json", filename=file_name)
                    elif file_ext in ['.jpg', '.jpeg', '.png', '.gif']:
                        msg.add_attachment(file_data, maintype="image", subtype=file_ext[1:], filename=file_name)
                    else:
                        msg.add_attachment(file_data, maintype="application", subtype="octet-stream",
                                           filename=file_name)

                print(f"[DEBUG] 附件已添加: {full_path}")
            except Exception as e:
                return f"❌ 附件读取失败: {str(e)}"

        # 发送邮件
        print(f"[DEBUG] 正在连接到 {smtp_server}:{smtp_port}")
        print(f"[DEBUG] 发件人: {sender_email}")
        print(f"[DEBUG] 收件人: {to}")

        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            print("[DEBUG] SMTP 连接建立成功")
            server.login(sender_email, sender_pass)
            print("[DEBUG] SMTP 登录成功")
            server.send_message(msg)
            print("[DEBUG] 邮件发送成功")

        attachment_info = f"，附件: {os.path.basename(full_path)}" if attachment_path and full_path else ""
        return f"✅ 邮件已成功发送给 {to}{attachment_info}"

    except smtplib.SMTPAuthenticationError:
        return "❌ SMTP 认证失败：请检查邮箱用户名和密码是否正确"
    except smtplib.SMTPConnectError:
        return f"❌ 无法连接到 SMTP 服务器：{smtp_server}:{smtp_port}"
    except smtplib.SMTPServerDisconnected:
        return "❌ SMTP 服务器连接意外断开"
    except smtplib.SMTPException as e:
        return f"❌ SMTP 错误: {str(e)}"
    except Exception as e:
        return f"❌ 邮件发送失败: {str(e)}"


@mcp.tool()
async def nl_query_mysql(nl_query: str) -> str:
    """
    接收自然语言查询，自动生成并执行 SQL 语句。

    参数:
        nl_query (str): 自然语言查询，如 "查询所有预约记录" 或 "修改张三的预约时间为明天下午2点"

    返回:
        str: 查询或操作结果
    """
    try:
        # 使用大模型生成 SQL
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("BASE_URL")
        )
        model = os.getenv("MODEL")

        # 改进的 prompt，支持更多 SQL 操作
        prompt = f"""你是一个 SQL 专家。请根据用户的自然语言意图，生成适用于 MySQL 的 SQL 语句。

        数据库结构：
        - appointment 表：包含字段 id(主键, 自增), username(姓名), id_card(身份证), department(科室), date(日期), time(时间)

        用户意图：{nl_query}

        要求：
        1. 只返回 SQL 语句，不要任何解释
        2. 如果是查询操作，使用 SELECT
        3. 如果是修改操作，使用 UPDATE
        4. 如果是删除操作，使用 DELETE
        5. 如果是插入操作，使用 INSERT
        6. 字符串值要用单引号包围
        7. 日期格式使用 'YYYY-MM-DD'，时间格式使用 'HH:MM:SS'
        8. 插入 appointment 表时，id 字段为自增主键，请在 SQL 中设置为 NULL 或省略该字段

        SQL语句："""

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )

        sql = response.choices[0].message.content.strip()
        # 清理 SQL 语句
        sql = sql.replace('```sql', '').replace('```', '').strip().rstrip(';')

        print(f"[DEBUG] 生成的 SQL: {sql}")

        # 连接数据库并执行 SQL
        connection = pymysql.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 3306)),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "test"),
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True  # 自动提交事务
        )

        with connection:
            with connection.cursor() as cursor:
                cursor.execute(sql)

                # 判断 SQL 类型
                command = sql.strip().split()[0].lower()

                if command == "select":
                    result = cursor.fetchall()
                    if not result:
                        return f"[成功] 查询执行成功，但没有找到匹配的记录。\n执行的SQL: {sql}"

                    # 美化输出表格
                    table = tabulate(result, headers="keys", tablefmt="grid", stralign="center")
                    return f"[成功] 查询结果：\n\n{table}\n\n执行的SQL: {sql}"

                elif command in ["update", "delete", "insert"]:
                    affected_rows = cursor.rowcount
                    operation_name = {"update": "更新", "delete": "删除", "insert": "插入"}[command]
                    return f"[成功] {operation_name}操作执行成功，影响了 {affected_rows} 行记录。\n执行的SQL: {sql}"

                else:
                    return f"[成功] SQL 执行完成。\n执行的SQL: {sql}"

    except pymysql.Error as e:
        return f"[错误] 数据库操作失败: {str(e)}\n尝试执行的SQL: {sql}"
    except Exception as e:
        return f"[错误] 处理查询时发生错误: {str(e)}"


@mcp.tool()
async def query_hospital_knowledge(question: str, max_results: int = 5) -> str:
    """
    查询北京协和医院相关信息，使用本地RAG知识库结合大语言模型回答问题。

    参数:
        question (str): 关于北京协和医院的问题
        max_results (int): 检索的相关文档数量，默认5个

    返回:
        str: 基于知识库的详细回答
    """
    try:
        # 检查是否与北京协和医院相关
        hospital_keywords = ["北京协和医院", "协和医院", "协和", "PUMCH", "医院", "医疗", "诊疗", "科室", "医生", "挂号", "就诊"]
        if not any(keyword in question for keyword in hospital_keywords):
            return f"[提示] 此工具专门用于查询北京协和医院相关信息。您的问题似乎不相关，建议重新描述问题。\n问题：{question}"

        # 获取OpenAI API配置
        openai_api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL")
        model = os.getenv("MODEL")

        if not openai_api_key:
            return "[错误] 未配置 OPENAI_API_KEY，无法进行RAG查询"

        # 初始化Embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=openai_api_key
        )

        # 初始化ChromaDB客户端
        persist_directory = "./chroma_db"
        index_name = "mcp_medical"

        if not os.path.exists(persist_directory):
            return f"[错误] 知识库不存在，请先运行rag.py创建知识库。路径：{persist_directory}"

        chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=False
            )
        )

        # 加载向量存储
        try:
            vectorstore = Chroma(
                client=chroma_client,
                collection_name=index_name,
                embedding_function=embeddings,
                persist_directory=persist_directory
            )
            logger.info(f"成功加载向量存储: {index_name}")
        except Exception as e:
            return f"[错误] 无法加载知识库: {str(e)}\n请确保已运行rag.py创建了知识库"

        # 检索相关文档
        try:
            relevant_docs = vectorstore.similarity_search(question, k=max_results)
            if not relevant_docs:
                return f"[错误] 未找到与问题相关的文档内容。\n问题：{question}"

            logger.info(f"检索到 {len(relevant_docs)} 个相关文档")
        except Exception as e:
            return f"[错误] 文档检索失败: {str(e)}"

        # 构建上下文信息
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            filename = doc.metadata.get('filename', 'Unknown')
            content = doc.page_content.strip()
            context_parts.append(f"文档{i} ({filename}):\n{content}")

        context = "\n\n---\n\n".join(context_parts)

        # 构建RAG prompt
        rag_prompt = f"""作为北京协和医院的智能助手，请基于以下知识库内容回答用户问题。

【知识库内容】
{context}

【用户问题】
{question}

【回答要求】
1. 基于知识库内容进行回答，确保信息准确
2. 如果知识库中没有直接相关信息，请明确说明
3. 回答要详细、专业，但易于理解
4. 如涉及医疗建议，请提醒用户咨询专业医生
5. 如果是关于挂号、就诊流程等，请提供具体指导

【回答】"""

        # 调用大语言模型生成回答
        try:
            client = OpenAI(api_key=openai_api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": rag_prompt}],
                temperature=0.3,  # 降低温度以提高准确性
                max_tokens=1500
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            return f"[错误] 大语言模型调用失败: {str(e)}"

        # 保存查询结果
        try:
            output_dir = "./rag_outputs"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"hospital_query_{timestamp}.md"
            file_path = os.path.join(output_dir, filename)

            report = f"""# 北京协和医院知识库查询报告

**查询时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📋 用户问题

{question}

---

## 🤖 AI回答

{answer}

---

## 📚 参考文档

"""
            for i, doc in enumerate(relevant_docs, 1):
                filename = doc.metadata.get('filename', 'Unknown')
                source = doc.metadata.get('source', 'Unknown')
                content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                report += f"""
### 文档 {i}: {filename}

**来源：** {source}

**内容预览：**
```
{content_preview}
```

"""

            report += "\n---\n\n*本报告由RAG系统自动生成*"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(report)

            result = f"[成功] 北京协和医院知识库查询完成！\n\n{answer}\n\n[保存] 详细报告已保存到：{file_path}\n[参考] 本回答基于 {len(relevant_docs)} 个相关文档"
            return result

        except Exception as e:
            logger.error(f"保存查询结果失败: {str(e)}")
            # 即使保存失败，也返回查询结果
            return f"[成功] 北京协和医院知识库查询完成！\n\n{answer}\n\n[参考] 本回答基于 {len(relevant_docs)} 个相关文档"

    except Exception as e:
        logger.error(f"RAG查询失败: {str(e)}")
        return f"[错误] 知识库查询失败: {str(e)}"


@mcp.tool()
async def cleanup_old_files() -> str:
    """
    检查指定目录中的文件数量是否超过 5，如果超过则删除该目录下的所有文件。

    返回:
        str: 清理操作结果
    """
    folders = [
        r"D:\PythonProject\mcp-project\google_news",
        r"D:\PythonProject\mcp-project\llm_outputs",
        r"D:\PythonProject\mcp-project\rag_outputs"
    ]
    threshold = 5
    summary = ""

    for folder in folders:
        try:
            if not os.path.exists(folder):
                summary += f"📂 目录不存在：{folder}\n"
                continue

            files = [os.path.join(folder, f) for f in os.listdir(folder)
                     if os.path.isfile(os.path.join(folder, f))]

            if len(files) > threshold:
                for file_path in files:
                    os.remove(file_path)
                summary += f"🗑️ {folder} 超过 {threshold} 个文件，已全部删除（共 {len(files)} 个文件）。\n"
            else:
                summary += f"✅ {folder} 文件数量为 {len(files)}，未超过 {threshold}，无需清理。\n"

        except Exception as e:
            summary += f"❌ 清理 {folder} 时出错: {str(e)}\n"

    return summary


if __name__ == "__main__":
    mcp.run(transport='stdio')