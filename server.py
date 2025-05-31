import os
import smtplib
import re
from datetime import datetime
from email.message import EmailMessage
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import logging
from utils.rag_chain import query_hospital_with_rag
from utils.appointment_tool import manage_appointment as core_manage_appointment
from utils.query_appointment import query_appointment as core_query_appointment

# 加载环境变量
load_dotenv()

# 初始化 MCP 服务器
mcp = FastMCP("MyMCPServer")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    if not smtp_server:
        return "❌ 未配置 SMTP_SERVER，请在 .env 文件中设置"
    if not sender_email:
        return "❌ 未配置 EMAIL_USER，请在 .env 文件中设置"
    if not sender_pass:
        return "❌ 未配置 EMAIL_PASS，请在 .env 文件中设置"

    try:
        smtp_port = int(smtp_port)
    except ValueError:
        return "❌ SMTP_PORT 配置错误，必须是有效的端口号"

    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, to):
        return f"❌ 收件人邮箱地址格式不正确: {to}"
    if not re.match(email_pattern, sender_email):
        return f"❌ 发件人邮箱地址格式不正确: {sender_email}"

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = to

        if not body:
            body = f"这是一封由 AI 助手自动发送的邮件。\n\n发送时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        msg.set_content(body)

        # 添加附件（如果有）
        attachment_info = ""
        if attachment_path:
            full_path = None
            if os.path.isabs(attachment_path):
                if os.path.exists(attachment_path):
                    full_path = attachment_path
                else:
                    return f"❌ 附件文件不存在: {attachment_path}"
            else:
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

            try:
                with open(full_path, "rb") as f:
                    file_data = f.read()
                    file_name = os.path.basename(full_path)
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

                    attachment_info = f"，附件: {file_name}"
                    print(f"[DEBUG] 附件已添加: {full_path}")
            except Exception as e:
                return f"❌ 附件读取失败: {str(e)}"

        # 发送邮件
        print(f"[DEBUG] 正在连接 SMTP: {smtp_server}:{smtp_port}")
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            print("[DEBUG] SMTP 连接成功，正在登录")
            server.login(sender_email, sender_pass)
            print("[DEBUG] SMTP 登录成功，正在发送邮件")
            server.send_message(msg)
            print("[DEBUG] 邮件发送成功")

        return f"✅ 邮件已成功发送给 {to}{attachment_info}"

    except smtplib.SMTPAuthenticationError:
        return "❌ SMTP 认证失败，请检查邮箱账号或密码"
    except smtplib.SMTPConnectError:
        return f"❌ 无法连接 SMTP 服务器：{smtp_server}:{smtp_port}"
    except smtplib.SMTPException as e:
        return f"❌ SMTP 错误: {str(e)}"
    except Exception as e:
        return f"❌ 邮件发送失败: {str(e)}"


@mcp.tool()
async def query_hospital_knowledge(question: str, max_results: int = 5) -> str:
    """
    查询北京协和医院相关信息，使用本地RAG知识库结合大语言模型回答问题。

    参数:
        question (str): 用户提出的关于北京协和医院的问题。
        max_results (int): 检索的相关文档数量，默认为5。

    返回:
        str: 查询结果，包括AI生成的答案与参考文档数量。
    """
    try:
        result = query_hospital_with_rag(question, max_results)
        answer = result["answer"]
        doc_count = len(result["source_documents"])

        return f"[成功] 北京协和医院知识库查询完成！\n\n{answer}\n\n[参考] 本回答基于 {doc_count} 个相关文档"

    except Exception as e:
        return f"[错误] 知识库查询失败: {str(e)}"


@mcp.tool()
async def manage_appointment(
        action: str,  # 操作类型："预约"/"添加"，"修改"/"更新"，"删除"/"取消"
        username: str = None,  # 用户姓名（预约和修改时必填）
        id_card: str = None,  # 身份证号（预约和修改时必填）
        department: str = None,  # 预约科室（可选）
        date: str = None,  # 预约日期，格式 YYYY-MM-DD（可选）
        time: str = None,  # 预约时间，格式 HH:MM:SS（可选）
        appointment_id: int = None,  # 预约记录 ID（修改和删除时可选）
        access_token: str = None  # 身份验证码（修改和删除时必填）
) -> str:
    """
    MCP 工具：统一入口管理预约系统（封装数据库操作）

    参数说明：
        - action: 操作类型（"预约"/"添加"，"修改"/"更新"，"删除"/"取消"）
        - username: 用户姓名（创建和修改时必填）
        - id_card: 用户身份证（创建和修改时必填）
        - department: 预约科室（可选）
        - date: 预约日期（格式为 YYYY-MM-DD，选填）
        - time: 预约时间（格式为 HH:MM:SS，选填）
        - appointment_id: 指定的预约记录 ID（可选）
        - access_token: 验证用户身份的验证码（修改和删除操作必填）

    返回值：
        - str: 包含操作结果的提示文本，可能是成功提示或错误原因
    """
    return core_manage_appointment(
        action=action,
        username=username,
        id_card=id_card,
        department=department,
        date=date,
        time=time,
        appointment_id=appointment_id,
        access_token=access_token
    )


@mcp.tool()
async def query_appointment(
        username: str = None,
        id_card: str = None,
        appointment_id: int = None,
        access_token: str = None
) -> str:
    """
    MCP 工具：查询预约记录

    参数:
        - username: 用户姓名（可选）
        - id_card: 身份证号（可选）
        - appointment_id: 预约ID（可选）
        - access_token: 验证码（可选）

    返回:
        - str: 查询结果
    """
    return core_query_appointment(
        username=username,
        id_card=id_card,
        appointment_id=appointment_id,
        access_token=access_token
    )


if __name__ == "__main__":
    mcp.run(transport='stdio')
