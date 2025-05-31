import asyncio
import os
import json
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from utils.custom_mongo_history import CustomMongoChatMessageHistory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from pymongo import MongoClient
import uuid
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
load_dotenv()


def safe_messages_to_dict(messages):
    result = []
    for m in messages:
        if isinstance(m, HumanMessage):
            result.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            result.append({"role": "assistant", "content": m.content})
        elif isinstance(m, SystemMessage):
            result.append({"role": "system", "content": m.content})
        else:
            print(f"[警告] 忽略未知消息类型: {type(m)}")
    return result


def get_system_prompt_from_file(file_path: str = "system_prompt.txt") -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到 system prompt 文件: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        template = f.read()

    current_time = datetime.now()
    context = {
        "current_date": current_time.strftime("%Y年%m月%d日"),
        "current_datetime": current_time.strftime("%Y年%m月%d日 %H:%M:%S"),
        "weekday_cn": {
            "Monday": "星期一",
            "Tuesday": "星期二",
            "Wednesday": "星期三",
            "Thursday": "星期四",
            "Friday": "星期五",
            "Saturday": "星期六",
            "Sunday": "星期日"
        }.get(current_time.strftime("%A"), "未知"),
        "tomorrow": (current_time + timedelta(days=1)).strftime('%Y年%m月%d日')
    }

    # 替换模板中的变量
    for key, value in context.items():
        template = template.replace(f"{{{{{key}}}}}", value)

    return template



class MCPClient:
    def __init__(self, server_script_path: str, session_id: Optional[str] = None):
        self.exit_stack = AsyncExitStack()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("MODEL")
        if not self.openai_api_key:
            raise ValueError("❌ 未找到 OpenAI API Key，请在 .env 文件中设置 OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.openai_api_key, base_url=self.base_url)

        self.mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        self.mongo_db = self.mongo_client["chat_memory"]
        self.collection_name = "mcp_memory"

        # ✅ 使用传入的 session_id，否则生成一个新的 UUID
        self.memory_id = session_id or str(uuid.uuid4())

        # 构造 memory（保持你的逻辑）
        chat_history = CustomMongoChatMessageHistory(
            session_id=self.memory_id,
            connection_string=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
            database_name="chat_memory",
            collection_name=self.collection_name
        )

        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            chat_memory=chat_history,
            k=20,
        )

        self.session: Optional[ClientSession] = None
        self.server_script_path = server_script_path

    async def __aenter__(self):
        await self.connect_to_server(self.server_script_path)
        return self

    async def __aexit__(self, *exc_info):
        await self.cleanup()

    async def connect_to_server(self, server_script_path: str):
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("服务器脚本必须是 .py 或 .js 文件")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        response = await self.session.list_tools()
        tools = response.tools
        print("\n已连接到服务器，支持以下工具:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        # 1. 获取当前所有可用的工具列表
        response = await self.session.list_tools()
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            } for tool in response.tools
        ]

        # 2. 获取历史消息记录（从 memory 中读出最近 20 条，已由 MongoDB 限制）
        history_messages = self.memory.chat_memory.messages
        messages = safe_messages_to_dict(history_messages)

        # 3. 插入或更新 system prompt，确保首条消息为系统设定
        current_system_prompt = get_system_prompt_from_file("system_prompt.txt")
        if not messages or messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": current_system_prompt})
        else:
            messages[0]["content"] = current_system_prompt

        # 4. 将用户输入添加到对话中
        messages.append({"role": "user", "content": query})

        # 5. 输出当前请求调试信息
        print(f"\n[查询] 用户查询: {query}")
        print(f"[时间] 当前系统时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[工具] 可用工具: {[tool['function']['name'] for tool in available_tools]}")

        try:
            # 6. 初次向模型发送请求（包含工具选择）
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=available_tools,
                tool_choice="auto"  # 让模型自动决定是否调用工具
            )

            # 7. 获取模型回复
            assistant_message = response.choices[0].message

            # 8. 追加 assistant 回复消息（包含可能的 tool_calls）
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": assistant_message.tool_calls
            })

            # 9. 如果模型决定调用工具
            if assistant_message.tool_calls:
                print(f"\n[工具调用] 模型决定调用 {len(assistant_message.tool_calls)} 个工具")

                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        # 解析工具调用参数
                        tool_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}

                    print(f"[调用] 工具名称: {tool_name}")
                    print(f"[参数] {tool_args}")

                    try:
                        # 执行工具调用
                        result = await self.session.call_tool(tool_name, tool_args)
                        tool_result = result.content[0].text if result.content else "工具执行完成"

                        print(f"[成功] 工具 {tool_name} 执行成功")

                        # 添加 tool 回复到消息列表
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result
                        })

                    except Exception as e:
                        print(f"[错误] 工具 {tool_name} 执行失败: {str(e)}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"工具执行失败: {str(e)}"
                        })

                # 10. 第二次向模型发送请求（携带 tool 调用结果），得到最终输出
                final_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                final_output = final_response.choices[0].message.content

            else:
                # 如果没有工具调用，直接使用第一次模型回复的内容
                final_output = assistant_message.content

        except Exception as e:
            # 捕获整个处理流程中的异常
            print(f"[错误] 处理查询时发生错误: {str(e)}")
            final_output = f"抱歉，处理您的请求时发生了错误: {str(e)}"

        # 11. 将当前轮对话追加进记忆（memory 本身已限制最多20条）
        self.memory.chat_memory.add_user_message(query)
        self.memory.chat_memory.add_ai_message(final_output)

        # 12. 返回最终输出结果
        return final_output

    async def chat_loop(self):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n🤖 医疗助手已启动！当前时间：{current_time}")
        print("💡 输入 'quit' 退出对话")
        print("🕒 助手已配置最新时间信息，可以准确回答时间相关问题")
        print("-" * 50)

        while True:
            try:
                query = input("\n你: ").strip()
                if query.lower() == 'quit':
                    break

                if not query:
                    print("请输入有效的查询内容")
                    continue

                response = await self.process_query(query)
                print(f"\n[AI回复] {response}")

            except KeyboardInterrupt:
                print("\n[退出] 用户中断，正在退出...")
                break
            except Exception as e:
                print(f"\n⚠️ 发生错误: {str(e)}")

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    server_script_path = "D:\\PythonProject\\mcp-project\\server.py"
    session_id = "test_user_001"  # ✅ 你可以换成动态获取的用户名或 ID
    client = MCPClient(server_script_path, session_id=session_id)
    try:
        await client.connect_to_server(server_script_path)
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
