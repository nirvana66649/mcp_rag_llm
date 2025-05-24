import asyncio
import os
import json
from datetime import datetime
import re
from openai import OpenAI
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
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


class MCPClient:
    def __init__(self,server_script_path: str):
        self.exit_stack = AsyncExitStack()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("MODEL")
        if not self.openai_api_key:
            raise ValueError("❌ 未找到 OpenAI API Key，请在 .env 文件中设置 OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.openai_api_key, base_url=self.base_url)

        self.mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        self.mongo_db = self.mongo_client["chat_memory_db"]
        self.collection_name = "mcp_memory"
        self.memory_id = str(uuid.uuid4())

        # ✅ 新用法：langchain_mongodb 中的 MongoDBChatMessageHistory
        chat_history = MongoDBChatMessageHistory(
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
        self.server_script_path = server_script_path  # 添加服务器路径参数

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

        history_messages = self.memory.chat_memory.messages
        messages = safe_messages_to_dict(history_messages)

        if not messages:
            messages.append({"role": "system", "content": "你是一个智能助手。"})

        messages.append({"role": "user", "content": query})

        print(f"\n[查询] 用户查询: {query}")
        print(f"[工具] 可用工具: {[tool['function']['name'] for tool in available_tools]}")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=available_tools,
                tool_choice="auto"
            )

            assistant_message = response.choices[0].message
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": assistant_message.tool_calls
            })

            if assistant_message.tool_calls:
                print(f"\n[工具调用] 模型决定调用 {len(assistant_message.tool_calls)} 个工具")

                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}

                    print(f"[调用] 工具名称: {tool_name}")
                    print(f"[参数] {tool_args}")

                    try:
                        result = await self.session.call_tool(tool_name, tool_args)
                        tool_result = result.content[0].text if result.content else "工具执行完成"

                        print(f"[成功] 工具 {tool_name} 执行成功")

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

                final_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                final_output = final_response.choices[0].message.content
            else:
                final_output = assistant_message.content

        except Exception as e:
            print(f"[错误] 处理查询时发生错误: {str(e)}")
            final_output = f"抱歉，处理您的请求时发生了错误: {str(e)}"

        self.memory.chat_memory.add_user_message(query)
        self.memory.chat_memory.add_ai_message(final_output)

        self.save_conversation_log(query, final_output)

        return final_output

    def save_conversation_log(self, query: str, response: str):
        def clean_filename(text: str) -> str:
            text = text.strip()
            text = re.sub(r'[\\/:*?"<>|]', '', text)
            return text[:50]

        safe_filename = clean_filename(query)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{safe_filename}_{timestamp}.txt"
        output_dir = "./llm_outputs"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"🗣 用户提问：{query}\n\n")
            f.write(f"🤖 模型回复：\n{response}\n")

        print(f"📄 对话记录已保存为：{file_path}")

    async def chat_loop(self):
        print("\n[启动] MCP 客户端已启动！输入 'quit' 退出")
        print("[提示] 你可以尝试以下命令：")
        print("   - 搜索小米汽车的新闻")
        print("   - 分析一下这段文字的情感：[你的文字]")
        print("   - 查询数据库中的预约记录")
        print("   - 修改用户张三的预约时间为明天下午2点")
        print("   - 发送邮件到 example@email.com")
        print("   - 搜索关于北京协和医院的信息")
        print("   - 清理文件夹文件")

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
    client = MCPClient()
    try:
        await client.connect_to_server(server_script_path)
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())


