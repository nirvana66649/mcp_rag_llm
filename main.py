import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from client import MCPClient


# 应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 初始化客户端
    server_script_path = "D:\\PythonProject\\mcp-project\\server.py"
    app.state.client = MCPClient(server_script_path)
    await app.state.client.__aenter__()
    yield
    # 清理资源
    await app.state.client.__aexit__(None, None, None)


app = FastAPI(lifespan=lifespan)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/uscagent/chat")
async def chat_endpoint(request_data: dict):
    """
    处理聊天请求的端点
    请求格式：
    {
        "query": "用户的问题",
        "session_id": "可选会话ID" 
    }
    """
    try:
        query = request_data.get("query", "").strip()
        session_id = request_data.get("session_id", str(uuid.uuid4()))

        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # 处理查询
        response = await app.state.client.process_query(query)

        return {
            "status": "success",
            "data": {
                "response": response,
                "session_id": session_id
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8081)
