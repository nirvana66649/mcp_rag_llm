# custom_mongo_history.py
from typing import List
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory # 自定义聊天存储需要继承这个类
from pymongo import MongoClient


def serialize_message(message: BaseMessage) -> dict:
    if isinstance(message, HumanMessage):
        return {"type": "human", "data": {"content": message.content}}
    elif isinstance(message, AIMessage):
        return {"type": "ai", "data": {"content": message.content}}
    elif isinstance(message, SystemMessage):
        return {"type": "system", "data": {"content": message.content}}
    else:
        raise ValueError(f"Unsupported message type: {type(message)}")


def deserialize_message(data: dict) -> BaseMessage:
    type_ = data["type"]
    content = data["data"]["content"]
    if type_ == "human":
        return HumanMessage(content=content)
    elif type_ == "ai":
        return AIMessage(content=content)
    elif type_ == "system":
        return SystemMessage(content=content)
    else:
        raise ValueError(f"Unsupported message type: {type_}")


class CustomMongoChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        session_id: str,
        connection_string: str = "mongodb://localhost:27017",
        database_name: str = "chat_memory",
        collection_name: str = "mcp_memory",
    ):
        self.session_id = session_id
        self.client = MongoClient(connection_string)
        self.collection = self.client[database_name][collection_name]
        self._ensure_record()

    def _ensure_record(self):
        if not self.collection.find_one({"session_id": self.session_id}):
            self.collection.insert_one({"session_id": self.session_id, "messages": []})

    @property
    def messages(self) -> List[BaseMessage]:
        record = self.collection.find_one({"session_id": self.session_id})
        if not record:
            return []
        return [deserialize_message(m) for m in record.get("messages", [])]

    def add_message(self, message: BaseMessage) -> None:
        message_dict = serialize_message(message)
        self.collection.update_one(
            {"session_id": self.session_id},
            {"$push": {"messages": message_dict}},
        )

    def clear(self) -> None:
        self.collection.update_one(
            {"session_id": self.session_id},
            {"$set": {"messages": []}},
        )
