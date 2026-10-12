"""Run with: python3 -m uvicorn api:app --host 127.0.0.1 --port 8000."""

import importlib
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=2000)
    history: List[Message] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def check_history(self):
        if len(self.history) % 2:
            raise ValueError("History must contain complete user/assistant exchanges.")
        for position, message in enumerate(self.history):
            expected = "user" if position % 2 == 0 else "assistant"
            if message.role != expected:
                raise ValueError("History must alternate user and assistant messages.")
        return self


class ChatResponse(BaseModel):
    answer: str
    history: List[Message]
    sources: List[Dict[str, Any]]


def load_engine():
    # Importing chat loads the embedding model, index, metadata, and Qwen.
    # Delay it until startup so schema inspection/tests need no model weights.
    return importlib.import_module("chat")


def create_app(engine_loader=load_engine):
    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = await run_in_threadpool(engine_loader)
        app.state.generation_lock = Lock()
        try:
            yield
        finally:
            app.state.engine = None

    app = FastAPI(title="PIU Conversational Chat API", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request):
        engine = request.app.state.engine
        return {"status": "ready", "device": str(engine.model.device)}

    @app.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest, request: Request):
        lock = request.app.state.generation_lock
        if not lock.acquire(blocking=False):
            raise HTTPException(
                status_code=503,
                detail="The chatbot is busy. Please retry shortly.",
                headers={"Retry-After": "5"},
            )
        try:
            engine = request.app.state.engine
            history = [message.model_dump() for message in payload.history]
            answer, sources = engine.answer_question(payload.question, history)
            history.extend([
                {"role": "user", "content": payload.question},
                {"role": "assistant", "content": answer},
            ])
            return ChatResponse(
                answer=answer,
                history=history[-2 * min(engine.MAX_HISTORY_TURNS, 4):],
                sources=sources,
            )
        finally:
            lock.release()

    return app


app = create_app()
