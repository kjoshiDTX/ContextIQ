#!/usr/bin/env python3
"""ContextIQ MCP Server — connects Claude Desktop to your personal knowledge graph.

Exposes 4 tools Claude can call during conversations:
  - save_insight: Save an idea/thought to your graph + vector store
  - get_user_context: Pull relevant personal context from your graph
  - save_task: Create a task in ContextIQ
  - add_to_journal: Save a Q&A pair to your journal

Setup:
    1. pip install -r requirements.txt
    2. Set env vars: CONTEXTIQ_API_URL, CONTEXTIQ_MCP_KEY
    3. Add to Claude Desktop config (see README.md)
"""

import asyncio
import os
import sys
import json

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, CallToolResult

# ─── Config ──────────────────────────────────────────────────────

API_URL = os.environ.get("CONTEXTIQ_API_URL", "http://localhost:8000/api")
MCP_KEY = os.environ.get("CONTEXTIQ_MCP_KEY", "")

if not MCP_KEY:
    print("ERROR: CONTEXTIQ_MCP_KEY environment variable is not set.", file=sys.stderr)
    print("Get your key from ContextIQ → Settings or run: GET /api/mcp/key", file=sys.stderr)
    sys.exit(1)

HEADERS = {
    "X-MCP-Key": MCP_KEY,
    "Content-Type": "application/json",
}

# ─── MCP Server ──────────────────────────────────────────────────

app = Server("contextiq")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="save_insight",
            description=(
                "Save an insight, idea, or key takeaway from this conversation to the user's "
                "ContextIQ knowledge graph. Use this when the user shares something meaningful "
                "they want to remember — a decision made, an idea explored, a realization reached. "
                "The insight is processed in the background: concepts are extracted and mapped "
                "onto their graph, and it becomes searchable in future sessions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The insight or idea to save. Should be a coherent paragraph or a few sentences capturing the key thought.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional short title for the insight (max 80 chars).",
                    },
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="get_user_context",
            description=(
                "Retrieve relevant context from the user's personal knowledge graph about a topic. "
                "Use this at the start of a conversation or when the user references something "
                "personal (goals, past decisions, projects, emotions, relationships). "
                "Returns synthesized context from their documents, journal entries, "
                "past conversation insights, and tasks."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic or question to look up context for.",
                    },
                },
                "required": ["topic"],
            },
        ),
        Tool(
            name="save_task",
            description=(
                "Create a task or to-do item in the user's ContextIQ task list. "
                "Use when the user mentions something they want to do, "
                "a follow-up action, or an action item from this conversation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short, clear task title.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional longer description or context for the task.",
                    },
                    "due_date": {
                        "type": "string",
                        "description": "Optional due date in YYYY-MM-DD format.",
                    },
                },
                "required": ["title"],
            },
        ),
        Tool(
            name="add_to_journal",
            description=(
                "Add a reflective Q&A entry to the user's ContextIQ journal. "
                "Use when the user has just worked through a personal question, "
                "made a decision, or reflected on something meaningful. "
                "This feeds into their personal knowledge graph."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The reflective question that was explored.",
                    },
                    "answer": {
                        "type": "string",
                        "description": "The user's answer or reflection.",
                    },
                },
                "required": ["question", "answer"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    async with httpx.AsyncClient(timeout=30.0) as client:
        if name == "save_insight":
            resp = await client.post(
                f"{API_URL}/mcp/insight",
                headers=HEADERS,
                json={
                    "content": arguments["content"],
                    "title": arguments.get("title"),
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Saved to your knowledge graph (session: {data.get('session_id', '?')}). "
                             "Concepts will be extracted and connected to your existing graph in the background.",
                    )]
                )
            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Failed to save insight: {resp.text}")]
                )

        elif name == "get_user_context":
            resp = await client.post(
                f"{API_URL}/mcp/context",
                headers=HEADERS,
                json={"topic": arguments["topic"]},
            )
            if resp.status_code == 200:
                data = resp.json()
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Here's what I found in your knowledge graph about '{arguments['topic']}':\n\n"
                             + data.get("context", "No relevant context found."),
                    )]
                )
            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Failed to fetch context: {resp.text}")]
                )

        elif name == "save_task":
            resp = await client.post(
                f"{API_URL}/mcp/task",
                headers=HEADERS,
                json={
                    "title": arguments["title"],
                    "description": arguments.get("description"),
                    "due_date": arguments.get("due_date"),
                },
            )
            if resp.status_code == 200:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Task '{arguments['title']}' added to your ContextIQ task list.",
                    )]
                )
            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Failed to create task: {resp.text}")]
                )

        elif name == "add_to_journal":
            resp = await client.post(
                f"{API_URL}/mcp/journal",
                headers=HEADERS,
                json={
                    "content": f"Q: {arguments['question']}\nA: {arguments['answer']}",
                    "title": arguments["question"][:80],
                    "session_data": [
                        {"question": arguments["question"], "answer": arguments["answer"]}
                    ],
                },
            )
            if resp.status_code == 200:
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text="Added to your journal. Concepts will be extracted and mapped to your knowledge graph.",
                    )]
                )
            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Failed to add journal entry: {resp.text}")]
                )

        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {name}")]
            )


# ─── Entry Point ─────────────────────────────────────────────────

async def main():
    async with stdio_server() as streams:
        await app.run(*streams, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
