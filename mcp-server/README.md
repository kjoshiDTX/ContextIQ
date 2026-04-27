# ContextIQ MCP Server

Connects Claude Desktop to your ContextIQ knowledge graph. When you chat with Claude, you (or Claude) can call tools to save insights, retrieve personal context, create tasks, and add journal entries — all flowing into your graph automatically.

## Setup

### 1. Install dependencies

```bash
cd mcp-server
pip install -r requirements.txt
```

### 2. Get your MCP API key

While ContextIQ is running, go to **Query** and ask:

> "What is my MCP API key?"

Or fetch it directly:

```bash
# Replace TOKEN with your Supabase session token
curl -H "Authorization: Bearer TOKEN" http://localhost:8000/api/mcp/key
```

### 3. Add to Claude Desktop config

Open `~/Library/Application Support/Claude/claude_desktop_config.json` and add:

```json
{
  "mcpServers": {
    "contextiq": {
      "command": "python",
      "args": ["/absolute/path/to/ContextIQ/mcp-server/server.py"],
      "env": {
        "CONTEXTIQ_API_URL": "http://localhost:8000/api",
        "CONTEXTIQ_MCP_KEY": "ciq_YOUR_KEY_HERE"
      }
    }
  }
}
```

Restart Claude Desktop. You should see "contextiq" in the MCP servers list.

## Available Tools

| Tool | When Claude uses it |
|------|---------------------|
| `save_insight` | You ask Claude to save something, or Claude identifies a key takeaway |
| `get_user_context` | You reference something personal; Claude retrieves your graph context |
| `save_task` | You mention something you need to do |
| `add_to_journal` | You reflect on a question or decision with Claude |

## Example prompts

```
"Save the key insight from this conversation to my graph."

"What do my notes say about my goals for this quarter?"

"Add a task: follow up with the design team about the new onboarding flow."

"I just realized I work best in 90-minute deep work blocks — add this to my journal."
```

## How it works

1. Claude calls a tool (e.g. `save_insight`)
2. MCP server sends the content to ContextIQ backend via `/api/mcp/*` endpoints
3. Backend runs triplet extraction (Gemini) → creates `ConversationConcept` nodes in Neo4j
4. Content is embedded into `user_content_chunks` for unified vector search
5. Open ContextIQ → Knowledge Graph to see your new purple nodes
