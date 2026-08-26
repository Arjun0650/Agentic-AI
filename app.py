import os
import json
import sqlite3
from datetime import date, timedelta

import chromadb
import uvicorn

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_core.tools import tool
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain.agents import create_agent


# ============================================================
# CONFIG
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is missing. Add it in Render Environment Variables."
    )

DB_PATH = "schedule.db"
CHROMA_PATH = "./chroma_schedule"
COLLECTION_NAME = "schedule_events"


# ============================================================
# SQLITE
# ============================================================

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            event_type TEXT DEFAULT 'task',
            date TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            location TEXT DEFAULT '',
            status TEXT DEFAULT 'scheduled',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()

    count = conn.execute(
        "SELECT COUNT(*) FROM events"
    ).fetchone()[0]

    if count == 0:
        today = date.today()

        templates = [
            {
                "title": "Team Meeting",
                "description": "Weekly project discussion with the team.",
                "event_type": "meeting",
                "start_time": "10:00",
                "end_time": "11:00",
                "location": "Conference Room",
            },
            {
                "title": "AI Workshop",
                "description": "Artificial Intelligence and Machine Learning workshop.",
                "event_type": "workshop",
                "start_time": "14:00",
                "end_time": "16:00",
                "location": "AI Lab",
            },
            {
                "title": "Project Development",
                "description": "Work on Agentic RAG Schedule Assistant.",
                "event_type": "task",
                "start_time": "09:00",
                "end_time": "11:00",
                "location": "Home",
            },
            {
                "title": "Doctor Appointment",
                "description": "Regular medical appointment.",
                "event_type": "appointment",
                "start_time": "16:00",
                "end_time": "17:00",
                "location": "City Hospital",
            },
            {
                "title": "Client Meeting",
                "description": "Discuss project requirements and progress.",
                "event_type": "meeting",
                "start_time": "15:00",
                "end_time": "16:00",
                "location": "Online",
            },
            {
                "title": "Python Practice",
                "description": "Practice Python and data structures.",
                "event_type": "task",
                "start_time": "18:00",
                "end_time": "19:00",
                "location": "Home",
            },
        ]

        for i in range(30):
            event_date = (today + timedelta(days=i)).isoformat()

            first = templates[i % len(templates)]

            conn.execute("""
                INSERT INTO events (
                    title,
                    description,
                    event_type,
                    date,
                    start_time,
                    end_time,
                    location
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                first["title"],
                first["description"],
                first["event_type"],
                event_date,
                first["start_time"],
                first["end_time"],
                first["location"],
            ))

            if i % 3 == 0:
                second = templates[(i + 2) % len(templates)]

                conn.execute("""
                    INSERT INTO events (
                        title,
                        description,
                        event_type,
                        date,
                        start_time,
                        end_time,
                        location
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    second["title"],
                    second["description"],
                    second["event_type"],
                    event_date,
                    second["start_time"],
                    second["end_time"],
                    second["location"],
                ))

        conn.commit()

    conn.close()


initialize_database()


# ============================================================
# DATABASE HELPERS
# ============================================================

def get_all_events():
    conn = get_connection()

    rows = conn.execute("""
        SELECT *
        FROM events
        WHERE status = 'scheduled'
        ORDER BY date, start_time
    """).fetchall()

    conn.close()

    return [dict(row) for row in rows]


def get_event_by_id(event_id):
    conn = get_connection()

    row = conn.execute("""
        SELECT *
        FROM events
        WHERE id = ?
    """, (event_id,)).fetchone()

    conn.close()

    return dict(row) if row else None


def get_events_by_date(event_date):
    conn = get_connection()

    rows = conn.execute("""
        SELECT *
        FROM events
        WHERE date = ?
        AND status = 'scheduled'
        ORDER BY start_time
    """, (event_date,)).fetchall()

    conn.close()

    return [dict(row) for row in rows]


# ============================================================
# EMBEDDINGS + CHROMADB
# ============================================================

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


def get_collection():
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME
    )


collection = get_collection()


def event_to_document(event):
    return f"""
Event ID: {event['id']}
Title: {event['title']}
Description: {event['description']}
Type: {event['event_type']}
Date: {event['date']}
Start Time: {event['start_time']}
End Time: {event['end_time']}
Location: {event['location']}
Status: {event['status']}
""".strip()


def rebuild_vector_store():
    global collection

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = get_collection()

    events = get_all_events()

    if not events:
        return

    documents = [event_to_document(e) for e in events]
    ids = [str(e["id"]) for e in events]

    metadatas = [
        {
            "event_id": int(e["id"]),
            "title": e["title"],
            "date": e["date"],
            "start_time": e["start_time"],
            "end_time": e["end_time"],
            "event_type": e["event_type"],
        }
        for e in events
    ]

    embeddings = embedding_model.embed_documents(documents)

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


rebuild_vector_store()


# ============================================================
# DATE HELPERS
# ============================================================

def next_weekday(weekday_number):
    today = date.today()

    days_ahead = (weekday_number - today.weekday()) % 7

    if days_ahead == 0:
        days_ahead = 7

    return today + timedelta(days=days_ahead)


def extract_date_from_query(query):
    query = query.lower()
    today = date.today()

    if "tomorrow" in query:
        return (today + timedelta(days=1)).isoformat()

    if "today" in query:
        return today.isoformat()

    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    for name, number in weekdays.items():
        if name in query:
            return next_weekday(number).isoformat()

    return None


# ============================================================
# RAG SEARCH
# ============================================================

def semantic_search(query, limit=8):
    count = collection.count()

    if count == 0:
        return []

    limit = min(limit, count)

    query_embedding = embedding_model.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=limit,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    output = []

    for document, metadata in zip(documents, metadatas):
        output.append({
            "document": document,
            "metadata": metadata,
        })

    return output


# ============================================================
# TOOL 1 — GET SCHEDULE
# ============================================================

@tool
def get_schedule(query: str) -> str:
    """
    Retrieve schedule information based on date, time,
    availability or natural-language query.
    """

    query_lower = query.lower()
    target_date = extract_date_from_query(query)

    if target_date:
        events = get_events_by_date(target_date)

        if "meeting" in query_lower:
            events = [
                e for e in events
                if e["event_type"] == "meeting"
            ]

        elif "workshop" in query_lower:
            events = [
                e for e in events
                if e["event_type"] == "workshop"
            ]

        elif "appointment" in query_lower:
            events = [
                e for e in events
                if e["event_type"] == "appointment"
            ]

        elif "task" in query_lower:
            events = [
                e for e in events
                if e["event_type"] == "task"
            ]

        if "morning" in query_lower:
            events = [
                e for e in events
                if "06:00" <= e["start_time"] < "12:00"
            ]

        elif "afternoon" in query_lower:
            events = [
                e for e in events
                if "12:00" <= e["start_time"] < "17:00"
            ]

        elif "evening" in query_lower:
            events = [
                e for e in events
                if e["start_time"] >= "17:00"
            ]

        return json.dumps({
            "success": True,
            "date": target_date,
            "events": events,
            "count": len(events),
        }, indent=2)

    results = semantic_search(query)

    return json.dumps({
        "success": True,
        "retrieval": "ChromaDB semantic RAG",
        "results": results,
    }, indent=2)


# ============================================================
# TOOL 2 — UPDATE SCHEDULE
# ============================================================

@tool
def update_schedule(
    action: str,
    title: str = "",
    description: str = "",
    event_type: str = "",
    event_date: str = "",
    start_time: str = "",
    end_time: str = "",
    location: str = "",
    event_id: int = 0,
) -> str:
    """
    Add, update, or delete schedule entries.
    """

    action = action.lower().strip()
    conn = get_connection()

    if action == "add":
        if not title or not event_date or not start_time or not end_time:
            conn.close()
            return "ERROR: title, event_date, start_time and end_time are required."

        if not event_type:
            event_type = "meeting"

        cursor = conn.execute("""
            INSERT INTO events (
                title,
                description,
                event_type,
                date,
                start_time,
                end_time,
                location,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'scheduled')
        """, (
            title,
            description,
            event_type,
            event_date,
            start_time,
            end_time,
            location,
        ))

        conn.commit()
        new_id = cursor.lastrowid
        conn.close()

        rebuild_vector_store()

        return json.dumps({
            "success": True,
            "action": "add",
            "event": get_event_by_id(new_id),
        }, indent=2)

    if action == "update":
        if not event_id:
            conn.close()
            return "ERROR: event_id is required. Use get_schedule first."

        fields = []
        values = []

        if title:
            fields.append("title = ?")
            values.append(title)

        if description:
            fields.append("description = ?")
            values.append(description)

        if event_type:
            fields.append("event_type = ?")
            values.append(event_type)

        if event_date:
            fields.append("date = ?")
            values.append(event_date)

        if start_time:
            fields.append("start_time = ?")
            values.append(start_time)

        if end_time:
            fields.append("end_time = ?")
            values.append(end_time)

        if location:
            fields.append("location = ?")
            values.append(location)

        if not fields:
            conn.close()
            return "ERROR: No update values provided."

        values.append(event_id)

        conn.execute(
            "UPDATE events SET "
            + ", ".join(fields)
            + " WHERE id = ?",
            values
        )

        conn.commit()
        conn.close()

        rebuild_vector_store()

        return json.dumps({
            "success": True,
            "action": "update",
            "event": get_event_by_id(event_id),
        }, indent=2)

    if action == "delete":
        if not event_id:
            conn.close()
            return "ERROR: event_id is required. Use get_schedule first."

        conn.execute("""
            UPDATE events
            SET status = 'cancelled'
            WHERE id = ?
        """, (event_id,))

        conn.commit()
        conn.close()

        rebuild_vector_store()

        return json.dumps({
            "success": True,
            "action": "delete",
            "event_id": event_id,
        }, indent=2)

    conn.close()

    return "ERROR: Invalid action. Use add, update or delete."


# ============================================================
# EXACTLY TWO TOOLS
# ============================================================

tools = [
    get_schedule,
    update_schedule,
]


# ============================================================
# GEMINI MODEL — IMPORTANT FIX
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)


# ============================================================
# AGENT
# ============================================================

SYSTEM_PROMPT = f"""
You are an intelligent Agentic RAG Schedule Assistant.

Today's date is {date.today().isoformat()}.

You manage the user's schedule for the next 30 days.

You have exactly two tools:

1. get_schedule
Use it to retrieve schedule information, check availability,
find meetings, workshops, tasks, appointments and search events.

2. update_schedule
Use it to add, update, move or delete schedule events.

Rules:

- Always call get_schedule before answering schedule questions.
- For moving, updating or deleting an existing event,
  first call get_schedule to find its event ID.
- Never invent events.
- Understand today, tomorrow and weekday names.
- Convert 3 PM to 15:00, 4 PM to 16:00, etc.
- If user says "3 PM for one hour", use 15:00 to 16:00.
- If no events exist during an availability period, say the user appears free.
- After changes, clearly confirm what changed.
- Return concise normal human-readable text.
"""


agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)


# ============================================================
# RESPONSE EXTRACTION
# ============================================================

def extract_text(content):
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)

            elif isinstance(block, dict):
                text = block.get("text")

                if text:
                    parts.append(str(text))

        return "\n".join(parts).strip()

    if content is None:
        return ""

    return str(content).strip()


def extract_final_response(result):
    if not isinstance(result, dict):
        return str(result)

    messages = result.get("messages", [])

    for message in reversed(messages):
        message_type = getattr(message, "type", "")
        content = getattr(message, "content", None)

        text = extract_text(content)

        if (
            message_type in ["ai", "assistant"]
            and text
            and text != "..."
        ):
            return text

    return "The agent processed the request but returned no readable response."


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Agentic RAG Schedule Assistant",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        result = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": request.message,
                }
            ]
        })

        return {
            "success": True,
            "response": extract_final_response(result),
        }

    except Exception as e:
        print("CHAT ERROR:", repr(e))

        return {
            "success": False,
            "response": f"Error: {str(e)}",
        }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "gemini-3.6-flash",
        "database": "SQLite",
        "vector_database": "ChromaDB",
        "rag": True,
        "tools": [
            "get_schedule",
            "update_schedule",
        ],
        "total_events": len(get_all_events()),
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">

<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Agentic RAG Schedule Assistant</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    background: linear-gradient(135deg,#06182f,#123d69);
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
}

.app {
    width: 94%;
    max-width: 900px;
    height: 90vh;
    background: white;
    border-radius: 22px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    box-shadow: 0 25px 80px rgba(0,0,0,.35);
}

.header {
    background: #071d38;
    color: white;
    padding: 25px 30px;
}

.header h1 {
    margin: 0;
    font-size: 28px;
}

.header p {
    margin: 8px 0 0;
    opacity: .75;
}

.status {
    display: inline-block;
    margin-top: 13px;
    background: rgba(255,255,255,.12);
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 12px;
}

.chat {
    flex: 1;
    overflow-y: auto;
    background: #f4f7fb;
    padding: 25px;
}

.message {
    max-width: 78%;
    margin-bottom: 15px;
    padding: 14px 18px;
    border-radius: 16px;
    line-height: 1.5;
    white-space: pre-wrap;
}

.bot {
    background: white;
    color: #172033;
    border: 1px solid #e2e7ee;
}

.user {
    margin-left: auto;
    background: #0b315c;
    color: white;
}

.input-area {
    display: flex;
    gap: 10px;
    padding: 18px;
    border-top: 1px solid #e1e5eb;
}

input {
    flex: 1;
    padding: 15px 17px;
    border: 1px solid #ccd3dd;
    border-radius: 12px;
    font-size: 16px;
    outline: none;
}

button {
    border: none;
    border-radius: 12px;
    padding: 0 25px;
    background: #0b315c;
    color: white;
    cursor: pointer;
    font-size: 15px;
}

button:disabled {
    opacity: .6;
}

</style>
</head>

<body>

<div class="app">

<div class="header">

<h1>Agentic RAG Schedule Assistant</h1>

<p>Gemini 3.6 Flash • ChromaDB RAG • SQLite</p>

<div class="status">● Agent Online</div>

</div>


<div class="chat" id="chat">

<div class="message bot">

Hello! I can manage your schedule for the next 30 days.

Try:
• What do I have scheduled tomorrow?
• Am I free Friday afternoon?
• Add a meeting tomorrow at 3 PM.
• Move my meeting from 3 PM to 4 PM.

</div>

</div>


<div class="input-area">

<input
    id="input"
    placeholder="Ask about your schedule..."
    onkeydown="if(event.key==='Enter') sendMessage()"
>

<button
    id="sendButton"
    onclick="sendMessage()"
>
Send
</button>

</div>

</div>


<script>

function addMessage(text, type) {

    const chat = document.getElementById("chat");

    const box = document.createElement("div");

    box.className = "message " + type;

    box.textContent = text;

    chat.appendChild(box);

    chat.scrollTop = chat.scrollHeight;

    return box;
}


async function sendMessage() {

    const input =
        document.getElementById("input");

    const button =
        document.getElementById("sendButton");

    const message =
        input.value.trim();

    if (!message) return;

    addMessage(message, "user");

    input.value = "";

    button.disabled = true;

    const thinking =
        addMessage("Thinking...", "bot");

    try {

        const response =
            await fetch("/chat", {

                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body:
                    JSON.stringify({
                        message: message
                    })
            });

        const data =
            await response.json();

        thinking.remove();

        addMessage(
            data.response,
            "bot"
        );

    }

    catch (error) {

        thinking.remove();

        addMessage(
            "Unable to contact the agent.",
            "bot"
        );

    }

    button.disabled = false;
}

</script>

</body>
</html>
"""


if __name__ == "__main__":
    port = int(
        os.environ.get("PORT", 8000)
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )
