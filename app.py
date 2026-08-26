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
# 1. CONFIGURATION
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is missing. "
        "Add it in Render Environment Variables."
    )

DB_PATH = "schedule.db"
CHROMA_PATH = "./chroma_schedule"
COLLECTION_NAME = "schedule_events"


# ============================================================
# 2. SQLITE DATABASE
# ============================================================

def get_connection():
    conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False
    )
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

        sample_events = [
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
                "description": "Discuss client requirements and progress.",
                "event_type": "meeting",
                "start_time": "15:00",
                "end_time": "16:00",
                "location": "Online",
            },
            {
                "title": "Python Practice",
                "description": "Practice Python programming and algorithms.",
                "event_type": "task",
                "start_time": "18:00",
                "end_time": "19:00",
                "location": "Home",
            },
        ]

        for day_number in range(30):

            event_date = (
                today + timedelta(days=day_number)
            ).isoformat()

            first = sample_events[
                day_number % len(sample_events)
            ]

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

            if day_number % 3 == 0:

                second = sample_events[
                    (day_number + 2)
                    % len(sample_events)
                ]

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
# 3. DATABASE FUNCTIONS
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

    return [
        dict(row)
        for row in rows
    ]


def get_event_by_id(event_id):

    conn = get_connection()

    row = conn.execute("""
        SELECT *
        FROM events
        WHERE id = ?
    """, (
        event_id,
    )).fetchone()

    conn.close()

    if row:
        return dict(row)

    return None


def get_events_by_date(event_date):

    conn = get_connection()

    rows = conn.execute("""
        SELECT *
        FROM events
        WHERE date = ?
        AND status = 'scheduled'
        ORDER BY start_time
    """, (
        event_date,
    )).fetchall()

    conn.close()

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# 4. GEMINI EMBEDDINGS
# ============================================================

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)


# ============================================================
# 5. CHROMADB
# ============================================================

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
        chroma_client.delete_collection(
            COLLECTION_NAME
        )
    except Exception:
        pass

    collection = get_collection()

    events = get_all_events()

    if not events:
        return

    documents = []
    ids = []
    metadatas = []

    for event in events:

        documents.append(
            event_to_document(event)
        )

        ids.append(
            str(event["id"])
        )

        metadatas.append({
            "event_id": int(event["id"]),
            "title": event["title"],
            "date": event["date"],
            "start_time": event["start_time"],
            "end_time": event["end_time"],
            "event_type": event["event_type"],
        })

    embeddings = (
        embedding_model.embed_documents(
            documents
        )
    )

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )


rebuild_vector_store()


# ============================================================
# 6. DATE HELPERS
# ============================================================

def get_next_weekday(weekday_number):

    today = date.today()

    days_ahead = (
        weekday_number
        - today.weekday()
    ) % 7

    if days_ahead == 0:
        days_ahead = 7

    return (
        today
        + timedelta(days=days_ahead)
    )


def extract_date_from_query(query):

    query = query.lower()

    today = date.today()

    if "tomorrow" in query:

        return (
            today
            + timedelta(days=1)
        ).isoformat()

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

    for weekday, number in weekdays.items():

        if weekday in query:

            return (
                get_next_weekday(
                    number
                )
            ).isoformat()

    return None


# ============================================================
# 7. RAG SEARCH
# ============================================================

def semantic_search(
    query,
    limit=8
):

    count = collection.count()

    if count == 0:
        return []

    limit = min(
        limit,
        count
    )

    query_embedding = (
        embedding_model.embed_query(
            query
        )
    )

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=limit
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    output = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        output.append({
            "document": document,
            "metadata": metadata,
        })

    return output


# ============================================================
# 8. TOOL 1 — GET SCHEDULE
# ============================================================

@tool
def get_schedule(query: str) -> str:
    """
    Retrieve relevant schedule information.

    Use this for:
    today, tomorrow, weekdays, meetings,
    workshops, tasks, appointments,
    availability checks and semantic schedule search.
    """

    query_lower = query.lower()

    target_date = (
        extract_date_from_query(
            query
        )
    )

    if target_date:

        events = get_events_by_date(
            target_date
        )

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
                if
                "06:00"
                <= e["start_time"]
                < "12:00"
            ]

        elif "afternoon" in query_lower:

            events = [
                e for e in events
                if
                "12:00"
                <= e["start_time"]
                < "17:00"
            ]

        elif "evening" in query_lower:

            events = [
                e for e in events
                if e["start_time"]
                >= "17:00"
            ]

        return json.dumps({
            "success": True,
            "query": query,
            "date": target_date,
            "count": len(events),
            "events": events,
        }, indent=2)

    rag_results = semantic_search(
        query
    )

    return json.dumps({
        "success": True,
        "query": query,
        "retrieval": "ChromaDB semantic RAG",
        "results": rag_results,
    }, indent=2)


# ============================================================
# 9. TOOL 2 — UPDATE SCHEDULE
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
    Add, update or delete schedule entries.

    action must be:
    add
    update
    delete
    """

    action = action.lower().strip()

    conn = get_connection()

    # --------------------------------------------------------
    # ADD
    # --------------------------------------------------------

    if action == "add":

        if not title:
            conn.close()
            return "ERROR: title is required."

        if not event_date:
            conn.close()
            return "ERROR: event_date is required."

        if not start_time:
            conn.close()
            return "ERROR: start_time is required."

        if not end_time:
            conn.close()
            return "ERROR: end_time is required."

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
            VALUES (
                ?, ?, ?, ?, ?, ?, ?,
                'scheduled'
            )
        """, (
            title,
            description,
            event_type,
            event_date,
            start_time,
            end_time,
            location
        ))

        conn.commit()

        new_id = cursor.lastrowid

        conn.close()

        rebuild_vector_store()

        event = get_event_by_id(
            new_id
        )

        return json.dumps({
            "success": True,
            "action": "add",
            "message": "Event added successfully.",
            "event": event,
        }, indent=2)

    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    elif action == "update":

        if not event_id:

            conn.close()

            return (
                "ERROR: event_id is required. "
                "Use get_schedule first."
            )

        existing = conn.execute("""
            SELECT *
            FROM events
            WHERE id = ?
            AND status = 'scheduled'
        """, (
            event_id,
        )).fetchone()

        if not existing:

            conn.close()

            return "ERROR: Event not found."

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

            return (
                "ERROR: No update values provided."
            )

        values.append(event_id)

        sql = (
            "UPDATE events SET "
            + ", ".join(fields)
            + " WHERE id = ?"
        )

        conn.execute(
            sql,
            values
        )

        conn.commit()
        conn.close()

        rebuild_vector_store()

        updated = get_event_by_id(
            event_id
        )

        return json.dumps({
            "success": True,
            "action": "update",
            "message": "Event updated successfully.",
            "event": updated,
        }, indent=2)

    # --------------------------------------------------------
    # DELETE
    # --------------------------------------------------------

    elif action == "delete":

        if not event_id:

            conn.close()

            return (
                "ERROR: event_id is required. "
                "Use get_schedule first."
            )

        conn.execute("""
            UPDATE events
            SET status = 'cancelled'
            WHERE id = ?
        """, (
            event_id,
        ))

        conn.commit()
        conn.close()

        rebuild_vector_store()

        return json.dumps({
            "success": True,
            "action": "delete",
            "message":
                f"Event {event_id} deleted successfully."
        }, indent=2)

    conn.close()

    return (
        "ERROR: Invalid action. "
        "Use add, update or delete."
    )


# ============================================================
# 10. EXACTLY TWO TOOLS
# ============================================================

tools = [
    get_schedule,
    update_schedule
]


# ============================================================
# 11. GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# 12. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are an intelligent Agentic RAG Schedule Assistant.

Today's date is:

{date.today().isoformat()}

You manage the user's schedule for the next 30 days.

You have EXACTLY TWO tools:

1. get_schedule

Use get_schedule whenever the user asks:

- What do I have scheduled?
- What do I have tomorrow?
- What meetings do I have?
- What workshops do I have?
- Do I have an appointment?
- Am I free?
- Am I available?
- Search my schedule.
- Find an event.

2. update_schedule

Use update_schedule whenever the user asks:

- Add an event.
- Add a meeting.
- Add an appointment.
- Add a workshop.
- Add a task.
- Move an event.
- Reschedule an event.
- Update an event.
- Delete an event.

IMPORTANT RULES:

If the user wants to move, update or delete an existing
event and you do not know the event ID:

FIRST call get_schedule.

Identify the correct event.

THEN call update_schedule.

Never invent schedule information.

Always retrieve schedule information before answering
questions about the schedule.

Understand:

today
tomorrow
Monday
Tuesday
Wednesday
Thursday
Friday
Saturday
Sunday

Convert times:

3 PM = 15:00
4 PM = 16:00
10 AM = 10:00

If the user says:

"3 PM for one hour"

use:

start_time = 15:00
end_time = 16:00

When checking availability,
use get_schedule first.

If there are no events in the requested period,
say that the user appears free.

After adding, updating or deleting,
clearly confirm what changed.

Return a normal readable response.
"""


# ============================================================
# 13. CREATE AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT
)


# ============================================================
# 14. RESPONSE EXTRACTION
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

    messages = result.get(
        "messages",
        []
    )

    for message in reversed(messages):

        message_type = getattr(
            message,
            "type",
            ""
        )

        content = getattr(
            message,
            "content",
            None
        )

        text = extract_text(
            content
        )

        if (
            message_type in [
                "ai",
                "assistant"
            ]
            and text
            and text != "..."
        ):
            return text

    for message in reversed(messages):

        content = getattr(
            message,
            "content",
            None
        )

        text = extract_text(
            content
        )

        if text and text != "...":
            return text

    return (
        "The request was processed, "
        "but the model returned no readable response."
    )


# ============================================================
# 15. FASTAPI
# ============================================================

app = FastAPI(
    title="Agentic RAG Schedule Assistant",
    description=(
        "AI schedule management using "
        "Gemini, ChromaDB and SQLite."
    ),
    version="1.0.0"
)


# ============================================================
# 16. REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    message: str


# ============================================================
# 17. CHAT ENDPOINT
# ============================================================

@app.post("/chat")
def chat(request: ChatRequest):

    try:

        result = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": request.message
                }
            ]
        })

        response_text = (
            extract_final_response(
                result
            )
        )

        return {
            "success": True,
            "response": response_text
        }

    except Exception as e:

        print(
            "CHAT ERROR:",
            repr(e)
        )

        return {
            "success": False,
            "response":
                f"Error: {str(e)}"
        }


# ============================================================
# 18. HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "application":
            "Agentic RAG Schedule Assistant",
        "database": "SQLite",
        "vector_database": "ChromaDB",
        "rag": True,
        "tools": [
            "get_schedule",
            "update_schedule"
        ],
        "total_events":
            len(get_all_events())
    }


# ============================================================
# 19. WEB INTERFACE
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home():

    return """
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>
Agentic RAG Schedule Assistant
</title>


<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background:
        linear-gradient(
            135deg,
            #06182f,
            #123d69
        );

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

    box-shadow:
        0 25px 80px
        rgba(0,0,0,.35);
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

    margin:
        8px 0 0;

    opacity: .75;
}


.status {

    display: inline-block;

    margin-top: 13px;

    background:
        rgba(
            255,
            255,
            255,
            .12
        );

    padding:
        6px 12px;

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

    padding:
        14px 18px;

    border-radius: 16px;

    line-height: 1.5;

    white-space: pre-wrap;
}


.bot {

    background: white;

    color: #172033;

    border:
        1px solid #e2e7ee;
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

    border-top:
        1px solid #e1e5eb;
}


input {

    flex: 1;

    padding:
        15px 17px;

    border:
        1px solid #ccd3dd;

    border-radius: 12px;

    font-size: 16px;

    outline: none;
}


input:focus {

    border-color: #154f88;
}


button {

    border: none;

    border-radius: 12px;

    padding:
        0 25px;

    background: #0b315c;

    color: white;

    cursor: pointer;

    font-size: 15px;
}


button:hover {

    background: #164f87;
}


button:disabled {

    opacity: .6;

    cursor: not-allowed;
}


@media(max-width:600px) {

    .app {

        width: 100%;

        height: 100vh;

        border-radius: 0;
    }


    .message {

        max-width: 90%;
    }

}

</style>

</head>


<body>


<div class="app">


<div class="header">

<h1>
Agentic RAG Schedule Assistant
</h1>

<p>
Gemini • ChromaDB RAG • SQLite
</p>

<div class="status">
● Agent Online
</div>

</div>


<div
    class="chat"
    id="chat"
>

<div class="message bot">

Hello! I can manage your schedule for the next 30 days.

Try asking:

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
    onkeydown="handleEnter(event)"
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

function handleEnter(event) {

    if (
        event.key === "Enter"
    ) {

        sendMessage();

    }

}


function addMessage(
    text,
    type
) {

    const chat =
        document.getElementById(
            "chat"
        );

    const message =
        document.createElement(
            "div"
        );

    message.className =
        "message " + type;

    message.textContent =
        text;

    chat.appendChild(
        message
    );

    chat.scrollTop =
        chat.scrollHeight;

    return message;
}


async function sendMessage() {

    const input =
        document.getElementById(
            "input"
        );

    const button =
        document.getElementById(
            "sendButton"
        );

    const message =
        input.value.trim();

    if (!message) {
        return;
    }


    addMessage(
        message,
        "user"
    );


    input.value = "";

    button.disabled = true;


    const thinking =
        addMessage(
            "Thinking...",
            "bot"
        );


    try {

        const response =
            await fetch(
                "/chat",
                {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(
                            {
                                message:
                                    message
                            }
                        )
                }
            );


        const data =
            await response.json();


        thinking.remove();


        if (
            data.success
        ) {

            addMessage(
                data.response,
                "bot"
            );

        }

        else {

            addMessage(
                data.response ||
                "Something went wrong.",
                "bot"
            );

        }

    }

    catch (error) {

        thinking.remove();

        addMessage(
            "Unable to contact the agent.",
            "bot"
        );

    }


    button.disabled = false;

    input.focus();

}

</script>


</body>

</html>
"""


# ============================================================
# 20. RENDER / LOCAL START
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
