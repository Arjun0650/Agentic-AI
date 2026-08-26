import os
import json
import sqlite3
from datetime import date, timedelta

import chromadb
import uvicorn

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from chromadb.utils import embedding_functions
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent


# ============================================================
# CONFIG
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
# SQLITE DATABASE
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

            event_date = (
                today + timedelta(days=i)
            ).isoformat()

            first = templates[
                i % len(templates)
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

            if i % 3 == 0:

                second = templates[
                    (i + 2)
                    % len(templates)
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

    return (
        dict(row)
        if row
        else None
    )


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
# CHROMADB — LOCAL EMBEDDINGS
# ============================================================

local_embedding_function = (
    embedding_functions.DefaultEmbeddingFunction()
)

chroma_client = (
    chromadb.PersistentClient(
        path=CHROMA_PATH
    )
)


def get_collection():

    return (
        chroma_client
        .get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=
                local_embedding_function
        )
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

    documents = [
        event_to_document(event)
        for event in events
    ]

    ids = [
        str(event["id"])
        for event in events
    ]

    metadatas = [
        {
            "event_id":
                int(event["id"]),

            "title":
                event["title"],

            "date":
                event["date"],

            "start_time":
                event["start_time"],

            "end_time":
                event["end_time"],

            "event_type":
                event["event_type"],
        }
        for event in events
    ]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )


rebuild_vector_store()


# ============================================================
# DATE HELPERS
# ============================================================

def next_weekday(
    weekday_number
):

    today = date.today()

    days_ahead = (
        weekday_number
        - today.weekday()
    ) % 7

    if days_ahead == 0:
        days_ahead = 7

    return (
        today
        + timedelta(
            days=days_ahead
        )
    )


def extract_date_from_query(
    query
):

    query = query.lower()

    today = date.today()

    if "tomorrow" in query:

        return (
            today
            + timedelta(days=1)
        ).isoformat()

    if "today" in query:

        return (
            today.isoformat()
        )

    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    for name, number in (
        weekdays.items()
    ):

        if name in query:

            return (
                next_weekday(
                    number
                )
            ).isoformat()

    return None


# ============================================================
# RAG SEARCH
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

    results = collection.query(
        query_texts=[query],
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
            "document":
                document,

            "metadata":
                metadata,
        })

    return output


# ============================================================
# TOOL 1 — GET SCHEDULE
# ============================================================

@tool
def get_schedule(
    query: str
) -> str:
    """
    Retrieve schedule information based on date,
    time, availability, event type, or natural-language query.
    """

    query_lower = query.lower()

    target_date = (
        extract_date_from_query(
            query
        )
    )

    # Exact date retrieval
    if target_date:

        events = get_events_by_date(
            target_date
        )

        if "meeting" in query_lower:

            events = [
                e for e in events
                if e["event_type"]
                == "meeting"
            ]

        elif "workshop" in query_lower:

            events = [
                e for e in events
                if e["event_type"]
                == "workshop"
            ]

        elif "appointment" in query_lower:

            events = [
                e for e in events
                if e["event_type"]
                == "appointment"
            ]

        elif "task" in query_lower:

            events = [
                e for e in events
                if e["event_type"]
                == "task"
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
            "success":
                True,

            "date":
                target_date,

            "count":
                len(events),

            "events":
                events,
        }, indent=2)

    # Semantic RAG retrieval
    rag_results = (
        semantic_search(
            query
        )
    )

    return json.dumps({
        "success":
            True,

        "retrieval":
            "ChromaDB semantic RAG",

        "results":
            rag_results,
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

    action must be:
    add
    update
    delete
    """

    action = action.lower().strip()

    conn = get_connection()

    # ADD
    if action == "add":

        if not title:

            conn.close()

            return (
                "ERROR: title is required."
            )

        if not event_date:

            conn.close()

            return (
                "ERROR: event_date is required."
            )

        if not start_time:

            conn.close()

            return (
                "ERROR: start_time is required."
            )

        if not end_time:

            conn.close()

            return (
                "ERROR: end_time is required."
            )

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
            location,
        ))

        conn.commit()

        new_id = (
            cursor.lastrowid
        )

        conn.close()

        rebuild_vector_store()

        return json.dumps({
            "success":
                True,

            "action":
                "add",

            "message":
                "Event added successfully.",

            "event":
                get_event_by_id(
                    new_id
                ),
        }, indent=2)

    # UPDATE
    if action == "update":

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

            return (
                "ERROR: Event not found."
            )

        fields = []
        values = []

        if title:

            fields.append(
                "title = ?"
            )

            values.append(
                title
            )

        if description:

            fields.append(
                "description = ?"
            )

            values.append(
                description
            )

        if event_type:

            fields.append(
                "event_type = ?"
            )

            values.append(
                event_type
            )

        if event_date:

            fields.append(
                "date = ?"
            )

            values.append(
                event_date
            )

        if start_time:

            fields.append(
                "start_time = ?"
            )

            values.append(
                start_time
            )

        if end_time:

            fields.append(
                "end_time = ?"
            )

            values.append(
                end_time
            )

        if location:

            fields.append(
                "location = ?"
            )

            values.append(
                location
            )

        if not fields:

            conn.close()

            return (
                "ERROR: No update values provided."
            )

        values.append(
            event_id
        )

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
            "success":
                True,

            "action":
                "update",

            "message":
                "Event updated successfully.",

            "event":
                get_event_by_id(
                    event_id
                ),
        }, indent=2)

    # DELETE
    if action == "delete":

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
            "success":
                True,

            "action":
                "delete",

            "message":
                f"Event {event_id} deleted successfully."
        }, indent=2)

    conn.close()

    return (
        "ERROR: Invalid action. "
        "Use add, update, or delete."
    )


# ============================================================
# EXACTLY TWO REQUIRED TOOLS
# ============================================================

tools = [
    get_schedule,
    update_schedule
]


# ============================================================
# GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=
        GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# AGENT PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are an intelligent Agentic RAG Schedule Assistant.

Today's date is:

{date.today().isoformat()}

You manage the user's schedule for the next 30 days.

You have EXACTLY TWO tools.

TOOL 1:
get_schedule

Use get_schedule when the user asks:

- What do I have scheduled?
- What do I have tomorrow?
- What meetings do I have?
- What workshops do I have?
- What appointments do I have?
- What tasks do I have?
- Am I free?
- Am I available?
- Find an event.
- Search my schedule.

TOOL 2:
update_schedule

Use update_schedule when the user asks:

- Add an event.
- Add a meeting.
- Add a workshop.
- Add an appointment.
- Add a task.
- Move an event.
- Reschedule an event.
- Update an event.
- Delete an event.

IMPORTANT:

For updates, moves, and deletes:

FIRST call get_schedule.

Find the correct event and its event ID.

THEN call update_schedule.

Never invent schedule events.

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

For availability questions:

FIRST use get_schedule.

If there are no events during that period,
say that the user appears free.

After adding, updating, or deleting an event,
clearly confirm what happened.

Return a normal concise human-readable response.
"""


# ============================================================
# CREATE AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=
        SYSTEM_PROMPT
)


# ============================================================
# RESPONSE EXTRACTION
# ============================================================

def extract_text(content):

    if isinstance(
        content,
        str
    ):
        return content.strip()

    if isinstance(
        content,
        list
    ):

        parts = []

        for block in content:

            if isinstance(
                block,
                str
            ):

                parts.append(
                    block
                )

            elif isinstance(
                block,
                dict
            ):

                text = block.get(
                    "text"
                )

                if text:

                    parts.append(
                        str(text)
                    )

        return "\n".join(
            parts
        ).strip()

    if content is None:

        return ""

    return str(
        content
    ).strip()


def extract_final_response(
    result
):

    if not isinstance(
        result,
        dict
    ):

        return str(result)

    messages = result.get(
        "messages",
        []
    )

    for message in reversed(
        messages
    ):

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
            message_type
            in [
                "ai",
                "assistant"
            ]
            and text
            and text != "..."
        ):

            return text

    return (
        "The agent processed the request, "
        "but no readable response was returned."
    )


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title=
        "Agentic RAG Schedule Assistant",
    version=
        "1.0.0"
)


class ChatRequest(
    BaseModel
):
    message: str


# ============================================================
# CHAT API
# ============================================================

@app.post("/chat")
def chat(
    request: ChatRequest
):

    try:

        result = agent.invoke({
            "messages": [
                {
                    "role":
                        "user",

                    "content":
                        request.message
                }
            ]
        })

        return {
            "success":
                True,

            "response":
                extract_final_response(
                    result
                )
        }

    except Exception as e:

        print(
            "CHAT ERROR:",
            repr(e)
        )

        return {
            "success":
                False,

            "response":
                f"Error: {str(e)}"
        }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status":
            "healthy",

        "model":
            "gemini-3.6-flash",

        "database":
            "SQLite",

        "vector_database":
            "ChromaDB",

        "embedding":
            "ChromaDB local embedding",

        "rag":
            True,

        "tools": [
            "get_schedule",
            "update_schedule"
        ],

        "total_events":
            len(
                get_all_events()
            )
    }


# ============================================================
# WEB PAGE
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
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

    align-items: center;

    justify-content: center;
}


.app {

    width: 94%;

    max-width: 950px;

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

    padding:
        25px 30px;
}


.header h1 {

    margin: 0;

    font-size: 29px;
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

    max-width: 80%;

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


@media(
    max-width: 600px
) {

    .app {

        width: 100%;

        height: 100vh;

        border-radius: 0;
    }


    .message {

        max-width: 92%;
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
Gemini 3.6 Flash • ChromaDB RAG • SQLite
</p>

<div class="status">
● Agent Online
</div>

</div>


<div
    id="chat"
    class="chat"
>


<div class="message bot">

Hello! I can manage your schedule for the next 30 days.

Try asking:

• What do I have scheduled tomorrow?
• Am I free Friday afternoon?
• What workshops do I have coming up?
• Add a project meeting tomorrow at 3 PM for one hour.
• Move my project meeting from 3 PM to 4 PM.

</div>


</div>


<div class="input-area">


<input
    id="input"
    placeholder="Ask about your schedule..."
    onkeydown="
        if(event.key === 'Enter') {
            sendMessage();
        }
    "
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


function addMessage(
    text,
    type
) {

    const chat =
        document.getElementById(
            "chat"
        );

    const element =
        document.createElement(
            "div"
        );

    element.className =
        "message " + type;

    element.textContent =
        text;

    chat.appendChild(
        element
    );

    chat.scrollTop =
        chat.scrollHeight;

    return element;
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


    input.value =
        "";


    button.disabled =
        true;


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

                    method:
                        "POST",

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


    button.disabled =
        false;

    input.focus();
}


</script>


</body>

</html>
"""


# ============================================================
# RENDER / LOCAL START
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
        host=
            "0.0.0.0",
        port=
            port
    )
