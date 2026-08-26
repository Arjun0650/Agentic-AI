import os
import json
import sqlite3
from datetime import date, timedelta

import chromadb
import uvicorn

from fastapi import FastAPI
from langserve import add_routes

from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
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
        "Add it in Render > Environment Variables."
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

    # --------------------------------------------------------
    # Generate sample schedule for next 30 days
    # --------------------------------------------------------

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
                "description": "Discuss client requirements and project progress.",
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

            item = sample_events[
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
                item["title"],
                item["description"],
                item["event_type"],
                event_date,
                item["start_time"],
                item["end_time"],
                item["location"],
            ))

            # Add second event every third day
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


# Create RAG index when app starts
rebuild_vector_store()


# ============================================================
# 6. DATE HELPERS
# ============================================================

def get_next_weekday(
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
        + timedelta(days=days_ahead)
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

    for weekday, number in (
        weekdays.items()
    ):

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

    collection_count = (
        collection.count()
    )

    if collection_count == 0:
        return []

    limit = min(
        limit,
        collection_count
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

            "document":
                document,

            "metadata":
                metadata,
        })

    return output


# ============================================================
# 8. TOOL 1 — get_schedule
# ============================================================

@tool
def get_schedule(
    query: str
) -> str:
    """
    Retrieve relevant schedule information.

    Use this for:
    - today
    - tomorrow
    - weekdays
    - meetings
    - workshops
    - tasks
    - appointments
    - availability
    - semantic schedule search
    """

    query_lower = query.lower()

    target_date = (
        extract_date_from_query(
            query
        )
    )

    # --------------------------------------------------------
    # Exact date search
    # --------------------------------------------------------

    if target_date:

        events = get_events_by_date(
            target_date
        )

        # Filter by type

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

        # ----------------------------------------------------
        # Time period filtering
        # ----------------------------------------------------

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

            "query":
                query,

            "date":
                target_date,

            "count":
                len(events),

            "events":
                events,

        }, indent=2)

    # --------------------------------------------------------
    # RAG search
    # --------------------------------------------------------

    rag_results = semantic_search(
        query
    )

    return json.dumps({

        "success": True,

        "query":
            query,

        "retrieval":
            "ChromaDB semantic RAG",

        "results":
            rag_results,

    }, indent=2)


# ============================================================
# 9. TOOL 2 — update_schedule
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

    # ========================================================
    # ADD
    # ========================================================

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
            location
        ))

        conn.commit()

        new_id = (
            cursor.lastrowid
        )

        conn.close()

        rebuild_vector_store()

        event = get_event_by_id(
            new_id
        )

        return json.dumps({

            "success": True,

            "action":
                "add",

            "message":
                "Event added successfully.",

            "event":
                event,

        }, indent=2)

    # ========================================================
    # UPDATE
    # ========================================================

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
                "ERROR: No update "
                "values provided."
            )

        values.append(
            event_id
        )

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

        updated = (
            get_event_by_id(
                event_id
            )
        )

        return json.dumps({

            "success": True,

            "action":
                "update",

            "message":
                "Event updated successfully.",

            "event":
                updated,

        }, indent=2)

    # ========================================================
    # DELETE
    # ========================================================

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

    google_api_key=
        GOOGLE_API_KEY,

    temperature=0
)


# ============================================================
# 12. AGENT SYSTEM PROMPT
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
- What workshops are coming?
- Do I have an appointment?
- Am I free?
- Am I available?
- Find a schedule event.
- Search my schedule.

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

IMPORTANT:

If the user wants to move, update, or delete an existing
event and you do not know its event ID:

FIRST use get_schedule.

Then identify the correct event ID.

THEN call update_schedule.

Never invent an event.

Always retrieve schedule information before answering
questions about the user's schedule.

Understand dates such as:

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

When checking availability:

FIRST call get_schedule.

If there are no events during that period,
say that the user appears free.

After adding, updating, or deleting,
clearly confirm what changed.

Return a normal human-readable final answer.
Do NOT return only "...".
"""


# ============================================================
# 13. CREATE AGENT
# ============================================================

agent = create_agent(

    model=llm,

    tools=tools,

    system_prompt=
        SYSTEM_PROMPT
)


# ============================================================
# 14. LANGSERVE INPUT
# ============================================================

class AgentInput(BaseModel):

    input: str = Field(

        description=
            "Message to the Agentic RAG Schedule Assistant"

    )


# ============================================================
# 15. FIXED AGENT EXECUTION
# ============================================================

def get_content_text(content):

    # Standard text
    if isinstance(
        content,
        str
    ):

        return content.strip()

    # Gemini content blocks
    if isinstance(
        content,
        list
    ):

        text_parts = []

        for block in content:

            if isinstance(
                block,
                str
            ):

                text_parts.append(
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

                    text_parts.append(
                        str(text)
                    )

        return "\n".join(
            text_parts
        ).strip()

    if content is None:

        return ""

    return str(
        content
    ).strip()


def run_schedule_agent(x):

    # --------------------------------------------------------
    # Extract user input
    # --------------------------------------------------------

    if isinstance(
        x,
        dict
    ):

        user_input = x.get(
            "input",
            ""
        )

    else:

        user_input = getattr(
            x,
            "input",
            str(x)
        )

    # --------------------------------------------------------
    # Execute agent
    # --------------------------------------------------------

    result = agent.invoke({

        "messages": [

            {
                "role": "user",
                "content": user_input
            }

        ]

    })

    # --------------------------------------------------------
    # Find final AI answer
    # --------------------------------------------------------

    if isinstance(
        result,
        dict
    ):

        messages = result.get(
            "messages",
            []
        )

        # Search backwards for AI response
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

            text = get_content_text(
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

        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        for message in reversed(
            messages
        ):

            content = getattr(
                message,
                "content",
                None
            )

            text = get_content_text(
                content
            )

            if (
                text
                and text != "..."
            ):

                return text

    # --------------------------------------------------------
    # Absolute fallback
    # --------------------------------------------------------

    return (
        "The schedule request was processed, "
        "but no readable final response was returned."
    )


# ============================================================
# 16. LANGSERVE RUNNABLE
# ============================================================

formatted_agent = (

    RunnableLambda(
        run_schedule_agent
    )

).with_types(

    input_type=
        AgentInput,

    output_type=
        str
)


# ============================================================
# 17. FASTAPI
# ============================================================

app = FastAPI(

    title=
        "Agentic RAG Schedule Assistant",

    description=
        "AI schedule management using Gemini, "
        "SQLite and ChromaDB RAG.",

    version=
        "1.0.0"
)


# ============================================================
# 18. HOME
# ============================================================

@app.get("/")
def home():

    return {

        "status":
            "running",

        "application":
            "Agentic RAG Schedule Assistant",

        "health":
            "/health",

        "playground":
            "/agent/playground/",

        "docs":
            "/docs"
    }


# ============================================================
# 19. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "database":
            "SQLite",

        "vector_database":
            "ChromaDB",

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
# 20. LANGSERVE ROUTES
# ============================================================

add_routes(

    app,

    formatted_agent,

    path="/agent",

    playground_type="default"
)


# ============================================================
# 21. LOCAL / RENDER SERVER
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
