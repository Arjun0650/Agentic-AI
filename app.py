import os
import json
import sqlite3
from datetime import date, datetime, timedelta

import uvicorn
import chromadb

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
# CONFIGURATION
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY environment variable is missing. "
        "Add GOOGLE_API_KEY in Render Environment Variables."
    )


DB_PATH = "schedule.db"
CHROMA_PATH = "./chroma_schedule"

COLLECTION_NAME = "schedule_events"


# ============================================================
# SQLITE DATABASE
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

    # Create sample schedule only once
    if count == 0:

        today = date.today()

        templates = [
            {
                "title": "Team Meeting",
                "description": "Weekly project discussion with the development team.",
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
                "description": "Work on the Agentic RAG Schedule Assistant.",
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

        # Generate events covering next 30 days
        for day_number in range(30):

            event_date = today + timedelta(days=day_number)

            # One main event per day
            first = templates[day_number % len(templates)]

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
                event_date.isoformat(),
                first["start_time"],
                first["end_time"],
                first["location"],
            ))

            # Add an extra event on selected days
            if day_number % 3 == 0:

                second = templates[
                    (day_number + 2) % len(templates)
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
                    event_date.isoformat(),
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


def events_on_date(event_date):

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
# GEMINI EMBEDDINGS
# ============================================================

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)


# ============================================================
# CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)


def event_to_document(event):

    return (
        f"Event ID: {event['id']}\n"
        f"Title: {event['title']}\n"
        f"Description: {event['description']}\n"
        f"Type: {event['event_type']}\n"
        f"Date: {event['date']}\n"
        f"Start Time: {event['start_time']}\n"
        f"End Time: {event['end_time']}\n"
        f"Location: {event['location']}\n"
        f"Status: {event['status']}"
    )


def rebuild_vector_store():

    global collection

    # Re-create collection cleanly
    try:
        chroma_client.delete_collection(
            COLLECTION_NAME
        )
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

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
            "date": event["date"],
            "start_time": event["start_time"],
            "end_time": event["end_time"],
            "event_type": event["event_type"],
            "title": event["title"],
        })

    embeddings = embedding_model.embed_documents(
        documents
    )

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


# Create RAG index at startup
rebuild_vector_store()


# ============================================================
# DATE HELPERS
# ============================================================

def next_weekday(weekday_number):
    """
    Monday = 0
    Tuesday = 1
    ...
    Sunday = 6
    """

    today = date.today()

    days_ahead = (
        weekday_number - today.weekday()
    ) % 7

    if days_ahead == 0:
        days_ahead = 7

    return today + timedelta(
        days=days_ahead
    )


def extract_known_date(query):

    query = query.lower()

    today = date.today()

    if "tomorrow" in query:
        return (
            today + timedelta(days=1)
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

    for name, number in weekdays.items():

        if name in query:

            return next_weekday(
                number
            ).isoformat()

    return None


# ============================================================
# RAG SEARCH
# ============================================================

def semantic_schedule_search(
    query,
    n_results=8
):

    total = collection.count()

    if total == 0:
        return []

    n_results = min(
        n_results,
        total
    )

    query_embedding = (
        embedding_model.embed_query(query)
    )

    result = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=n_results,
    )

    output = []

    documents = (
        result.get(
            "documents",
            [[]]
        )[0]
    )

    metadatas = (
        result.get(
            "metadatas",
            [[]]
        )[0]
    )

    distances = (
        result.get(
            "distances",
            [[]]
        )[0]
    )

    for index, document in enumerate(
        documents
    ):

        item = {
            "document": document,
            "metadata": (
                metadatas[index]
                if index < len(metadatas)
                else {}
            ),
        }

        if index < len(distances):
            item["distance"] = (
                distances[index]
            )

        output.append(item)

    return output


# ============================================================
# TOOL 1 — GET SCHEDULE
# ============================================================

@tool
def get_schedule(query: str) -> str:
    """
    Retrieve schedule information using the SQLite schedule
    database and ChromaDB semantic RAG search.

    Use this tool for:
    - today's schedule
    - tomorrow's schedule
    - Friday / weekday schedule
    - meetings
    - workshops
    - appointments
    - tasks
    - availability checks
    - searching schedule by natural-language query
    """

    query_lower = query.lower()

    target_date = extract_known_date(
        query
    )

    # --------------------------------------------------------
    # Exact date retrieval when date is understood
    # --------------------------------------------------------

    if target_date:

        events = events_on_date(
            target_date
        )

        # Event type filters
        if "meeting" in query_lower:

            events = [
                event
                for event in events
                if event["event_type"]
                == "meeting"
            ]

        elif "workshop" in query_lower:

            events = [
                event
                for event in events
                if event["event_type"]
                == "workshop"
            ]

        elif "appointment" in query_lower:

            events = [
                event
                for event in events
                if event["event_type"]
                == "appointment"
            ]

        elif "task" in query_lower:

            events = [
                event
                for event in events
                if event["event_type"]
                == "task"
            ]

        # Afternoon filtering
        if "afternoon" in query_lower:

            events = [
                event
                for event in events
                if event["start_time"]
                >= "12:00"
                and event["start_time"]
                < "17:00"
            ]

        # Morning filtering
        elif "morning" in query_lower:

            events = [
                event
                for event in events
                if event["start_time"]
                >= "06:00"
                and event["start_time"]
                < "12:00"
            ]

        # Evening filtering
        elif "evening" in query_lower:

            events = [
                event
                for event in events
                if event["start_time"]
                >= "17:00"
            ]

        return json.dumps({
            "query": query,
            "date": target_date,
            "events": events,
            "count": len(events),
        }, indent=2)

    # --------------------------------------------------------
    # Semantic RAG search
    # --------------------------------------------------------

    rag_results = semantic_schedule_search(
        query,
        n_results=8
    )

    return json.dumps({
        "query": query,
        "retrieval_type": "ChromaDB semantic RAG",
        "results": rag_results,
    }, indent=2)


# ============================================================
# TOOL 2 — UPDATE SCHEDULE
# ============================================================

@tool
def update_schedule(
    action: str,
    title: str = "",
    description: str = "",
    event_type: str = "meeting",
    event_date: str = "",
    start_time: str = "",
    end_time: str = "",
    location: str = "",
    event_id: int = 0,
) -> str:
    """
    Add, update, or delete schedule entries.

    action:
    - add
    - update
    - delete

    For update/delete, event_id should normally be found
    first using get_schedule.
    """

    action = action.lower().strip()

    conn = get_connection()

    # ========================================================
    # ADD
    # ========================================================

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

        new_event_id = cursor.lastrowid

        conn.close()

        # Keep vector DB synchronized
        rebuild_vector_store()

        new_event = get_event_by_id(
            new_event_id
        )

        return json.dumps({
            "success": True,
            "action": "add",
            "message": "Event added successfully.",
            "event": new_event,
        }, indent=2)

    # ========================================================
    # UPDATE
    # ========================================================

    if action == "update":

        if not event_id:
            conn.close()

            return (
                "ERROR: event_id is required for update. "
                "Use get_schedule first to find the event."
            )

        existing = conn.execute("""
            SELECT *
            FROM events
            WHERE id = ?
            AND status = 'scheduled'
        """, (event_id,)).fetchone()

        if not existing:
            conn.close()

            return (
                f"ERROR: scheduled event "
                f"{event_id} was not found."
            )

        fields = []
        values = []

        if title:
            fields.append("title = ?")
            values.append(title)

        if description:
            fields.append(
                "description = ?"
            )
            values.append(description)

        if event_type:
            fields.append(
                "event_type = ?"
            )
            values.append(event_type)

        if event_date:
            fields.append("date = ?")
            values.append(event_date)

        if start_time:
            fields.append(
                "start_time = ?"
            )
            values.append(start_time)

        if end_time:
            fields.append(
                "end_time = ?"
            )
            values.append(end_time)

        if location:
            fields.append(
                "location = ?"
            )
            values.append(location)

        if not fields:

            conn.close()

            return (
                "ERROR: no update values "
                "were provided."
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

        updated_event = get_event_by_id(
            event_id
        )

        return json.dumps({
            "success": True,
            "action": "update",
            "message": "Event updated successfully.",
            "event": updated_event,
        }, indent=2)

    # ========================================================
    # DELETE
    # ========================================================

    if action == "delete":

        if not event_id:

            conn.close()

            return (
                "ERROR: event_id is required for delete. "
                "Use get_schedule first."
            )

        existing = conn.execute("""
            SELECT *
            FROM events
            WHERE id = ?
            AND status = 'scheduled'
        """, (event_id,)).fetchone()

        if not existing:

            conn.close()

            return (
                f"ERROR: scheduled event "
                f"{event_id} was not found."
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
            "message": (
                f"Event {event_id} deleted successfully."
            ),
        }, indent=2)

    conn.close()

    return (
        "ERROR: invalid action. "
        "Use add, update, or delete."
    )


# ============================================================
# EXACTLY TWO AGENT TOOLS
# ============================================================

tools = [
    get_schedule,
    update_schedule,
]


# ============================================================
# GEMINI CHAT MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)


# ============================================================
# AGENT
# ============================================================

SYSTEM_PROMPT = f"""
You are an intelligent Agentic RAG Schedule Assistant.

Today's date is:
{date.today().isoformat()}

You manage the user's schedule for the next 30 days.

You have EXACTLY TWO TOOLS:

1. get_schedule

Use get_schedule whenever the user asks to:
- view their schedule
- find meetings
- find workshops
- find appointments
- find tasks
- search for an event
- ask what is scheduled
- check availability
- check whether they are free
- find an event before changing or deleting it

The get_schedule tool uses ChromaDB semantic retrieval for
natural-language schedule searches.

2. update_schedule

Use update_schedule whenever the user asks to:
- add an event
- add a meeting
- add an appointment
- add a task
- add a workshop
- move an event
- update an event
- reschedule an event
- delete an event

IMPORTANT AGENT RULES:

1. Never invent schedule information.

2. For moving, updating, or deleting an existing event,
   first call get_schedule to identify the event and obtain
   its event ID.

3. After finding the event, call update_schedule using the
   correct event_id.

4. Understand natural-language dates.

Examples:

today
tomorrow
Monday
Tuesday
Wednesday
Thursday
Friday
Saturday
Sunday

5. Convert user times to 24-hour HH:MM format.

Examples:

3 PM -> 15:00
4 PM -> 16:00
10 AM -> 10:00

6. If the user says:
   "3 PM for one hour"

   use:
   start_time = 15:00
   end_time = 16:00

7. When checking availability, call get_schedule first.

8. If there are no events during the requested period,
   clearly say the user appears free.

9. Keep responses concise and natural.

10. After adding, updating, or deleting an event,
    clearly confirm what changed.

11. If the user's request is ambiguous, retrieve relevant
    schedule information before deciding what to modify.
"""


agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)


# ============================================================
# LANGSERVE INPUT
# ============================================================

class AgentInput(BaseModel):

    input: str = Field(
        description=(
            "Message to the Agentic "
            "RAG Schedule Assistant"
        )
    )


def format_for_agent(x):

    if isinstance(x, dict):

        user_input = (
            x.get("input", "")
        )

    else:

        user_input = getattr(
            x,
            "input",
            str(x)
        )

    return {
        "messages": [
            (
                "user",
                user_input
            )
        ]
    }


# ============================================================
# IMPORTANT FIX — EXTRACT GEMINI RESPONSE CORRECTLY
# ============================================================

def content_to_text(content):

    # Normal string response
    if isinstance(content, str):

        return content.strip()

    # Gemini/LangChain may return list of blocks
    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, str):

                parts.append(item)

            elif isinstance(item, dict):

                if "text" in item:

                    text = item.get("text")

                    if text:
                        parts.append(
                            str(text)
                        )

                elif (
                    item.get("type")
                    == "text"
                ):

                    text = item.get(
                        "text",
                        ""
                    )

                    if text:
                        parts.append(text)

        result = "\n".join(
            parts
        ).strip()

        if result:
            return result

    if content is None:
        return ""

    return str(content)


def extract_text_response(
    agent_output
):

    try:

        if agent_output is None:

            return (
                "The agent returned "
                "an empty response."
            )

        # ----------------------------------------------------
        # Standard LangGraph / create_agent output
        # ----------------------------------------------------

        if isinstance(
            agent_output,
            dict
        ):

            messages = (
                agent_output.get(
                    "messages"
                )
            )

            if messages:

                # Search backwards for the final
                # assistant message with actual text
                for message in reversed(
                    messages
                ):

                    role = getattr(
                        message,
                        "type",
                        ""
                    )

                    content = getattr(
                        message,
                        "content",
                        None
                    )

                    text = content_to_text(
                        content
                    )

                    if (
                        role in (
                            "ai",
                            "assistant"
                        )
                        and text
                        and text != "..."
                    ):

                        return text

                # Fallback to any message containing text
                for message in reversed(
                    messages
                ):

                    content = getattr(
                        message,
                        "content",
                        None
                    )

                    text = content_to_text(
                        content
                    )

                    if (
                        text
                        and text != "..."
                    ):

                        return text

            # ------------------------------------------------
            # Some versions nest state output
            # ------------------------------------------------

            for value in (
                agent_output.values()
            ):

                if not isinstance(
                    value,
                    dict
                ):
                    continue

                nested_messages = (
                    value.get(
                        "messages"
                    )
                )

                if not nested_messages:
                    continue

                for message in reversed(
                    nested_messages
                ):

                    content = getattr(
                        message,
                        "content",
                        None
                    )

                    text = content_to_text(
                        content
                    )

                    if (
                        text
                        and text != "..."
                    ):

                        return text

            # Last fallback
            return json.dumps(
                agent_output,
                default=str,
                indent=2,
            )

        # Direct AI message
        if hasattr(
            agent_output,
            "content"
        ):

            text = content_to_text(
                agent_output.content
            )

            if text:
                return text

        return str(
            agent_output
        )

    except Exception as error:

        return (
            "Error extracting agent "
            f"response: {error}"
        )


# ============================================================
# LANGSERVE CHAIN
# ============================================================

formatted_agent = (
    RunnableLambda(
        format_for_agent
    )
    | agent
    | RunnableLambda(
        extract_text_response
    )
).with_types(
    input_type=AgentInput,
    output_type=str,
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=(
        "Agentic RAG Schedule Assistant"
    ),
    description=(
        "30-day AI schedule assistant "
        "using Gemini, SQLite and ChromaDB."
    ),
    version="1.0.0",
)


@app.get("/")
def root():

    return {
        "status": "running",
        "application": (
            "Agentic RAG "
            "Schedule Assistant"
        ),
        "health": "/health",
        "playground": (
            "/agent/playground/"
        ),
        "docs": "/docs",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "database": "SQLite",
        "vector_database": "ChromaDB",
        "rag": True,
        "tools": [
            "get_schedule",
            "update_schedule",
        ],
        "events": len(
            get_all_events()
        ),
    }


# ============================================================
# LANGSERVE ROUTES
# ============================================================

add_routes(
    app,
    formatted_agent,
    path="/agent",
    playground_type="default",
)


# ============================================================
# LOCAL / RENDER RUN
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
        port=port,
    )
