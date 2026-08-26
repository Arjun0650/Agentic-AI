import os
import json
import sqlite3
from datetime import datetime, timedelta, date

import uvicorn

from fastapi import FastAPI
from langserve import add_routes

from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from pydantic import BaseModel, Field


# =========================================================
# DATABASE
# =========================================================

DB_PATH = "schedule.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():

    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT,
            event_type TEXT,
            date TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            location TEXT,
            status TEXT DEFAULT 'scheduled'
        )
    """)

    conn.commit()

    count = conn.execute(
        "SELECT COUNT(*) FROM events"
    ).fetchone()[0]

    if count == 0:

        today = date.today()

        samples = [
            (
                "Team Meeting",
                "Weekly project discussion",
                "meeting",
                (today + timedelta(days=1)).isoformat(),
                "10:00",
                "11:00",
                "Conference Room"
            ),
            (
                "AI Workshop",
                "Artificial Intelligence workshop",
                "workshop",
                (today + timedelta(days=2)).isoformat(),
                "14:00",
                "16:00",
                "AI Lab"
            ),
            (
                "Project Task",
                "Work on Agentic RAG Schedule Assistant",
                "task",
                (today + timedelta(days=3)).isoformat(),
                "09:00",
                "11:00",
                "Home"
            ),
            (
                "Doctor Appointment",
                "Regular appointment",
                "appointment",
                (today + timedelta(days=4)).isoformat(),
                "16:00",
                "17:00",
                "Hospital"
            ),
        ]

        conn.executemany("""
            INSERT INTO events
            (
                title,
                description,
                event_type,
                date,
                start_time,
                end_time,
                location
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, samples)

        conn.commit()

    conn.close()


initialize_database()


# =========================================================
# TOOL 1 — GET SCHEDULE
# =========================================================

@tool
def get_schedule(query: str) -> str:
    """
    Retrieve schedule information based on a user's
    date, time, availability, meeting, task, workshop,
    appointment, or general schedule query.
    """

    query_lower = query.lower()

    conn = get_connection()

    events = conn.execute("""
        SELECT *
        FROM events
        WHERE status = 'scheduled'
        ORDER BY date, start_time
    """).fetchall()

    conn.close()

    events = [dict(e) for e in events]

    today = date.today()

    # Tomorrow
    if "tomorrow" in query_lower:

        target = (
            today + timedelta(days=1)
        ).isoformat()

        events = [
            e for e in events
            if e["date"] == target
        ]

    # Today
    elif "today" in query_lower:

        target = today.isoformat()

        events = [
            e for e in events
            if e["date"] == target
        ]

    # Meeting
    if "meeting" in query_lower:

        events = [
            e for e in events
            if e["event_type"] == "meeting"
        ]

    # Workshop
    if "workshop" in query_lower:

        events = [
            e for e in events
            if e["event_type"] == "workshop"
        ]

    # Appointment
    if "appointment" in query_lower:

        events = [
            e for e in events
            if e["event_type"] == "appointment"
        ]

    # Task
    if "task" in query_lower:

        events = [
            e for e in events
            if e["event_type"] == "task"
        ]

    if not events:
        return "No matching scheduled events were found."

    return json.dumps(events, indent=2)


# =========================================================
# TOOL 2 — UPDATE SCHEDULE
# =========================================================

@tool
def update_schedule(
    action: str,
    title: str = "",
    event_date: str = "",
    start_time: str = "",
    end_time: str = "",
    event_id: int = 0
) -> str:
    """
    Add, update or delete schedule events.

    action must be:
    add
    update
    delete
    """

    conn = get_connection()

    action = action.lower()

    # ADD
    if action == "add":

        cursor = conn.execute("""
            INSERT INTO events
            (
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
            title,
            "",
            "meeting",
            event_date,
            start_time,
            end_time,
            ""
        ))

        conn.commit()

        event_id = cursor.lastrowid

        conn.close()

        return (
            f"Event added successfully. "
            f"Event ID: {event_id}"
        )

    # UPDATE
    elif action == "update":

        if not event_id:
            conn.close()
            return "event_id is required."

        if start_time:

            conn.execute("""
                UPDATE events
                SET start_time = ?
                WHERE id = ?
            """, (
                start_time,
                event_id
            ))

        if end_time:

            conn.execute("""
                UPDATE events
                SET end_time = ?
                WHERE id = ?
            """, (
                end_time,
                event_id
            ))

        if event_date:

            conn.execute("""
                UPDATE events
                SET date = ?
                WHERE id = ?
            """, (
                event_date,
                event_id
            ))

        conn.commit()
        conn.close()

        return "Event updated successfully."

    # DELETE
    elif action == "delete":

        if not event_id:
            conn.close()
            return "event_id is required."

        conn.execute("""
            UPDATE events
            SET status = 'cancelled'
            WHERE id = ?
        """, (
            event_id,
        ))

        conn.commit()

        conn.close()

        return "Event deleted successfully."

    conn.close()

    return "Invalid action."


# =========================================================
# TOOLS
# =========================================================

tools = [
    get_schedule,
    update_schedule
]


# =========================================================
# GEMINI MODEL
# =========================================================

GOOGLE_API_KEY = os.environ.get(
    "GOOGLE_API_KEY"
)

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY environment variable is missing."
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# =========================================================
# AGENT
# =========================================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=f"""
You are an intelligent Schedule Assistant.

Today's date is {date.today().isoformat()}.

You manage the user's schedule.

You have exactly two tools:

1. get_schedule
Use it whenever the user asks:
- What is scheduled?
- What do I have tomorrow?
- What meetings do I have?
- Am I free?
- Find an appointment
- Find a workshop
- Find a task

2. update_schedule
Use it whenever the user asks:
- Add an event
- Add a meeting
- Move an event
- Change an event
- Delete an event

IMPORTANT RULES:

For update or delete operations where you do not know
the event ID, first call get_schedule to locate the
correct event.

Understand dates such as today and tomorrow.

Convert times to 24-hour HH:MM format.

If the user says:
"3 PM for one hour"

use:
start_time = 15:00
end_time = 16:00

Never invent schedule events.
"""
)


# =========================================================
# LANGSERVE INPUT / OUTPUT
# =========================================================

class AgentInput(BaseModel):

    input: str = Field(
        description="Message to the Schedule Assistant"
    )


def format_for_agent(x):

    if isinstance(x, dict):
        user_input = x["input"]

    else:
        user_input = x.input

    return {
        "messages": [
            ("user", user_input)
        ]
    }


def extract_text_response(output):

    if not isinstance(output, dict):
        return str(output)

    messages = output.get("messages")

    if messages:

        last = messages[-1]

        return getattr(
            last,
            "content",
            str(last)
        )

    return str(output)


formatted_agent = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Agentic Schedule Assistant"
)


@app.get("/")
def root():

    return {
        "status": "running",
        "message": "Agentic Schedule Assistant API",
        "playground": "/agent/playground/"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


add_routes(
    app,
    formatted_agent,
    path="/agent",
    playground_type="default"
)


# =========================================================
# LOCAL RUN
# =========================================================

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
