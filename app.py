import os
import json
import sqlite3
import hashlib
import math
import re
from datetime import date, datetime, timedelta

import chromadb
import uvicorn

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent


# ============================================================
# 1. CONFIGURATION
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is missing. "
        "Add GOOGLE_API_KEY in Render Environment Variables."
    )


DB_PATH = "schedule.db"
CHROMA_PATH = "./chroma_schedule"
COLLECTION_NAME = "schedule_events"

# Very small local vectors.
# No Gemini embedding API.
# No ONNX model.
# No large model download.
EMBEDDING_SIZE = 128


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
    # Create sample data only once
    # --------------------------------------------------------

    if count == 0:

        today = date.today()

        templates = [

            {
                "title": "Team Meeting",
                "description": "Weekly project discussion with the development team.",
                "event_type": "meeting",
                "start_time": "10:00",
                "end_time": "11:00",
                "location": "Conference Room"
            },

            {
                "title": "AI Workshop",
                "description": "Artificial Intelligence and Machine Learning workshop.",
                "event_type": "workshop",
                "start_time": "14:00",
                "end_time": "16:00",
                "location": "AI Lab"
            },

            {
                "title": "Project Development",
                "description": "Development work for the Agentic RAG Schedule Assistant.",
                "event_type": "task",
                "start_time": "09:00",
                "end_time": "11:00",
                "location": "Home"
            },

            {
                "title": "Doctor Appointment",
                "description": "Regular doctor appointment.",
                "event_type": "appointment",
                "start_time": "16:00",
                "end_time": "17:00",
                "location": "City Hospital"
            },

            {
                "title": "Client Meeting",
                "description": "Discuss project requirements and progress with the client.",
                "event_type": "meeting",
                "start_time": "15:00",
                "end_time": "16:00",
                "location": "Online"
            },

            {
                "title": "Python Practice",
                "description": "Practice Python programming and data structures.",
                "event_type": "task",
                "start_time": "18:00",
                "end_time": "19:00",
                "location": "Home"
            }
        ]

        # ----------------------------------------------------
        # Generate events for the next 30 days
        # ----------------------------------------------------

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
                first["location"]
            ))

            # Add a second event every third day
            if i % 3 == 0:

                second = templates[
                    (i + 2) % len(templates)
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
                    second["location"]
                ))

        conn.commit()

    conn.close()


# ============================================================
# 3. DATABASE HELPERS
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

    return dict(row) if row else None


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


def event_exists(event_id):

    conn = get_connection()

    row = conn.execute("""
        SELECT id
        FROM events
        WHERE id = ?
        AND status = 'scheduled'
    """, (
        event_id,
    )).fetchone()

    conn.close()

    return row is not None


# ============================================================
# 4. INITIALIZE SQLITE
# ============================================================

initialize_database()


# ============================================================
# 5. LIGHTWEIGHT LOCAL EMBEDDING
# ============================================================

def lightweight_embedding(text: str):

    """
    Small feature-hashing embedding.

    IMPORTANT:
    - No Gemini API call
    - No ONNX
    - No HuggingFace
    - No model download
    - Very low RAM use
    """

    vector = [
        0.0
    ] * EMBEDDING_SIZE

    words = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )

    for word in words:

        digest = hashlib.md5(
            word.encode("utf-8")
        ).hexdigest()

        index = (
            int(
                digest[:8],
                16
            )
            % EMBEDDING_SIZE
        )

        vector[index] += 1.0

    # Normalize vector

    norm = math.sqrt(
        sum(
            value * value
            for value in vector
        )
    )

    if norm > 0:

        vector = [
            value / norm
            for value in vector
        ]

    return vector


# ============================================================
# 6. CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


def get_collection():

    # We always supply our own embeddings.
    # Chroma therefore does NOT need to download
    # its default ONNX model.

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

    # Delete old collection
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
    embeddings = []

    for event in events:

        document = event_to_document(
            event
        )

        documents.append(
            document
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
                event["event_type"]
        })

        embeddings.append(
            lightweight_embedding(
                document
            )
        )

    # IMPORTANT:
    # Embeddings are supplied manually.
    # Chroma does NOT call its ONNX embedding model.

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )


# Build RAG database
rebuild_vector_store()


# ============================================================
# 7. DATE HELPERS
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

    query_lower = query.lower()

    today = date.today()

    # --------------------------------------------------------
    # Today
    # --------------------------------------------------------

    if "today" in query_lower:

        return today.isoformat()

    # --------------------------------------------------------
    # Tomorrow
    # --------------------------------------------------------

    if "tomorrow" in query_lower:

        return (
            today
            + timedelta(days=1)
        ).isoformat()

    # --------------------------------------------------------
    # YYYY-MM-DD
    # --------------------------------------------------------

    iso_match = re.search(
        r"\b(\d{4}-\d{2}-\d{2})\b",
        query
    )

    if iso_match:

        try:

            parsed = datetime.strptime(
                iso_match.group(1),
                "%Y-%m-%d"
            ).date()

            return parsed.isoformat()

        except ValueError:

            pass

    # --------------------------------------------------------
    # Weekday
    # --------------------------------------------------------

    weekdays = {

        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6
    }

    for name, number in weekdays.items():

        if name in query_lower:

            return (
                next_weekday(
                    number
                )
            ).isoformat()

    # --------------------------------------------------------
    # Month name + day
    # Example: August 15
    # --------------------------------------------------------

    month_names = {

        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12
    }

    for month_name, month_number in (
        month_names.items()
    ):

        pattern = (
            rf"\b{month_name}\s+"
            rf"(\d{{1,2}})\b"
        )

        match = re.search(
            pattern,
            query_lower
        )

        if match:

            day_number = int(
                match.group(1)
            )

            year = today.year

            try:

                candidate = date(
                    year,
                    month_number,
                    day_number
                )

                # If that date already passed,
                # use next year.
                if candidate < today:

                    candidate = date(
                        year + 1,
                        month_number,
                        day_number
                    )

                return candidate.isoformat()

            except ValueError:

                return None

    return None


# ============================================================
# 8. CHROMADB RAG SEARCH
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

    # Create query vector locally
    query_vector = lightweight_embedding(
        query
    )

    # IMPORTANT:
    # query_embeddings is used instead of query_texts.
    # This prevents Chroma from invoking its default
    # ONNX embedding model.

    results = collection.query(
        query_embeddings=[
            query_vector
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

    distances = results.get(
        "distances",
        [[]]
    )[0]

    output = []

    for index, document in enumerate(
        documents
    ):

        metadata = {}

        if index < len(metadatas):

            metadata = (
                metadatas[index]
                or {}
            )

        item = {

            "document":
                document,

            "metadata":
                metadata
        }

        if index < len(distances):

            item["distance"] = (
                distances[index]
            )

        output.append(
            item
        )

    return output


# ============================================================
# 9. TOOL 1 — GET SCHEDULE
# ============================================================

@tool
def get_schedule(
    query: str
) -> str:

    """
    Retrieve relevant schedule information.

    Use this tool for:
    - today's schedule
    - tomorrow's schedule
    - weekday schedules
    - explicit dates
    - meetings
    - workshops
    - tasks
    - appointments
    - availability checks
    - natural-language RAG searches
    """

    query_lower = query.lower()

    target_date = (
        extract_date_from_query(
            query
        )
    )

    # ========================================================
    # DATE-BASED RETRIEVAL
    # ========================================================

    if target_date:

        events = get_events_by_date(
            target_date
        )

        # ----------------------------------------------------
        # Event type filters
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Time of day filters
        # ----------------------------------------------------

        if "morning" in query_lower:

            events = [
                event
                for event in events
                if (
                    "06:00"
                    <= event["start_time"]
                    < "12:00"
                )
            ]

        elif "afternoon" in query_lower:

            events = [
                event
                for event in events
                if (
                    "12:00"
                    <= event["start_time"]
                    < "17:00"
                )
            ]

        elif "evening" in query_lower:

            events = [
                event
                for event in events
                if (
                    event["start_time"]
                    >= "17:00"
                )
            ]

        return json.dumps(
            {
                "success": True,
                "retrieval_type":
                    "SQLite exact schedule retrieval",
                "query": query,
                "date": target_date,
                "count": len(events),
                "events": events
            },
            indent=2
        )

    # ========================================================
    # RAG RETRIEVAL
    # ========================================================

    rag_results = semantic_search(
        query
    )

    return json.dumps(
        {
            "success": True,
            "retrieval_type":
                "ChromaDB vector RAG",
            "query": query,
            "results": rag_results
        },
        indent=2
    )


# ============================================================
# 10. TOOL 2 — UPDATE SCHEDULE
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
    event_id: int = 0
) -> str:

    """
    Add, update, or delete schedule entries.

    action must be:
    - add
    - update
    - delete

    For update/delete operations, use get_schedule first
    when the event ID is not known.
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

        # Keep Chroma synchronized
        rebuild_vector_store()

        event = get_event_by_id(
            new_id
        )

        return json.dumps(
            {
                "success": True,
                "action": "add",
                "message":
                    "Event added successfully.",
                "event": event
            },
            indent=2
        )

    # ========================================================
    # UPDATE
    # ========================================================

    if action == "update":

        if not event_id:

            conn.close()

            return (
                "ERROR: event_id is required. "
                "Use get_schedule first to find "
                "the correct event."
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
                f"ERROR: scheduled event "
                f"{event_id} was not found."
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
                "ERROR: no update values "
                "were provided."
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

        # Keep Chroma synchronized
        rebuild_vector_store()

        updated = get_event_by_id(
            event_id
        )

        return json.dumps(
            {
                "success": True,
                "action": "update",
                "message":
                    "Event updated successfully.",
                "event": updated
            },
            indent=2
        )

    # ========================================================
    # DELETE
    # ========================================================

    if action == "delete":

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

        # Keep Chroma synchronized
        rebuild_vector_store()

        return json.dumps(
            {
                "success": True,
                "action": "delete",
                "message":
                    f"Event {event_id} deleted successfully."
            },
            indent=2
        )

    conn.close()

    return (
        "ERROR: invalid action. "
        "Use add, update, or delete."
    )


# ============================================================
# 11. EXACTLY TWO REQUIRED TOOLS
# ============================================================

tools = [
    get_schedule,
    update_schedule
]


# ============================================================
# 12. GEMINI AGENT MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# 13. AGENT PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are an intelligent Agentic RAG Schedule Assistant.

TODAY'S DATE:
{date.today().isoformat()}

Your job is to manage the user's schedule for the
next 30 days.

You have EXACTLY TWO tools.

============================================================
TOOL 1: get_schedule
============================================================

Use get_schedule whenever the user asks:

- What do I have scheduled?
- What do I have tomorrow?
- What do I have today?
- What meetings do I have?
- What workshops do I have?
- What appointments do I have?
- What tasks do I have?
- Am I free?
- Am I available?
- Find an event.
- Search my schedule.
- What is happening on a certain date?

get_schedule performs database retrieval and
ChromaDB vector RAG retrieval.

============================================================
TOOL 2: update_schedule
============================================================

Use update_schedule whenever the user asks:

- Add an event.
- Add a meeting.
- Add a workshop.
- Add an appointment.
- Add a task.
- Move an event.
- Reschedule an event.
- Update an event.
- Delete an event.

============================================================
IMPORTANT RULES
============================================================

1. Never invent schedule information.

2. Always use get_schedule before answering a question
   about the user's schedule.

3. When the user wants to MOVE, UPDATE or DELETE an
   existing event:

   FIRST call get_schedule.

   Find the correct event and its event ID.

   THEN call update_schedule.

4. Understand natural-language dates:

   today
   tomorrow
   Monday
   Tuesday
   Wednesday
   Thursday
   Friday
   Saturday
   Sunday
   August 15
   YYYY-MM-DD

5. Convert times to 24-hour HH:MM format.

   3 PM = 15:00
   4 PM = 16:00
   10 AM = 10:00

6. If the user says:

   "3 PM for one hour"

   use:

   start_time = 15:00
   end_time = 16:00

7. If moving an event, preserve the event duration
   unless the user explicitly changes the duration.

8. For availability questions:

   FIRST call get_schedule.

   If no event overlaps the requested period,
   clearly say the user appears free.

9. After adding, updating, or deleting an event,
   clearly confirm what changed.

10. Give short, natural, human-readable answers.

11. Do not output raw tool JSON unless necessary.

12. Do not answer only with "...".
"""


# ============================================================
# 14. CREATE AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT
)


# ============================================================
# 15. RESPONSE EXTRACTION
# ============================================================

def extract_text(
    content
):

    # Normal text response

    if isinstance(
        content,
        str
    ):

        return content.strip()

    # Gemini may return content blocks

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

        return str(
            result
        )

    messages = result.get(
        "messages",
        []
    )

    # Search from the end for final AI response

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

    # Secondary fallback

    for message in reversed(
        messages
    ):

        content = getattr(
            message,
            "content",
            None
        )

        text = extract_text(
            content
        )

        if (
            text
            and text != "..."
        ):

            return text

    return (
        "The schedule request was processed, "
        "but no readable response was returned."
    )


# ============================================================
# 16. FASTAPI
# ============================================================

app = FastAPI(
    title=
        "Agentic RAG Schedule Assistant",
    description=
        "Agentic schedule manager using Gemini, "
        "SQLite and ChromaDB vector RAG.",
    version=
        "1.0.0"
)


# ============================================================
# 17. CHAT REQUEST
# ============================================================

class ChatRequest(
    BaseModel
):

    message: str


# ============================================================
# 18. CHAT API
# ============================================================

@app.post("/chat")
def chat(
    request: ChatRequest
):

    try:

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role":
                            "user",

                        "content":
                            request.message
                    }
                ]
            }
        )

        final_response = (
            extract_final_response(
                result
            )
        )

        return {
            "success":
                True,

            "response":
                final_response
        }

    except Exception as error:

        print(
            "CHAT ERROR:",
            repr(error)
        )

        return {
            "success":
                False,

            "response":
                f"Error: {str(error)}"
        }


# ============================================================
# 19. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "application":
            "Agentic RAG Schedule Assistant",

        "model":
            "gemini-3.6-flash",

        "database":
            "SQLite",

        "vector_database":
            "ChromaDB",

        "embedding":
            "lightweight 128-dimensional local feature hashing",

        "external_embedding_api":
            False,

        "onnx_model":
            False,

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
# 20. WEB INTERFACE
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
            #06172e,
            #123e6c
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
        24px 30px;
}


.header h1 {

    margin: 0;

    font-size: 29px;
}


.header p {

    margin:
        8px 0 0;

    opacity: .75;

    font-size: 15px;
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

    line-height: 1.55;

    white-space: pre-wrap;

    word-wrap: break-word;
}


.bot {

    background: white;

    color: #172033;

    border:
        1px solid #e2e7ee;

    box-shadow:
        0 3px 12px
        rgba(0,0,0,.04);
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

    background: white;
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

    box-shadow:
        0 0 0 3px
        rgba(21,79,136,.08);
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


    .header h1 {

        font-size: 23px;
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
Gemini 3.6 Flash • ChromaDB Vector RAG • SQLite
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
• Delete my project meeting tomorrow.

</div>


</div>


<div class="input-area">


<input
    id="input"
    type="text"
    autocomplete="off"
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
                data.response
                || "Something went wrong.",
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
# 21. RENDER / LOCAL RUN
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
