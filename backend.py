import os
import uuid
import operator
from typing import TypedDict, Annotated
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import certifi
import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights


# =========================================================
# ENVIRONMENT CONFIGURATION
# =========================================================

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


def get_required_env(name: str) -> str:
    """Get a required environment variable."""

    value = os.getenv(name)

    if not value:
        raise ValueError(
            f"{name} is missing. Add it to your .env file."
        )

    return value.strip()


def get_database_url() -> str:
    """Read PostgreSQL URL and enforce SSL."""

    database_url = get_required_env("DATABASE_URL")
    parts = urlsplit(database_url)

    if parts.scheme not in ("postgresql", "postgres"):
        raise ValueError(
            "DATABASE_URL must be a PostgreSQL connection URL."
        )

    query_params = dict(parse_qsl(parts.query))
    query_params["sslmode"] = "require"

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(query_params),
            parts.fragment,
        )
    )


# =========================================================
# API CONFIGURATION
# =========================================================

GROQ_API_KEY = get_required_env("GROQ_API_KEY")


# =========================================================
# LLM CONFIGURATION
# =========================================================

# Available general-purpose model shown in Groq documentation.
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=0,
    max_tokens=2048,
)


# =========================================================
# GRAPH STATE
# =========================================================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int


# =========================================================
# HELPER: LIMIT TOOL OUTPUT
# =========================================================

def limit_text(value, max_chars: int = 4000) -> str:
    """Convert tool output to text and limit prompt size."""

    text = str(value)

    if len(text) > max_chars:
        text = text[:max_chars] + "\n[Results truncated]"

    return text


# =========================================================
# FLIGHT AGENT
# =========================================================

def flight_agent(state: TravelState):
    """Search for flights."""

    query = state["user_query"]

    flight_data = search_flights(query)

    flight_text = limit_text(flight_data, 4000)

    return {
        "flight_results": flight_text,
        "messages": [
            AIMessage(content="Flight search completed.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================================================
# HOTEL AGENT
# =========================================================

def hotel_agent(state: TravelState):
    """Search for hotel information."""

    query = f"Best hotels for {state['user_query']}"

    hotel_data = tavily_search(query)

    hotel_text = limit_text(hotel_data, 4000)

    return {
        "hotel_results": hotel_text,
        "messages": [
            AIMessage(content="Hotel search completed.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================================================
# ITINERARY AGENT
# =========================================================

def itinerary_agent(state: TravelState):
    """Generate a travel itinerary."""

    user_query = limit_text(
        state.get("user_query", ""),
        1500,
    )

    flight_data = limit_text(
        state.get("flight_results", ""),
        3000,
    )

    hotel_data = limit_text(
        state.get("hotel_results", ""),
        3000,
    )

    prompt = f"""
Create a practical travel itinerary.

USER REQUEST:
{user_query}

FLIGHT RESULTS:
{flight_data}

HOTEL RESULTS:
{hotel_data}

REQUIREMENTS:
- Create a day-by-day itinerary.
- Consider budget, travel time, and convenience.
- Recommend relevant attractions and activities.
- Do not invent flight prices or hotel availability.
- Clearly identify estimates and unverified information.
- Mention missing information when necessary.
- Keep the response concise.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are an expert travel planner. "
                    "Create useful, realistic itineraries. "
                    "Never invent booking information."
                )
            ),
            HumanMessage(content=prompt),
        ]
    )

    itinerary_text = limit_text(
        response.content,
        4000,
    )

    return {
        "itinerary": itinerary_text,
        "messages": [
            AIMessage(content="Travel itinerary generated.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================================================
# FINAL RESPONSE AGENT
# =========================================================

def final_agent(state: TravelState):
    """Generate the final travel plan."""

    user_query = limit_text(
        state.get("user_query", ""),
        1500,
    )

    flight_data = limit_text(
        state.get("flight_results", ""),
        2500,
    )

    hotel_data = limit_text(
        state.get("hotel_results", ""),
        2500,
    )

    itinerary = limit_text(
        state.get("itinerary", ""),
        3500,
    )

    final_prompt = f"""
Create the final travel plan.

USER REQUEST:
{user_query}

FLIGHT INFORMATION:
{flight_data}

HOTEL SUGGESTIONS:
{hotel_data}

PROPOSED ITINERARY:
{itinerary}

Use these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Day-by-Day Itinerary
5. Estimated Budget
6. Final Recommendations

INSTRUCTIONS:
- Be clear, practical, and concise.
- Never invent confirmed prices or availability.
- Distinguish estimates from verified information.
- Do not claim that a booking has been made.
- Explain when information is unavailable.
- Avoid unnecessary repetition.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a professional AI travel-planning "
                    "assistant. Provide concise and accurate travel "
                    "recommendations. Never invent booking details."
                )
            ),
            HumanMessage(content=final_prompt),
        ]
    )

    # Append the actual final answer to the graph messages.
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================================================
# BUILD LANGGRAPH WORKFLOW
# =========================================================

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# =========================================================
# POSTGRESQL CHECKPOINTER
# =========================================================

DATABASE_URL = get_database_url()

_conn = None
checkpointer = None
travel_graph = None


def initialize_database():
    """Connect to PostgreSQL and initialize LangGraph."""

    global _conn, checkpointer, travel_graph

    try:
        _conn = psycopg.connect(
            DATABASE_URL,
            autocommit=True,
            row_factory=dict_row,
            connect_timeout=20,
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=3,
        )

        checkpointer = PostgresSaver(_conn)

        # Initialize LangGraph checkpoint tables.
        checkpointer.setup()

        # Compile the graph with persistent checkpoints.
        travel_graph = graph.compile(
            checkpointer=checkpointer
        )

        print("PostgreSQL connected successfully.")
        print("LangGraph checkpointer initialized successfully.")
        print("Groq model: openai/gpt-oss-120b")

    except Exception:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass

        _conn = None
        checkpointer = None
        travel_graph = None

        raise


# Initialize PostgreSQL and compile the graph.
initialize_database()


# =========================================================
# FUNCTION FOR FASTAPI
# =========================================================

def run_travel_agent(
    user_input: str,
    thread_id: str | None = None,
):
    """Execute the travel workflow."""

    if not user_input or not user_input.strip():
        raise ValueError("Please enter a travel request.")

    if travel_graph is None:
        raise RuntimeError(
            "The travel graph has not been initialized."
        )

    user_input = user_input.strip()

    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0,
        },
        config=config,
    )

    messages = result.get("messages", [])

    final_answer = (
        messages[-1].content
        if messages
        else "Sorry, I could not generate a travel plan."
    )

    # Handle models that return content as structured blocks.
    if isinstance(final_answer, list):
        final_answer = "\n".join(
            str(block.get("text", ""))
            if isinstance(block, dict)
            else str(block)
            for block in final_answer
        )

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }