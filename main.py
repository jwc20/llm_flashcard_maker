from typing import Annotated
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select

import uuid
import jsonpickle
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor
import html
from datetime import datetime, timezone
from lfm import Llm, LlmOutput
from uuid import UUID, uuid4

from sqlmodel import Field, SQLModel, Column, JSON


class Flashcard(SQLModel, table=True):
    id: UUID | None = Field(default_factory=uuid4, primary_key=True)
    front: str = Field(index=True)
    back: str = Field(index=True)
    references: list[str] = Field(default=[], sa_column=Column(JSON))
    examples: list[str] = Field(default=[], sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield
    print("shutting down")


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

llm = Llm()

executor = ThreadPoolExecutor(max_workers=2)


class FlashcardRequest(BaseModel):
    source_text: str
    questions: str
    system_prompt: str | None = None
    max_tokens: int = 512


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/generate-batch", response_class=HTMLResponse)
async def generate_batch(
        request: Request,
        session: SessionDep,
        source_text: str = Form(...),
        questions: str = Form(...),
):
    try:
        question_list = [q.strip() for q in questions.split('\n') if q.strip()]

        if not question_list:
            raise ValueError("No valid questions provided")

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            executor,
            llm.generate_batch,
            source_text,
            question_list,
        )

        if not results:
            raise ValueError("No flashcards generated")

        flashcards_data = []
        for r in results:
            data = r.model_dump()
            data['id'] = uuid.uuid4()
            flashcards_data.append(data)
            db_flashcard = Flashcard.model_validate(data)
            session.add(db_flashcard)

        session.commit()

        flashcards_json = jsonpickle.dumps(
            [{"front": f.front, "back": f.back, "references": f.references, "examples": f.examples} for f in results]
        )
        flashcards_json_escaped = html.escape(flashcards_json)

        html_content = "<h3>Generated Flashcards</h3>"
        html_content += f"<p>Successfully generated {len(results)} flashcard(s).</p>"
        html_content += "<div>"
        for i, flashcard in enumerate(results):
            escaped_front = html.escape(flashcard.front)
            escaped_back = html.escape(flashcard.back)
            html_content += f"""
                <div style="border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                    <h4>Flashcard {i + 1}</h4>
                    <p><strong>Question:</strong> {escaped_front}</p>
                    <p><strong>Answer:</strong> {escaped_back}</p>
                </div>
            """
        html_content += "</div>"
        html_content += f"""
            <form hx-post="/export-flashcards" hx-target="#export-container" hx-swap="outerHTML">
                <input type="hidden" name="flashcards_json" value="{flashcards_json_escaped}">
                <button type="submit">Export to json</button>
            </form>
            <div id="export-container"></div>
        """

        return HTMLResponse(content=html_content)

    except ValueError as e:
        return HTMLResponse(content=f"<div style='color: red;'>Error: {str(e)}</div>")
    except Exception as e:
        return HTMLResponse(content=f"<div style='color: red;'>An unexpected error occurred: {str(e)}</div>")


@app.post("/export-flashcards", response_class=HTMLResponse)
async def export_flashcards(
        request: Request,
        flashcards_json: str = Form(...)
):
    try:
        flashcards = jsonpickle.loads(flashcards_json)
        export_data = {
            "flashcards": flashcards,
            "total": len(flashcards)
        }

        return templates.TemplateResponse(
            "export_result.html",
            {
                "request": request,
                "export_data": jsonpickle.dumps(export_data, indent=2),
                "count": len(flashcards)
            }
        )
    except jsonpickle.JSONDecodeError as e:
        return HTMLResponse(f"<div class='text-red-500'>Export failed: Invalid JSON data.</div>")
    except Exception as e:
        return HTMLResponse(f"<div class='text-red-500'>Export failed: {str(e)}</div>")
