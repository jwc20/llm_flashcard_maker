from typing import Annotated
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select

import json
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


from sqlmodel import Field, SQLModel, Column, JSON

class Flashcard(SQLModel, table=True):
    id: str | None = Field(default=None, primary_key=True)
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
    
# class Flashcard(SQLModel):
#     id: str = Field(default=None, primary_key=True)
#     front: str = Field(index=True) 
#     back: str = Field(index=True)
#     references: list[str] = Field(index=True)
#     examples: list[str] = Field(index=True)



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

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            executor,
            llm.generate_batch,
            source_text,
            question_list,
        )
        # 
        # results = [LlmOutput(front='What is java?',
        #                   back='Java 1.0 was released in 1996 by Sun Microsystems, initially designed to simplify the creation of programs for electronic devices. The need arose from the increasing complexity of these devices and the desire to ensure their reliability, particularly when they controlled critical functions. C and C++ were already established languages, but Sun’s scientists recognized the potential for safety issues in devices that controlled everyday life.  Java’s core strength lies in its automatic memory management, a feature that significantly reduces the risk of memory leaks and security vulnerabilities.  Unlike languages like C and C++, Java’s inherent safety mechanisms are built into the language itself, allowing developers to focus on the logic of their programs rather than worrying about low-level details.  The language’s evolution included tools for embedding dynamic content on web pages, further expanding its reach and utility.  Java’s continued development and adoption demonstrate its enduring value as a versatile and reliable programming language.',
        #                   references=['Sun Microsystems. (1996). Java 1.0 Release Notes.',
        #                               'Sun Microsystems. (1996). Java 1.0 Release Notes.'], examples=[
        #         'For example, when a leaf is exposed to sunlight, it uses carbon dioxide from the air and water from the soil to create glucose and oxygen, with sunlight providing the necessary energy.']), LlmOutput(
        #     front='What are the advantages of java?',
        #     back='Java 1.0 was released in 1996 by Sun Microsystems, originating from the need to simplify program creation for electronic devices, which were increasingly complex due to improved computing power.  It emerged from a desire to create a language that could reliably run on devices, mitigating potential issues with memory leaks and security vulnerabilities.  C and C++ were established languages, but Java’s developers prioritized safety, ensuring programs could be deployed on devices.  Java’s core strength lies in its automatic memory management, a feature built into the language that significantly reduces the risk of memory errors and security flaws. Unlike languages like C and C++, Java’s inherent safety mechanisms are built into the language itself.  The language’s evolution included tools for embedding dynamic content on web pages, further expanding its reach and utility.  Java’s continued development and adoption demonstrate its enduring value as a versatile and reliable programming language.',
        #     references=['Sun Microsystems. (1996). Java 1.0 Release Notes.',
        #                 'Sun Microsystems. (1996). Java 1.0 Release Notes.'], examples=[
        #         'For example, when a leaf is exposed to sunlight, it uses carbon dioxide from the air and water from the soil to create glucose and oxygen, with sunlight providing the necessary energy.'])]

        if not results:
            raise ValueError("No flashcards generated")

        flashcards_data = [r.model_dump() for r in results]
        flashcards_json = jsonpickle.dumps(flashcards_data)
        flashcards_json_escaped = html.escape(flashcards_json)

        
        for d in flashcards_data:
            d['id'] = str(uuid.uuid4())
            db_flashcard = Flashcard.model_validate(d)
            session.add(db_flashcard)
        session.commit()

        html_content = "<h3>Generated Flashcards</h3>"
        html_content += f"<p>Successfully generated {len(results)} flashcard(s).</p>"
        html_content += "<div>"
        for i, flashcard in enumerate(results):
            html_content += f"""
                <div style="border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                    <h4>Flashcard {i + 1}</h4>
                    <p><strong>Question:</strong> {flashcard.front}</p>
                    <p><strong>Answer:</strong> {flashcard.back}</p>
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

    except Exception as e:
        return HTMLResponse(content=f"<div style='color: red;'>Error generating flashcards: {str(e)}</div>")


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
    except Exception as e:
        return HTMLResponse(f"<div class='text-red-500'>Export failed: {str(e)}</div>")
