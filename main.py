import jsonpickle
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor

from lfm import Llm

app = FastAPI()
templates = Jinja2Templates(directory="templates")

llm = Llm()

# thread pool for running blocking LLM operations
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
        source_text: str = Form(...),
        questions: str = Form(...),
        system_prompt: str | None = Form(None),
        max_tokens: int = Form(512)
):
    try:
        question_list = [q.strip() for q in questions.split('\n') if q.strip()]

        if not question_list:
            raise ValueError("No valid questions provided")

        if system_prompt == "":
            system_prompt = None

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            executor,
            llm.generate_batch,
            source_text,
            question_list,
            system_prompt,
            max_tokens
        )

        if not results:
            raise ValueError("No flashcards generated")

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
        # html_content += "</div>"
        # html_content += f"""
        #     <form hx-post="/export-flashcards" hx-target="#export-container" hx-swap="outerHTML">
        #         <input type="hidden" name="flashcards_json" value='{}'>
        #         <button type="submit">Export All Flashcards</button>
        #     </form>
        #     <div id="export-container"></div>
        # """

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
