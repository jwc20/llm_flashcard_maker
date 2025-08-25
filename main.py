import json
import jsonpickle
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor
import html

from lfm import Llm

app = FastAPI()
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
        source_text: str = Form(...),
        questions: str = Form(...),
):
    try:
        question_list = [q.strip() for q in questions.split('\n') if q.strip()]

        if not question_list:
            raise ValueError("No valid questions provided")

        # if system_prompt == "":
        #     system_prompt = None

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            executor,
            llm.generate_batch,
            source_text,
            question_list,
            # system_prompt,
            # max_tokens
        )

        if not results:
            raise ValueError("No flashcards generated")
        # 
        # 
        # # results_json = jsonpickle.encode(results)
        # print(results)

        # [LlmOutput(front='Which of the following statements are true? You may select more than one, but you must select at least one.', back="Java utilizes a hybrid approach: compilation to bytecode and interpretation by a virtual machine (JVM). Bytecode is stored in files with a '.class' extension, allowing for near-machine-level execution without strict assumptions about the processor. Interpreted languages are generally slower due to translation from high-level to low-level. The JVM acts as a bridge, combining compilation and interpretation to provide platform independence and optimizations.  For example, the JVM’s Just-In-Time (JIT) compilation further enhances performance by dynamically translating bytecode to machine code during runtime.", references=['“Java, on the other hand, takes a hybrid approach to offer the benefits of compilation and interpretation.”', "“Java utilizes a hybrid approach: compilation to bytecode and interpretation by a virtual machine (JVM). Bytecode is stored in files with a '.class' extension, allowing for near-machine-level execution without strict assumptions about the processor.”"], examples=['The Java interpreter, often referred to as the JVM, translates bytecode into machine code, enabling Java programs to run on various platforms.  For instance, the JVM’s JIT compilation optimizes the bytecode during runtime, improving performance.'])]
        # convert above to json

        # results = [LlmOutput(
        #     front='Which of the following statements are true? You may select more than one, but you must select at least one.',
        #     back="Java utilizes a hybrid approach: compilation to bytecode and interpretation by a virtual machine (JVM). Bytecode is stored in files with a '.class' extension, allowing for near-machine-level execution without strict assumptions about the processor. Interpreted languages are generally slower due to translation from high-level to low-level. The JVM acts as a bridge, combining compilation and interpretation to provide platform independence and optimizations.  For example, the JVM’s Just-In-Time (JIT) compilation further enhances performance by dynamically translating bytecode to machine code during runtime.",
        #     references=[
        #         '“Java, on the other hand, takes a hybrid approach to offer the benefits of compilation and interpretation.”',
        #         "“Java utilizes a hybrid approach: compilation to bytecode and interpretation by a virtual machine (JVM). Bytecode is stored in files with a '.class' extension, allowing for near-machine-level execution without strict assumptions about the processor.”"],
        #     examples=[
        #         'The Java interpreter, often referred to as the JVM, translates bytecode into machine code, enabling Java programs to run on various platforms.  For instance, the JVM’s JIT compilation optimizes the bytecode during runtime, improving performance.'])]

        flashcards_data = [r.model_dump() for r in results]
        flashcards_json = jsonpickle.dumps(flashcards_data)
        flashcards_json_escaped = html.escape(flashcards_json)

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
