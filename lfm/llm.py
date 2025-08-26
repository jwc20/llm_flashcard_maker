import jsonpickle
from pydantic import BaseModel

SYSTEM_PROMPT = """
Create an Anki flashcard JSON from user-provided text (the "source"), using only the information in that input to generate the back side (answer) based on textbook, article, or similar content. Always reason step by step about what is needed to create a high-quality, educationally useful answer before producing a clear, concise backside. Add references (e.g., citation to section or page if provided in source), and examples if relevant for the concept.

Proceed as follows:

- Analyze the provided "source" text and determine the essential information necessary to answer or explain it.
- Think through and document the logical steps or key points needed for an accurate response. Do not include information not present or infer extra content not directly supported by the source.
- Next, formulate the backside of the Anki card, ensuring clarity, completeness, and alignment with educational goals.
- Add references (such as "as stated in source, section X") if available, and add at least one relevant example in the "examples" field if doing so helps clarify.
- Output must be a JSON object as described below.
- Output order must always be: first, go through all reasoning steps; finally, present the JSON output.

**Expected Output Format:**

- Output a single JSON object, containing:
  - "back": [the complete, clear backside answer based only on the provided "source" text]
  - "references": [list of explicit source references, or empty list if none can be identified]
  - "examples": [list of examples to aid understanding, or empty list if not relevant]

**Include this example:**

Example Input (source):  
"The process of photosynthesis in plants converts carbon dioxide and water into glucose and oxygen using sunlight as the energy source."

**Reasoning steps:**  
1. Identify main process (photosynthesis).
2. Determine the steps involved: carbon dioxide + water + sunlight → glucose + oxygen.
3. Limit answer to only what is stated, do not elaborate outside the provided details.
4. Example may involve a specific plant being exposed to sunlight.


**Important Considerations:**

- Never add or infer information not found in the source text.
- Always perform explicit reasoning about how to create the back side before producing the output.
- The JSON output must not be wrapped in code blocks or any other formatting.
- If references/examples are not possible, return an empty list for each.

**Reminder:**  
Your task is to generate a clear, accurate backside for an Anki flashcard only using the user's provided text, with explicit reasoning before output, always delivering in the required JSON format.

"""


SYSTEM_PROMPT_QUESTION = """
Write study questions for a given source text, suitable for the front (question) side of Anki flashcards.

- Read and understand the source text provided.
- Identify key facts, core concepts, definitions, and important details within the text.
- For each key point, generate a clear and concise question that prompts recall or understanding of that information. Use various question formats (e.g., "What is...?", "How does...?", "Why...?", "List...") as appropriate for the extracted information.
- Each question should be specific, unambiguous, and designed to effectively test comprehension or memory of the material.
- Do NOT include the answer or information that would belong on the back side of the flashcard.
- If the source text is lengthy, select the most critical or exam-relevant points to generate questions.
- Think step-by-step about which facts are most worth remembering or understanding before producing your questions.
- Continue until all main ideas or facts in the source have corresponding questions.

**Output Format:**  
A numbered list of questions. Each item is a single study question, phrased appropriately for the front of an Anki flashcard.

**Example:**  

**Input source text:**  
_The mitochondria is an organelle found in most eukaryotic cells. It is often referred to as the powerhouse of the cell because it produces most of the cell's energy in the form of ATP._

**Output:**  
1. What is the function of mitochondria in eukaryotic cells?  
2. Why is the mitochondria known as the "powerhouse of the cell"?  
3. In what form does the mitochondria provide energy to the cell?

(If working with more complex or longer source texts, generate more questions to cover all main facts and concepts.)

**Important Reminders:**  
- Focus only on the question side for each flashcard.  
- Cover all essential details from the provided source.  
- Use clear, specific, and exam-relevant phrasing for each question.

"""


SYSTEM_PROMPT_BULLET = """
Revise the backside of Anki flashcards to make definitions more concise and easier to digest, formatting each definition as clear bullet points.  
Focus on removing unnecessary words, simplifying language, and breaking down complex explanations into discrete, easy-to-understand bullet points.  
Maintain all essential information while increasing clarity and brevity.  
Before finalizing, review to ensure no key facts are omitted and that the card remains accurate and helpful for studying.

**Output Format:**  
- Provide the revised definition only, formatted as a bulleted list.
- Keep all content in plain text, no markdown formatting needed unless specifically requested.

**Example Input:**  
Original backside:  
A mitochondrion is an organelle found in large numbers in most cells, in which the biochemical processes of respiration and energy production occur. It has a double membrane, the inner part of which is folded inward to form layers (cristae).

**Example Output:**  
- Organelle present in most cells  
- Site of respiration and energy production  
- Surrounded by a double membrane  
- Inner membrane folded into cristae

(For real examples, expect 3–7 bullets capturing the key facts, concisely phrased.)

**Important:**  
- Do not alter the card’s intended meaning or lose essential information.
- Be sure to use simple, direct language and short phrases.
- Always use bullet points for each discrete fact or concept.

---

**Reminder:**  
Edit the backside of Anki flashcards so that definitions become more concise, digestible, and clearly presented as bullet points, while retaining all critical information.

"""


class LlmOutput(BaseModel):
    front: str
    back: str
    references: list[str]
    examples: list[str]


class Llm:
    def __init__(self, repo: str | None = None):
        self._repo = repo
        self._check_repo()

        self._model = None
        self._tokenizer = None
        self._is_loaded = False

    def _check_prompt(self, prompt: str) -> None:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be None or empty")

    def _check_repo(self) -> None:
        if self._repo is None:
            self._repo = "mlx-community/gemma-3-1b-it-bf16"

    def _load_model(self) -> None:
        try:
            if self._repo and not self._is_loaded:
                # lazy import
                from mlx_lm import load, stream_generate  # type: ignore

                self._stream_generate = stream_generate  # cache function ref
                self._model, self._tokenizer = load(
                    self._repo, tokenizer_config={"eos_token": "<end_of_turn>"}
                )
                # self._model, self._tokenizer = load(self._repo, adapter_path="../data_anomaly_adapters")
                self._is_loaded = True

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self._repo}: {str(e)}")

    def generate(
        self,
        source_input: str,
        prompt: str,
        # system_prompt: str | None = None,
        # max_tokens: int = 512,
    ) -> LlmOutput:

        system_prompt = SYSTEM_PROMPT
        max_tokens = 512

        try:
            self._check_prompt(prompt)
            self._load_model()
            if self._model is None or self._tokenizer is None:
                raise RuntimeError("Model or tokenizer not properly loaded")

            if system_prompt is None:
                system_prompt = SYSTEM_PROMPT

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {"role": "source", "content": source_input},
            ]

            formatted_prompt = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )

            result = []
            generate_kwargs = {
                "model": self._model,
                "tokenizer": self._tokenizer,
                "prompt": formatted_prompt,
                "max_tokens": max_tokens,
            }

            for response in self._stream_generate(**generate_kwargs):
                if response.text == "<end_of_turn>":
                    break
                if "\n" in response.text:
                    response.text = response.text.replace("\n", " ")
                result.append(response.text)

            positions_to_remove = []
            for i in range(len(result)):
                if i > 0 and result[i] == "json" and result[i - 1] == "```":
                    positions_to_remove.append(i - 1)
                    positions_to_remove.append(i)
                elif i > 0 and result[i] == "```":
                    positions_to_remove.append(i)
                elif i > 0 and (result[i] == "\n" or result[i] == "<end_of_turn>"):
                    positions_to_remove.append(i)

            for i in reversed(positions_to_remove):
                del result[i]

            result = "".join(result)

            try:
                result_data = jsonpickle.decode(result)
            except Exception as json_error:
                raise RuntimeError(f"Error decoding JSON: {str(json_error)}")

            result_data["front"] = prompt
            return LlmOutput.model_validate(result_data)
        except ValueError as e:
            raise
        except RuntimeError as e:
            raise
        except Exception as e:
            raise RuntimeError(f"Unexpected error during text generation: {str(e)}")

    def generate_batch(
        self,
        source_input: str,
        prompts: list[str],
        # system_prompt: str | None = None,
        # max_tokens: int = 512,
    ) -> list[LlmOutput]:
        results = []
        # if system_prompt == "":
        #     system_prompt = None

        source_input = source_input.replace("\n", " ").strip()

        for i, prompt in enumerate(prompts):
            try:
                result = self.generate(
                    source_input=source_input,
                    prompt=prompt,
                    # system_prompt=system_prompt,
                    # max_tokens=max_tokens,
                )

                result = LlmOutput(
                    front=result.front,
                    back=result.back,
                    references=result.references,
                    examples=result.examples,
                )
                results.append(result)
            except Exception as e:
                print(f"Error generating flashcard {i + 1}: {e}")
                continue
        return results

    def summarize(self):
        pass

    def create_question(self):
        pass

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def repo(self) -> str:
        return self._repo
