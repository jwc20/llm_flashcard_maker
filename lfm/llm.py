import jsonpickle
from mlx_lm import load, stream_generate
from pydantic import BaseModel


# SYSTEM_PROMPT = """
# Create an Anki flashcard JSON from user-provided text (the "source"), using only the information in that input to generate the back side (answer) based strictly on textbook, article, or similar content. The back side must be concise—include only the essential, directly relevant information to answer the implied question or explain the concept, avoiding elaboration, unnecessary background, or extraneous detail.
# 
# Always reason step by step before generating the back side, identifying necessary content and documenting your logic. Do not infer or add content not present in the source. Add references (e.g., citation to section or page if provided in source) and examples if relevant for the concept.
# 
# Proceed as follows:
# - Analyze the provided "source" text and determine the minimal essential information necessary to answer or explain it.
# - Document the logical, step-by-step reasoning for deciding what is strictly required for the backside.
# - Ensure your "back" answer is as brief as accuracy allows while still being clear and educational. Do not include background, history, motivation, or extra explanation.
# - Next, formulate the backside of the Anki card, ensuring maximum conciseness, completeness, and alignment with educational goals.
# - Add references ("as stated in source, section X") if available, and at least one tightly focused example in the "examples" field if it helps clarify—otherwise leave empty.
# - Output must be a JSON object as described below.
# - Output order: present all reasoning steps first; only then output the required JSON object.
# 
# **Expected Output Format:**
# - Output a single JSON object, containing:
#    - "back": [the concise, clear backside answer based only on the provided "source" text]
#    - "references": [list of explicit source references, or empty list if none can be identified]
#    - "examples": [list of concise examples to aid understanding, or empty list if not relevant]
# 
# **Include this concise-answer example:**
# Example Input (source):
# "When was Java released and who released it? Java 1.0 was released in 1996 by Sun Microsystems, initially developed to help write programs that run on appliances and electronic devices, which were increasingly becoming more complex. The need arose from the potential for memory errors and security vulnerabilities in existing languages like C and C++, where programmers could miss critical issues. Java’s core strength lies in its automatic memory management, which inherently avoids many of these problems. During Java’s development, it explored web-based applets to enable dynamic content on web pages, greatly increasing its adoption. Today, Java is a general-purpose language used to create standalone applications for conventional computers, and it powers billions of devices worldwide."
# 
# **Reasoning steps:**
# 1. Identify the direct question: "When was Java released and who released it?"
# 2. Locate the information that directly answers the question: "Java 1.0 was released in 1996 by Sun Microsystems."
# 3. Ignore all background, motivation, and additional details, as they are not asked for.
# 4. No clear examples or references indicated in the source.
# 
# **JSON Output:**
# {
#   "back": "Java 1.0 was released in 1996 by Sun Microsystems.",
#   "references": [],
#   "examples": []
# }
# 
# (For real use cases, examples and references should match the specificity and content of the source, but only include concise examples if they directly clarify the main answer.)
# 
# **Important Considerations:**
# - The "back" field must be as concise as possible, including only the information that directly answers the card's implied question—do not include additional background, context, or expanded explanations.
# - Never infer, elaborate, or add information not present in the source text.
# - Reasoning must always explicitly precede the JSON output.
# - The JSON output must not be wrapped in code blocks or any other formatting.
# - If references or concise examples are not possible, use empty lists.
# - Always deliver in the required JSON format with the order: reasoning, then JSON object.
# 
# Your primary objective is to generate a concise, accurate backside for an Anki flashcard based strictly and only on the user's provided text, performing explicit reasoning first, then outputting the result in the required JSON format.
# 
# """


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

**JSON Output:**  
{
  "back": "Photosynthesis in plants transforms carbon dioxide and water into glucose and oxygen, with sunlight providing the necessary energy.",
  "references": ["Sentence from source"],
  "examples": ["For example, when a leaf is exposed to sunlight, it uses carbon dioxide from the air and water from the soil to create glucose and oxygen."]
}

(For real use cases, examples and references should match the specificity and content of the source.)

**Important Considerations:**

- Never add or infer information not found in the source text.
- Always perform explicit reasoning about how to create the back side before producing the output.
- The JSON output must not be wrapped in code blocks or any other formatting.
- If references/examples are not possible, return an empty list for each.

**Reminder:**  
Your task is to generate a clear, accurate backside for an Anki flashcard only using the user's provided text, with explicit reasoning before output, always delivering in the required JSON format.

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
                self._model, self._tokenizer = load(
                    self._repo,
                    tokenizer_config={"eos_token": "<end_of_turn>"}
                )
                # self._model, self._tokenizer = load(self._repo, adapter_path="../data_anomaly_adapters")
                self._is_loaded = True

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self._repo}: {str(e)}")

    def generate(
            self,
            source_input: str,
            prompt: str,
            system_prompt: str | None = None,
            max_tokens: int = 512,
    ) -> LlmOutput:
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
                "max_tokens": max_tokens
            }

            for response in stream_generate(**generate_kwargs):
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
            system_prompt: str | None = None,
            max_tokens: int = 512,
    ) -> list[LlmOutput]:
        results = []

        if system_prompt == "":
            system_prompt = None
            
        # clean up source text
        source_input = source_input.replace("\n", " ").strip()
        

        for i, prompt in enumerate(prompts):
            try:
                result = self.generate(
                    source_input=source_input,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                )

                result = LlmOutput(
                    front=result.front,
                    back=result.back,
                    references=result.references,
                    examples=result.examples
                )

                results.append(result)

            except Exception as e:
                print(f"Error generating flashcard {i + 1}: {e}")
                continue

        return results

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def repo(self) -> str:
        return self._repo
