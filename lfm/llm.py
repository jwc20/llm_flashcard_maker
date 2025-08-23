import jsonpickle
from mlx_lm import load, stream_generate
from pydantic import BaseModel

system_prompt = """
For each data point, identify and resolve the data anomalies. Specifically, follow these guidelines:

1. **Missing Values**:
   - If a field is absent or lacks a value in the text, make reasonable inferences whenever possible.
   - Ensure that each missing field is addressed in the JSON.
   - If the field is not present in the data, add it with a default value. (e.g., "state": "unknown")

2. **Inconsistencies**:
   - Ensure that all similar values are consistently formatted and spelled. For example, for the "state" field, "New Mexico", "NM", and "nm" should all be represented as "New Mexico".

3. **Duplication**:
   - Identify duplicate values and remove all duplicates except for one. 
   - Address duplication only after resolving missing values and inconsistencies.

4. **Case and Format Issues**:
   - Ensure that all field values are consistently formatted and spelled. For example, for the "name" field, "John Doe", "john doe", and "John Doe" should all be represented as "John Doe".

5. **Output Format**:
   - The output should be a list of JSON objects, where each object represents a unique person.
   - Do not include any additional text or formatting in the output.
"""


class Person(BaseModel):
    name: str
    city: str
    state: str


class People(BaseModel):
    people: list[Person]


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
                # self._model, self._tokenizer = load(self._repo)
                self._model, self._tokenizer = load(self._repo,
                                                    adapter_path="../data_anomaly_adapters")
                self._is_loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self._repo}: {str(e)}")

    def generate(self, system_prompt: str, prompt: str) -> list[Person]:
        try:
            self._check_prompt(prompt)
            self._load_model()

            if self._model is None or self._tokenizer is None:
                raise RuntimeError("Model or tokenizer not properly loaded")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            formatted_prompt = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )

            result = []

            for response in stream_generate(
                    self._model, self._tokenizer, formatted_prompt, max_tokens=512
            ):
                if response.finish_reason == "stop":
                    break
                generated_text = response.text
                # generated_text = generated_text.strip()
                # if generated_text:
                #     result.append(jsonpickle.decode(generated_text))
                result.append(generated_text)

            result_string = "".join(result)
            if result_string:
                # Clean the string
                result_string = result_string.replace("```", "").replace("<end_of_turn>", "")
                # Keep only ASCII characters, replace others with space
                result_string = ''.join(char if ord(char) < 128 else ' ' for char in result_string)
                result_string = result_string.replace("\n", "")
                # Ensure string starts with [ and ends with ]
                result_string = result_string.strip()
                if not result_string.startswith('['):
                    new_start_pos = 0
                    for c in result_string:
                        if c == '[':
                            result_string = result_string[new_start_pos:]
                        else:
                            new_start_pos += 1

                if not result_string.endswith(']'):
                    new_end_pos = len(result_string) - 1
                    for c in result_string[::-1]:
                        if c == ']':
                            result_string = result_string[:new_end_pos + 1]
                        else:
                            new_end_pos -= 1

                try:
                    result = jsonpickle.decode(result_string)
                except Exception as e:
                    print(f"Error decoding JSON: {e}")
                    result = []
            else:
                result = []

            return result

        except ValueError as e:
            raise
        except RuntimeError as e:
            raise
        except Exception as e:
            raise RuntimeError(f"Unexpected error during text generation: {str(e)}")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def repo(self) -> str:
        return self._repo


if __name__ == "__main__":
    import json

    data = [
        {"name": "John Doe", "city": "new york", "state": "new york"},  # Case and format issues
        {"name": "Sam Brown", "city": "dallas", "state": "texas"},  # Inconsistent state abbreviation
        {"name": "John Doe", "city": "new york", "state": "new york"},  # Duplicate entry
    ]

    llm = Llm()
    _prompt = "Process the following data: \n\n" + jsonpickle.dumps(data)

    data = llm.generate(system_prompt, _prompt)
    people = data
    print(jsonpickle.encode(people, indent=4))
