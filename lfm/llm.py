from mlx_lm import load, stream_generate


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
                self._model, self._tokenizer = load(self._repo)
                self._is_loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self._repo}: {str(e)}")

    def generate(self, prompt: str) -> str | None:
        try:
            self._check_prompt(prompt)
            self._load_model()

            if self._model is None or self._tokenizer is None:
                raise RuntimeError("Model or tokenizer not properly loaded")

            messages = [{"role": "user", "content": prompt}]

            formatted_prompt = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )

            result = []
            for response in stream_generate(
                self._model, self._tokenizer, formatted_prompt, max_tokens=512
            ):
                result.append(response.text)

            return "".join(result)

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
