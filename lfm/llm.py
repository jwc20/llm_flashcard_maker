from mlx_lm import load, stream_generate


class Llm:
    def __init__(self, repo: str | None = None):
        # self.prompt = prompt
        # self._check_prompt()

        self._repo = repo
        self._check_repo()

        self._model = None
        self._tokenizer = None

    def _check_prompt(self, prompt):
        if prompt is None or len(prompt) <= 0:
            raise ValueError("prompt must not be none")

    def _check_repo(self):
        if self._repo is None:
            self._repo = "mlx-community/gemma-3-1b-it-bf16"

    def _load_repo(self):
        if self._repo:
            self._model, self._tokenizer = load(self._repo)

    def generate(self, prompt):
        self._check_prompt(prompt)
        
        messages = [{"role": "user", "content": prompt}]
        
        if self._repo:
            self._model, self._tokenizer = load(self._repo)
            
        if self._tokenizer is not None and self._model is not None:
            prompt = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )

            result = []

            for response in stream_generate(
                self._model, self._tokenizer, prompt, max_tokens=512
            ):
                # print(response.text, end="", flush=True)
                result.append(response.text)

            return "".join(result)


# def main():
#     repo = "mlx-community/gemma-3-1b-it-bf16"
#     model, tokenizer = load(repo)

#     prompt = "Write a story about Einstein"

#     messages = [{"role": "user", "content": prompt}]
#     prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

#     for response in stream_generate(model, tokenizer, prompt, max_tokens=512):
#         print(response.text, end="", flush=True)
#     print()


if __name__ == "__main__":
    llm = Llm()

    test = llm.generate("Write a story about Einstein")

    print(test)