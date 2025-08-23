from mlx_lm import load, stream_generate


def main():
    repo = "mlx-community/gemma-3-1b-it-bf16"
    model, tokenizer = load(repo)

    prompt = "Write a story about Einstein"

    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    for response in stream_generate(model, tokenizer, prompt, max_tokens=512):
        print(response.text, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
