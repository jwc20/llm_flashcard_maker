# llm_flashcard_maker

## usage

clone and install requirements.

run fastapi

```python
fastapi dev main.py
```

to see the flashcards in sql database use datagrip/dbeaver/db browser for sqlite or datasette.

## note

this app uses mlx_lm library, only mac can use this.

## resources for future extensions

- [fixing data problems with llm](https://www.subsystem.ai/blog/fixing-data-problems-with-large-language-models-a-practical-guide)

  - [Can Foundation Models Wrangle Your Data?](https://arxiv.org/abs/2205.09911)
  - [Large Language Models as Data Preprocessors](https://arxiv.org/abs/2308.16361)

- [llm engineers handbook](https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072/)
- [data ingestion for llm fine tuning and rag inference](https://ibrahim-olawale13.medium.com/data-ingestion-for-llm-fine-tuning-and-rag-inference-part-1-6ff730c722ed)

- [qdrant pdf retrieval](https://qdrant.tech/documentation/advanced-tutorials/pdf-retrieval-at-scale/)
- [qdrant faq question answering](https://qdrant.tech/articles/faq-question-answering/)
- [fastapi and htmx](https://testdriven.io/blog/fastapi-htmx/)
- [mlx-lm](https://github.com/ml-explore/mlx-lm?tab=readme-ov-file)
- [Fine-Tuning Gemma 3 1B to Build a Tool-Calling Agent](https://www.linkedin.com/pulse/fine-tuning-gemma-3-1b-build-tool-calling-agent-mihir-jha--nqrof/)
- [chatgpt document analyzer](https://web.archive.org/web/20230926113710/https://hackernoon.com/build-a-document-analyzer-with-chatgpt-google-cloud-and-python)
