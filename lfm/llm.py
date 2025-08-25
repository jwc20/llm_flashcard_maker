import jsonpickle
from mlx_lm import load, stream_generate
from pydantic import BaseModel


SOURCE_INPUT = """
Photosynthesis is a system of biological processes by which photopigment-bearing autotrophic organisms, such as most plants, algae and cyanobacteria, convert light energy — typically from sunlight — into the chemical energy necessary to fuel their metabolism. The term photosynthesis usually refers to oxygenic photosynthesis, a process that releases oxygen as a byproduct of water splitting. Photosynthetic organisms store the converted chemical energy within the bonds of intracellular organic compounds (complex compounds containing carbon), typically carbohydrates like sugars (mainly glucose, fructose and sucrose), starches, phytoglycogen and cellulose. When needing to use this stored energy, an organism's cells then metabolize the organic compounds through cellular respiration. Photosynthesis plays a critical role in producing and maintaining the oxygen content of the Earth's atmosphere, and it supplies most of the biological energy necessary for complex life on Earth.[2]

Some organisms also perform anoxygenic photosynthesis, which does not produce oxygen. Some bacteria (e.g. purple bacteria) uses bacteriochlorophyll to split hydrogen sulfide as a reductant instead of water, releasing sulfur instead of oxygen, which was a dominant form of photosynthesis in the euxinic Canfield oceans during the Boring Billion.[3][4] Archaea such as Halobacterium also perform a type of non-carbon-fixing anoxygenic photosynthesis, where the simpler photopigment retinal and its microbial rhodopsin derivatives are used to absorb green light and produce a proton (hydron) gradient across the cell membrane, and the subsequent ion movement powers transmembrane proton pumps to directly synthesize adenosine triphosphate (ATP), the "energy currency" of cells. Such archaeal photosynthesis might have been the earliest form of photosynthesis that evolved on Earth, as far back as the Paleoarchean, preceding that of cyanobacteria (see Purple Earth hypothesis).[5]

While the details may differ between species, the process always begins when light energy is absorbed by the reaction centers, proteins that contain photosynthetic pigments or chromophores. In plants, these pigments are chlorophylls (a porphyrin derivative that absorbs the red and blue spectra of light, thus reflecting green) held inside chloroplasts, abundant in leaf cells. In cyanobacteria, they are embedded in the plasma membrane. In these light-dependent reactions, some energy is used to strip electrons from suitable substances, such as water, producing oxygen gas. The hydrogen freed by the splitting of water is used in the creation of two important molecules that participate in energetic processes: reduced nicotinamide adenine dinucleotide phosphate (NADPH) and ATP.

In plants, algae, and cyanobacteria, sugars are synthesized by a subsequent sequence of light-independent reactions called the Calvin cycle. In this process, atmospheric carbon dioxide is incorporated into already existing organic compounds, such as ribulose bisphosphate (RuBP).[6] Using the ATP and NADPH produced by the light-dependent reactions, the resulting compounds are then reduced and removed to form further carbohydrates, such as glucose. In other bacteria, different mechanisms like the reverse Krebs cycle are used to achieve the same end.

The first photosynthetic organisms probably evolved early in the evolutionary history of life using reducing agents such as hydrogen or hydrogen sulfide, rather than water, as sources of electrons.[7] Cyanobacteria appeared later; the excess oxygen they produced contributed directly to the oxygenation of the Earth,[8] which rendered the evolution of complex life possible. The average rate of energy captured by global photosynthesis is approximately 130 terawatts,[9][10][11] which is about eight times the total power consumption of human civilization.[12] Photosynthetic organisms also convert around 100–115 billion tons (91–104 Pg petagrams, or billions of metric tons), of carbon into biomass per year.[13][14] Photosynthesis was discovered in 1779 by Jan Ingenhousz who showed that plants need light, not just soil and water.
"""


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
                self._model, self._tokenizer = load(self._repo, tokenizer_config={"eos_token": "<end_of_turn>"})
                # self._model, self._tokenizer = load(self._repo, adapter_path="../data_anomaly_adapters")
                self._is_loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self._repo}: {str(e)}")

    def generate(self, system_prompt: str, prompt: str):
        try:
            self._check_prompt(prompt)
            self._load_model()

            if self._model is None or self._tokenizer is None:
                raise RuntimeError("Model or tokenizer not properly loaded")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {"role": "source", "content": SOURCE_INPUT},
            ]

            formatted_prompt = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )

            result = []

            for response in stream_generate(
                    self._model, self._tokenizer, formatted_prompt, max_tokens=512
            ):
                print(response.text, end="", flush=True)
                result.append(response.text)

            
            return "".join(result)

            # return LlmOutput.model_validate(result_data)
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

    llm = Llm()
    _prompt = "What is the process of photosynthesis in plants?"

    data = llm.generate(SYSTEM_PROMPT, _prompt)
    people = data
    print(jsonpickle.encode(people, indent=4))
