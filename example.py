from lfm.llm import Llm


# source: https://en.wikipedia.org/wiki/Photosynthesis
SOURCE_INPUT = """
Photosynthesis is a system of biological processes by which photopigment-bearing autotrophic organisms, such as most plants, algae and cyanobacteria, convert light energy — typically from sunlight — into the chemical energy necessary to fuel their metabolism. The term photosynthesis usually refers to oxygenic photosynthesis, a process that releases oxygen as a byproduct of water splitting. Photosynthetic organisms store the converted chemical energy within the bonds of intracellular organic compounds (complex compounds containing carbon), typically carbohydrates like sugars (mainly glucose, fructose and sucrose), starches, phytoglycogen and cellulose. When needing to use this stored energy, an organism's cells then metabolize the organic compounds through cellular respiration. Photosynthesis plays a critical role in producing and maintaining the oxygen content of the Earth's atmosphere, and it supplies most of the biological energy necessary for complex life on Earth.[2]

Some organisms also perform anoxygenic photosynthesis, which does not produce oxygen. Some bacteria (e.g. purple bacteria) uses bacteriochlorophyll to split hydrogen sulfide as a reductant instead of water, releasing sulfur instead of oxygen, which was a dominant form of photosynthesis in the euxinic Canfield oceans during the Boring Billion.[3][4] Archaea such as Halobacterium also perform a type of non-carbon-fixing anoxygenic photosynthesis, where the simpler photopigment retinal and its microbial rhodopsin derivatives are used to absorb green light and produce a proton (hydron) gradient across the cell membrane, and the subsequent ion movement powers transmembrane proton pumps to directly synthesize adenosine triphosphate (ATP), the "energy currency" of cells. Such archaeal photosynthesis might have been the earliest form of photosynthesis that evolved on Earth, as far back as the Paleoarchean, preceding that of cyanobacteria (see Purple Earth hypothesis).[5]

While the details may differ between species, the process always begins when light energy is absorbed by the reaction centers, proteins that contain photosynthetic pigments or chromophores. In plants, these pigments are chlorophylls (a porphyrin derivative that absorbs the red and blue spectra of light, thus reflecting green) held inside chloroplasts, abundant in leaf cells. In cyanobacteria, they are embedded in the plasma membrane. In these light-dependent reactions, some energy is used to strip electrons from suitable substances, such as water, producing oxygen gas. The hydrogen freed by the splitting of water is used in the creation of two important molecules that participate in energetic processes: reduced nicotinamide adenine dinucleotide phosphate (NADPH) and ATP.

In plants, algae, and cyanobacteria, sugars are synthesized by a subsequent sequence of light-independent reactions called the Calvin cycle. In this process, atmospheric carbon dioxide is incorporated into already existing organic compounds, such as ribulose bisphosphate (RuBP).[6] Using the ATP and NADPH produced by the light-dependent reactions, the resulting compounds are then reduced and removed to form further carbohydrates, such as glucose. In other bacteria, different mechanisms like the reverse Krebs cycle are used to achieve the same end.

The first photosynthetic organisms probably evolved early in the evolutionary history of life using reducing agents such as hydrogen or hydrogen sulfide, rather than water, as sources of electrons.[7] Cyanobacteria appeared later; the excess oxygen they produced contributed directly to the oxygenation of the Earth,[8] which rendered the evolution of complex life possible. The average rate of energy captured by global photosynthesis is approximately 130 terawatts,[9][10][11] which is about eight times the total power consumption of human civilization.[12] Photosynthetic organisms also convert around 100–115 billion tons (91–104 Pg petagrams, or billions of metric tons), of carbon into biomass per year.[13][14] Photosynthesis was discovered in 1779 by Jan Ingenhousz who showed that plants need light, not just soil and water.
"""

if __name__ == "__main__":
    llm = Llm()
    _prompt = "What is the process of photosynthesis in plants?"
    data = llm.generate(prompt=_prompt, source_input=SOURCE_INPUT)

    print(" ")
    print("data", data)
    print(" ")