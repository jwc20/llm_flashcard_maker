from lfm.llm import Llm

# source: GTx CS1331xI Introduction to Object-Oriented Programming with Java I: Foundations and Syntax Basics (2025-2026)
SOURCE_INPUT_JAVA = """
Java 1.0 was officially released in 1996 by a company called Sun Microsystems. It originated out of a need for a language to help write programs that run on appliances and other kinds of electronic devices, which were increasingly getting smarter as computing power became cheaper, faster, and smaller.  Popular languages like C and C++ were already around for many years before Java, and C++ is even Object-Oriented. However, a group of scientists at Sun, who had been experimenting in this space, determined that existing languages needed to be safer in order to be deployed on devices, which sometimes controlled critical aspects of daily life.  
While C and C++ are very efficient and flexible languages, it is quite possible for even experienced programmers to miss significant memory errors and security bugs that could cause a device to behave inconsistently, or even worse, fail during operation.  Imagine an elevator regularly getting stuck because it could not fully execute the code that navigates it through floors due to a memory leak that its programmer did not notice. Or, if you don’t mind tight spaces, imagine a microwave burning your dinner from your favorite takeout spot (and maybe more) because a memory issue caused it to overheat.
Although Java does not guarantee flawless programs, its programmers inherently avoid much of the memory and security issues that are common in languages like C and C++. The reason is primarily due to very powerful mechanisms, like automatic memory management, that its creators built into the language.  Such features are therefore inherent to every Java program, regardless of the experience level of the programmer.
Fortunately, you don’t have to be a device programmer in order to reap such benefits.  During Java’s development, its creators explored several different directions with the language, which greatly increased its scope and popularity.  For example, they noticed the static nature of websites at the time (the 1990s) and developed tools for using Java to write programs that could actually be embedded on web-pages.  Such programs, called applets, helped Java’s adoption by allowing a large population of web programmers to create pages with dynamic content like games, stock tickers, animations, and much more.  
Over time, Java evolved into a general-purpose programming language allowing programmers to write software applications that run on conventional computers like desktops and laptops. As illustrated in the graphic below, new features have been added to the language since version 1.0, and more are expected due to the significant amount of use and investment that has been put into it.
"""

if __name__ == "__main__":
    llm = Llm()
    prompts = ["Compared to C and C++, what issues does Java avoid? How does Java avoid those issues?"]


    print("\n" + "=" * 60)
    print("DIVERSE BATCH WITH EXPLICIT STYLES")
    print("=" * 60)

    results = llm.generate_batch(
        source_input=SOURCE_INPUT_JAVA,
        prompts=prompts,
    )

    for i, result in enumerate(results, 1):
        print(f"\nFlashcard {i}:")
        print(f"  Front: {result.front}")
        print(" ")
        print(f"  Back: {result.back}")
        print(" ")
        print(f"  References: {result.references}")
        print(" ")
        print(f"  Examples: {result.examples}")
        print(" ")
