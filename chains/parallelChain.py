from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel


load_dotenv()

model1 = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
model2 = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

prompt1 = PromptTemplate(
    template='generate short and simple notes from the following text \n {text}',
    input_variables=['text'],
)
prompt2 = PromptTemplate(
    template='generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)
prompt3 = PromptTemplate(
    template='merge the provided notes and quiz into single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

parallelChain = RunnableParallel({
    'notes':prompt1 | model1 | parser,
    'quiz':prompt2 | model2 | parser,
})

mergedChain = prompt3 | model1 | parser

finalChain = parallelChain | mergedChain

text = """Black holes are often imagined as cosmic vacuum cleaners, aggressively sweeping up everything in their path. In reality, they are not holes at all, but profound concentrations of matter packed into unimaginably tiny spaces. At their core, they represent the ultimate triumph of gravity over all other forces in nature—a region where spacetime itself is warped so extremely that the familiar laws of physics cease to function.The anatomy of a black hole begins before you ever reach its shadowy boundary. Because black holes themselves neither emit nor reflect light, they are effectively invisible. However, as a black hole draws in surrounding gas and dust, this material settles into a flat, rapidly spinning structure known as the accretion disk. Friction and extreme gravitational forces heat this disk to staggering temperatures, causing it to glow brilliantly across multiple wavelengths. The black hole’s intense gravity acts as a lens, warping the light from the far side of the disk so it appears to bend over and under the dark sphere, creating a surreal, glowing halo.   The defining feature of a black hole is the event horizon—the spherical point of no return. It is not a physical surface that you could touch, but a mathematically defined threshold where the velocity required to escape the black hole’s grasp exceeds the speed of light. Because Albert Einstein's theory of general relativity proved that nothing can travel faster than light, anything that crosses the event horizon is forever disconnected from the observable universe.For an intrepid explorer falling toward a relatively small, stellar-mass black hole, the journey would be catastrophic long before they crossed the threshold. Because gravity strengthens exponentially as distance decreases, the pull on the explorer's feet would be significantly stronger than the pull on their head. This extreme tidal force stretches the object vertically and squeezes it horizontally—a grisly but aptly named process called spaghettification. Interestingly, if one were to fall into a supermassive black hole—like Sagittarius A* at the center of the Milky Way, which is 4 million times the mass of our Sun—the transition across the event horizon might actually be quite smooth, as the tidal forces are vastly weaker at that wider boundary.

Ultimately, all paths inside the event horizon lead to the singularity. General relativity predicts that this is an infinitely dense point where the mass of the original collapsed star is entirely concentrated. Here, the equations of gravity, space, and time break down completely. While the singularity remains one of the greatest mysteries in modern astrophysics, black holes themselves are far from being empty voids. They are dynamic, foundational elements of the cosmos, anchoring galaxies and constantly challenging the very limits of human knowledge."""
result = finalChain.invoke({'text' : text})
print(result)