from transformers import pipeline
from IPython.display import Audio



pipe = pipeline("text-to-speech", model = "suno/bark-small")
text = "Ladybugs have had important roles in culture and religion, being associated with luck, love, fertility and prophecy."
ouput = pipe(text)
Audio(ouput["audio"], rate = output["sampling_rate"])