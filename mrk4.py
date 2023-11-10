from datasets import load_dataset
from datasets import Audio
from transformers import pipeline


minds = load_dataset("PolyAI/minds14", name="de-DE", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
example = minds[0]
print(example["transcription"])

asr = pipeline("automatic-speech-recognition", model = "maxidl/wav2vec2-large-xlsr-german")
print(asr(example["audio"]["array"]))