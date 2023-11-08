from datasets import load_dataset
import gradio as gr
from datasets import Audio
import librosa
from transformers import WhisperFeatureExtractor, AutoProcessor
import numpy as np
import matplotlib.pyplot as plt


minds = load_dataset("PolyAI/minds14", name = "en-AU", split = "train")
# print(minds)
example = minds[0]
# print(example)

id2label = minds.features["intent_class"].int2str
# print(id2label(example["intent_class"]))

columns_to_remove = ["lang_id", "english_transcription"]
minds = minds.remove_columns(columns_to_remove)
# print(minds)

def generate_audio():
	example = minds.shuffle()[0]
	audio = example["audio"]
	return (audio["sampling_rate"], audio["array"],), id2label(example["intent_class"])

# with gr.Blocks() as demo:
# 	with gr.Column():
# 		for _ in range(4):
# 			audio, label = generate_audio()
# 			output = gr.Audio(audio, label = label)

# demo.launch(debug = True)

minds = minds.cast_column("audio", Audio(sampling_rate = 16_000))
print(minds[0])

Max_Duration_in_Seconds = 20.0

def is_audio_length_in_range(input_length):
	return input_length < Max_Duration_in_Seconds

new_column = [librosa.get_duration(path = x) for x in minds["path"]]
minds = minds.add_column("duration", new_column)
minds = minds.filter(is_audio_length_in_range, input_columns = ["duration"])
minds = minds.remove_columns(["duration"])
# print(minds)

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

def prepare_dataset(example):
	audio = example["audio"]
	features = feature_extractor(audio["array"], sampling_rate = audio["sampling_rate"], padding = True)
	return features

minds = minds.map(prepare_dataset)
# print(minds)

example = minds[0]
input_features = example["input_features"]
# librosa.display.specshow(np.asarray(input_features[0]), x_axis = "time", y_axis = "mel", sr = feature_extractor.sampling_rate, hop_length = feature_extractor.hop_length)
# plt.colorbar()
# plt.show()

processor = AutoProcessor.from_pretrained("openai/whisper-small")