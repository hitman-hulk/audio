from datasets import load_dataset


minds = load_dataset("PolyAI/minds14", name = "en-AU", split = "train")
# print(minds)
example = minds[0]
print(example)