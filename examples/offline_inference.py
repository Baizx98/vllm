import os

import pandas as pd

from vllm import LLM, SamplingParams

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dataset_path = "/Tan/dataset/balanced-copa"
train_file = os.path.join(dataset_path, "train.csv")
df = pd.read_csv(train_file)
premise_list = df["premise"].tolist()

# Sample prompts.
prompts = premise_list
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(
    model="/Tan/model/Llama-3.2-1B-Instruct",
    gpu_memory_utilization=0.2,
    max_model_len=2048,
    enable_chunked_prefill=False,
    enforce_eager=True,
)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
