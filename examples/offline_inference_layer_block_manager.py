import os

from vllm import LLM, SamplingParams

os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(
    model="/Tan/model/Llama-3.2-1B-Instruct",
    gpu_memory_utilization=0.9,
    max_model_len=2048,
    enable_chunked_prefill=False,
    enforce_eager=True,
    enable_layer_wise_block=True,
)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
