from vllm import LLM, SamplingParams
import json

from datasets import load_dataset





ds = load_dataset("/Tan/dataset/writingprompts", split="train")
print("Loaded writingprompts, number of samples:", len(ds))

PROMPT_TEMPLATE = """
You are a creative fiction novelist (Role).  
Task: Given the following writing prompt from a user, continue and write a **long, detailed, coherent story**, with rich descriptions, emotional depth, vivid scenes, and well-structured narrative (Instruction + Specification).  
Output requirements: at least 20000 words; clear beginning, middle and end; include character development, setting descriptions, plot progression, emotional arc (Specification).  

Writing prompt: \"\"\"{user_prompt}\"\"\"

Please produce the story in plain text. No extra commentary. Just the story. (Presentation)
"""

prompts = []
for i,item in enumerate(ds):
    if i>500:
        break
    user_prompt = item.get("prompt", "").strip()
    if not user_prompt:
        continue
    full_prompt = PROMPT_TEMPLATE.format(user_prompt=user_prompt)
    prompts.append(full_prompt)


# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=5000)

# Create an LLM.
llm = LLM(
    model="/Tan/model/Qwen2.5-14B-Instruct",
    disable_log_stats=True,
    gpu_memory_utilization=0.93,
    disable_async_output_proc=True,
    enforce_eager=True,
    enable_chunked_prefill=False,
    max_model_len=16000,

)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Pmpt: {len(output.prompt_token_ids)}, Generated: {len(generated_text)}")
