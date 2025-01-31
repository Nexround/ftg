from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM
from src.module.Qwen2Model import CustomQwen2ForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# model_name = "/root/.cache/modelscope/hub/Qwen/Qwen2-0.5B-Instruct"

model = CustomQwen2ForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
print(model)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.forward(
    **model_inputs,
    target_token_idx=(len(model_inputs)-1),
    use_cache=False
)
partitioning = model.forward_with_partitioning(target_token_idx=(len(model_inputs)-1))
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(partitioning)