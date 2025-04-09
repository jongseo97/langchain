from transformers import AutoModelForCausalLM, AutoTokenizer

model_name_or_path = "/home/jspark/projects/LLM-tool-calling/model/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype="auto",
    device_map="auto",
)