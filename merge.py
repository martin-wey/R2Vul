from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct", device_map="auto")
model = PeftModel.from_pretrained(base_model, "PATH TO MODEL ADAPTER")
merged_model = model.merge_and_unload()

output_name = "r2vul-1.5B"
merged_model.save_pretrained(output_name)
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct").save_pretrained(output_name)
