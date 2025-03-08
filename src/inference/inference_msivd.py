import json
import os

import evaluate
import torch
from datasets import DatasetDict, load_from_disk
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM, BitsAndBytesConfig
)

from src.arguments import InferenceArguments, DataArguments


system_prompt = """<s>system
You are an expert in locating and fixing security vulnerabilities in code, can help answer vulnerability questions, and suggest repairs
</s>
"""

round1 = """<s>human
Does the following C code have a security vulnerability: 
'''{code}
'''
</s>
<s>bot
"""

round2 = """<s>human
What is the description of the vulnerablity?
</s>
<s>bot
"""

round3 = """<s>human
Locate the lines that are vulnerable and should be repaired.
</s>
<s>bot
"""

rounds = (round1, round2, round3)

def main():
    parser = HfArgumentParser((InferenceArguments, DataArguments))
    args, data_args = parser.parse_args_into_dataclasses()
    set_seed(42)

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        "codellama/CodeLlama-13b-Instruct-hf",
        trust_remote_code=True
    )
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        bnb_4bit_use_double_quant=True,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer.eos_token = "</s>"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    tokenizer.pad_token = "<unk>"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    dataset = load_from_disk(os.path.join(data_args.data_dir, data_args.dataset_name))["test"]
    test_datasets = DatasetDict({
        f"test_{lang}": dataset.filter(lambda x: x["lang"] == lang, num_proc=data_args.preprocessing_num_workers)
        for lang in data_args.languages
    })

    generation_config = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
    }

    precision = evaluate.load("precision")
    recall = evaluate.load("recall")

    def compute_f_scores(precision, recall):
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

        beta_0_3 = 0.3
        beta_0_3_squared = beta_0_3 ** 2
        f03_score = (1 + beta_0_3_squared) * (precision * recall) / (beta_0_3_squared * precision + recall + 1e-10)

        return {"f1": f1_score, "f03": f03_score}

    for ds_key, dataset in test_datasets.items():
        args.lang = ds_key.split("_")[-1]
        dataset = dataset.select_columns(["function", "vulnerable", "lang"])

        predictions = []
        labels = []
        file_name = f"{ds_key}_t{args.temperature}" if args.do_sample else f"{ds_key}"
        output_file_path = os.path.join(args.output_dir, file_name)

        with open(f"{output_file_path}.jsonl", "w") as f:
            for sample in tqdm(dataset):
                # MSIVD model works as a chat model, with three rounds of interaction max.
                # if the function is predicted as non-vulnerable: a single round.
                # if the function is predicted as vulnerable: three rounds to get the whole explanation.
                prompt = system_prompt
                response = ""
                for i, round in enumerate(rounds, start=1):
                    if i == 1:
                        prompt += round.format(code=sample['function'])
                    else:
                        prompt += round

                    inputs = tokenizer(prompt, return_tensors="pt")
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    outputs = model.generate(
                        inputs=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        **generation_config,
                    )
                    outputs = outputs[:, inputs['input_ids'].shape[1]:]
                    decoded_sequence = tokenizer.decode(outputs[0], skip_special_tokens=False)
                    prompt += decoded_sequence + "\n"
                    response += decoded_sequence + "\n"

                    if i == 1:
                        if "No." in decoded_sequence:
                            predictions.append(0)
                            break
                        else:
                            predictions.append(1)
                        if "_noexpl" in args.model_name_or_path:
                            break
                labels.append(sample["vulnerable"])
                json.dump({
                    "response": response,
                    "prediction": predictions[-1],
                }, f)
                f.write("\n")

        precision_results = precision.compute(predictions=predictions, references=labels)
        recall_results = recall.compute(predictions=predictions, references=labels)

        f_results = compute_f_scores(precision_results["precision"], recall_results["recall"])
        metrics = precision_results | recall_results | f_results
        print(metrics)

        with open(f'{output_file_path}_metrics.json', 'w') as f:
            json.dump(metrics | {"predictions": predictions}, f)

if __name__ == '__main__':
    main()