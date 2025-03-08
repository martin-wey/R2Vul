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
    AutoModelForCausalLM
)

from src.arguments import InferenceArguments, DataArguments

prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Detect whether the following code contains vulnerabilities.

### Input:
{code}

### Response:
"""


def main():
    parser = HfArgumentParser((InferenceArguments, DataArguments))
    args, data_args = parser.parse_args_into_dataclasses()
    set_seed(42)

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-13b-hf", padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        pad_token_id=tokenizer.eos_token_id
    )
    dataset = load_from_disk(os.path.join(data_args.data_dir, data_args.dataset_name))["test"]
    test_datasets = DatasetDict({
        f"test_{lang}": dataset.filter(lambda x: x["lang"] == lang, num_proc=data_args.preprocessing_num_workers)
        for lang in data_args.languages
    })

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
                prediction_result = 0
                inputs = tokenizer(prompt.format(code=sample["function"]), return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                outputs = model.generate(
                    inputs=inputs["input_ids"],
                    max_new_tokens=100
                )
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                decoded_sequence = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if len(decoded_sequence) > 0 and decoded_sequence[0] == "1":
                    prediction_result = 1

                predictions.append(prediction_result)
                labels.append(sample["vulnerable"])

                json.dump({
                    "response": decoded_sequence,
                    "prediction": prediction_result,
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
