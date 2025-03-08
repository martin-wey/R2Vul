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
from src.generation_utils import (
    base_generation,
    self_consistency_generation
)
from src.utils import add_samples


def main():
    parser = HfArgumentParser((InferenceArguments, DataArguments))
    args, data_args = parser.parse_args_into_dataclasses()
    set_seed(42)

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    dataset = load_from_disk(os.path.join(data_args.data_dir, data_args.dataset_name))["test"]
    test_datasets = DatasetDict({
        f"test_{lang}": dataset.filter(lambda x: x["lang"] == lang, num_proc=data_args.preprocessing_num_workers)
        for lang in data_args.languages
    })
    # sample extra non-vulnerable functions for class imbalance experiments
    if args.test_ratio != "1:1":
        test_datasets = add_samples(test_datasets, args.test_ratio)

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

    gen_function = self_consistency_generation if args.strategy == "cot-consistency" else base_generation
    for ds_key, dataset in test_datasets.items():
        args.lang = ds_key.split("_")[-1]
        dataset = dataset.select_columns(["function", "vulnerable", "lang"])

        predictions = []
        labels = []
        file_name = f"{ds_key}_{args.strategy}_t{args.temperature}" if args.do_sample else f"{ds_key}_{args.strategy}_greedy"
        if args.test_ratio != "1:1":
            file_name += f"_{args.test_ratio}"
        output_file_path = os.path.join(args.output_dir, file_name)
        with open(f"{output_file_path}.jsonl", "w") as f:
            for sample in tqdm(dataset):
                outputs = gen_function(sample, model, tokenizer, generation_config, args)
                predictions.append(outputs["prediction"])
                labels.append(sample["vulnerable"])
                json.dump(outputs, f)
                f.write('\n')

        precision_results = precision.compute(predictions=predictions, references=labels)
        recall_results = recall.compute(predictions=predictions, references=labels)

        f_results = compute_f_scores(precision_results["precision"], recall_results["recall"])
        metrics = precision_results | recall_results | f_results
        print(metrics)

        with open(f'{output_file_path}_metrics.json', 'w') as f:
            json.dump(metrics | {"predictions": predictions}, f)


if __name__ == '__main__':
    main()