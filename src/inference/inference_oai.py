import json
import os
import re

import anthropic
import evaluate
import instructor
from datasets import DatasetDict, load_from_disk
from openai import OpenAI
from pydantic import BaseModel
from src.utils import add_samples
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    set_seed
)

from src.arguments import InferenceArguments, DataArguments


class DetectionOutput(BaseModel):
    """Structured output for detection results."""
    thinking: str
    prediction: str


# Constants
MODELS = {
    "openai": {"gpt-4o", "gpt-4.1"},
    "anthropic": {"claude-opus-4-20250514", "claude-sonnet-4-20250514"}
}

COT_PROMPT_TEMPLATE = """You are an AI assistant expert in software vulnerabilities. You use a chain-of-thought approach to answer queries.
Follow these steps:

1. Think through the problem step by step within the <thinking> tags.
2. Provide your final, concise answer within the <output> tags.

Important: The <thinking> section is for your internal reasoning process only.
Do not include any part of the final answer in this section.
The actual response for the query must be entirely contained within the <output> tags.

Strictly use the following format for your response:
<thinking>
[Your step-by-step reasoning goes here. This is your internal thought process, not the final answer.]
</thinking>
<output>
[Your final, concise answer to the query. This is the only part that will be shown to the user.]
</output>

Analyze the following function and determine whether it contains any vulnerabilities.
Indicate your final decision with:
- YES: the function is vulnerable
- NO: the function is not vulnerable
Output your final decision within the <output> and </output> tags. Strictly follow the output format.

Input function:
```{lang}
{function}
```
"""


def process_output(output_text):
    """Extract prediction from generated output text."""
    output_match = re.search(r'<output>(.*?)(?:</output>|$)', output_text, re.DOTALL)
    extracted = output_match.group(1).strip() if output_match else output_text

    if "NO" in extracted:
        return 0
    elif "YES" in extracted:
        return 1
    else:  # Default to non-vulnerable if unclear
        return 0


def compute_f_scores(precision, recall):
    """Compute F1 and F0.3 scores."""
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

    beta_0_3 = 0.3
    beta_0_3_squared = beta_0_3 ** 2
    f03_score = (1 + beta_0_3_squared) * (precision * recall) / (beta_0_3_squared * precision + recall + 1e-10)

    return {"f1": f1_score, "f03": f03_score}


def initialize_client(model_name):
    """Initialize the appropriate client based on model provider."""
    if model_name in MODELS["openai"]:
        return OpenAI()
    else:
        return instructor.from_anthropic(anthropic.Anthropic())


def prepare_model_kwargs(model_name, messages, args):
    """Prepare keyword arguments based on model type."""
    kwargs = {
        "model": model_name,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_new_tokens,
    }

    if model_name in MODELS["openai"]:
        kwargs["response_format"] = DetectionOutput
        if model_name == "o3":
            del kwargs["temperature"], kwargs["top_p"]
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
    else:
        kwargs["response_model"] = DetectionOutput

    return kwargs


def get_model_response(client, model_name, messages, args):
    kwargs = prepare_model_kwargs(model_name, messages, args)

    if model_name in MODELS["openai"]:
        completion = client.beta.chat.completions.parse(**kwargs)
        output = completion.choices[0].message.parsed
    else:
        output = client.chat.completions.create(**kwargs)

    return output, output


def main():
    parser = HfArgumentParser((InferenceArguments, DataArguments))
    args, data_args = parser.parse_args_into_dataclasses()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize client based on model
    client = initialize_client(args.model_name_or_path)

    # Load and prepare datasets
    dataset = load_from_disk(os.path.join(data_args.data_dir, data_args.dataset_name))["test"]
    test_datasets = DatasetDict({
        f"test_{lang}": dataset.filter(lambda x: x["lang"] == lang, num_proc=data_args.preprocessing_num_workers)
        for lang in data_args.languages
    })

    # Load evaluation metrics
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")

    # Process each language dataset
    for ds_key, dataset in test_datasets.items():
        lang = ds_key.split("_")[-1]

        # Prepare output file names
        output_file = f"{ds_key}_t{args.temperature}_r{args.test_ratio}_s{args.seed}"
        output_file_metrics = f"{output_file}_metrics.json"

        dataset = dataset.select_columns(["function", "vulnerable", "lang"])
        predictions, labels = [], []

        print(f"\nProcessing {ds_key}...")
        with open(os.path.join(args.output_dir, f"{output_file}.jsonl"), "w") as f:
            for sample in tqdm(dataset):
                messages = [{
                    "role": "user",
                    "content": COT_PROMPT_TEMPLATE.format(
                        lang=sample["lang"],
                        function=sample["function"]
                    )
                }]

                # Get model response
                raw_output, parsed_output = get_model_response(
                    client, args.model_name_or_path, messages, args
                )

                # Extract prediction
                if args.model_name_or_path == "codex-mini-latest":
                    prediction = process_output(raw_output)
                    reasoning = raw_output
                else:
                    prediction = 1 if "YES" in parsed_output.prediction else 0
                    reasoning = parsed_output.thinking

                predictions.append(prediction)
                labels.append(sample["vulnerable"])

                # Save output
                output_dict = {
                    "prediction": prediction,
                    "reasoning": reasoning,
                }
                json.dump(output_dict, f)
                f.write("\n")

        # Compute metrics
        precision_results = precision.compute(predictions=predictions, references=labels)
        recall_results = recall.compute(predictions=predictions, references=labels)
        f_results = compute_f_scores(precision_results["precision"], recall_results["recall"])

        metrics = {
            "precision": precision_results["precision"],
            "recall": recall_results["recall"],
            "f1": f_results["f1"],
            "f03": f_results["f03"],
            "predictions": predictions,
            "labels": labels,
        }

        # Print summary
        print(f"\nMetrics for {ds_key}:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"F0.3: {metrics['f03']:.4f}")

        # Save metrics
        with open(os.path.join(args.output_dir, output_file_metrics), "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()