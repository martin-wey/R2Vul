import json
import os
import re

import evaluate
import numpy as np
import requests
from datasets import DatasetDict, load_from_disk
from huggingface_hub import InferenceClient
from src.utils import add_samples
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    set_seed
)

from src.arguments import InferenceArguments, DataArguments
from src.generation_utils import (
    cot_system_prompt,
    prompt_template
)

# Constants for calibration
LEN_V_TEMPLATE_TOKENS = 10
LEN_NV_TEMPLATE_TOKENS = 11


def generate(client, tokenize_input, args, max_tokens=None):
    output = client.text_generation(
        prompt=tokenize_input,
        details=True,
        decoder_input_details=True,
        do_sample=args.do_sample,
        max_new_tokens=max_tokens if max_tokens is not None else args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    return output


def process_output(output):
    output_match = re.search(r'<output>(.*?)(?:</output>|$)', output, re.DOTALL)
    output = output_match.group(1).strip() if output_match else output
    if "NO" in output:
        prediction = 0
    elif "YES" in output:
        prediction = 1
    else:  # else we do not know
        prediction = 0
    return prediction


def compute_f_scores(precision, recall):
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

    beta_0_3 = 0.3
    beta_0_3_squared = beta_0_3 ** 2
    f03_score = (1 + beta_0_3_squared) * (precision * recall) / (beta_0_3_squared * precision + recall + 1e-10)

    return {"f1": f1_score, "f03": f03_score}


def main():
    parser = HfArgumentParser((InferenceArguments, DataArguments))
    args, data_args = parser.parse_args_into_dataclasses()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Setup client
    tgi_model_endpoint = f"http://{args.model_name_or_path}:80"
    client = InferenceClient(
        base_url=tgi_model_endpoint,
    )

    # Load dataset
    dataset = load_from_disk(os.path.join(data_args.data_dir, data_args.dataset_name))["test"]
    test_datasets = DatasetDict({
        f"test_{lang}": dataset.filter(lambda x: x["lang"] == lang, num_proc=data_args.preprocessing_num_workers)
        for lang in data_args.languages
    })

    # Sample extra non-vulnerable functions for class imbalance experiments
    if args.test_ratio != "1:1":
        test_datasets = add_samples(test_datasets, args.test_ratio)

    # Load metrics
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")

    # Process each language dataset
    for ds_key, dataset in test_datasets.items():
        lang = ds_key.split("_")[-1]

        # Prepare output file names
        output_file = f"{ds_key}_t{args.temperature}_r{args.test_ratio}_s{args.seed}"
        output_file_metrics = output_file
        if args.use_calibration:
            output_file += f"_c{args.confidence_threshold}"
            output_file_metrics += f"_c{args.confidence_threshold}"

        dataset = dataset.select_columns(["function", "vulnerable", "lang"])
        predictions, labels, scores = [], [], []
        reflection_predictions = [] if args.use_calibration else None

        with open(os.path.join(args.output_dir, f"{output_file}.jsonl"), "w") as f:
            for sample in tqdm(dataset):
                messages = [
                    {"role": "system", "content": cot_system_prompt},
                    {
                        "role": "user",
                        "content": prompt_template.format(
                            function=sample["function"],
                            lang=sample["lang"]
                        )
                    }
                ]

                if args.use_calibration:
                    # Calibration-based generation
                    response = requests.post(
                        f"{tgi_model_endpoint}/chat_tokenize",
                        headers={"Content-Type": "application/json"},
                        json={"messages": messages}
                    )
                    tokenize_response = response.json()['templated_text']

                    # Compute P(V) - probability of vulnerable
                    positive_prefix = "<thinking>\n1. **Specific Code Constructs**:"
                    output_v = generate(client, tokenize_response + positive_prefix, args, max_tokens=1)
                    prefix_tokens_v = output_v.details.prefill[-LEN_V_TEMPLATE_TOKENS:]
                    sum_logprobs_v = sum([p.logprob for p in prefix_tokens_v])
                    normalized_logprobs_v = sum_logprobs_v / max(1, len(prefix_tokens_v))

                    # Compute P(NV) - probability of non-vulnerable
                    negative_prefix = "<thinking>\n1. **Analysis of Code Safety**:"
                    output_nv = generate(client, tokenize_response + negative_prefix, args, max_tokens=1)
                    prefix_tokens_nv = output_nv.details.prefill[-LEN_NV_TEMPLATE_TOKENS:]
                    sum_logprobs_nv = sum([p.logprob for p in prefix_tokens_nv])
                    normalized_logprobs_nv = sum_logprobs_nv / max(1, len(prefix_tokens_nv))

                    # Calculate confidence
                    logit_margin = normalized_logprobs_v - normalized_logprobs_nv
                    confidence = 1 / (1 + np.exp(-logit_margin))
                    scores.append(confidence.item())

                    if confidence.item() > args.confidence_threshold:
                        # High confidence in vulnerability
                        predictions.append(1)
                        if args.generate_vulnerable:
                            output = generate(client, tokenize_response + positive_prefix, args)
                            generated_response = positive_prefix + output.generated_text
                            output_dict = {
                                "prediction": 1,
                                "reasoning": generated_response,
                                "confidence": confidence.item()
                            }
                            json.dump(output_dict, f)
                            f.write("\n")
                    else:
                        # Low confidence, predict non-vulnerable
                        predictions.append(0)
                        if reflection_predictions is not None:
                            reflection_predictions.append(0)
                        if args.generate_non_vulnerable:
                            output = generate(client, tokenize_response + negative_prefix, args)
                            generated_response = negative_prefix + output.generated_text
                            output_dict = {
                                "prediction": 0,
                                "reasoning": generated_response,
                                "confidence": confidence.item()
                            }
                            json.dump(output_dict, f)
                            f.write("\n")
                else:
                    # Standard generation without calibration
                    output = client.chat_completion(
                        messages=messages,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_new_tokens,
                    ).choices[0].message.content

                    prediction = process_output(output)
                    predictions.append(prediction)

                    output_dict = {
                        "prediction": prediction,
                        "reasoning": output,
                    }
                    json.dump(output_dict, f)
                    f.write("\n")

                labels.append(sample["vulnerable"])

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

        if args.use_calibration:
            metrics["scores"] = scores
            if reflection_predictions:
                metrics["reflection_predictions"] = reflection_predictions

        print(f"\nMetrics for {ds_key}:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"F0.3: {metrics['f03']:.4f}")

        # Save metrics
        with open(os.path.join(args.output_dir, f"{output_file_metrics}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    main()
