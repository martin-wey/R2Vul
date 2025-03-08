import functools
import json
import os

import evaluate
import torch
from datasets import load_from_disk, DatasetDict
from peft import AutoPeftModelForSequenceClassification
from torch.utils.data import DataLoader
from transformers import set_seed, AutoTokenizer, HfArgumentParser, AutoModelForSequenceClassification

from src.arguments import InferenceArguments, DataArguments
from src.utils import add_samples


def main():
    parser = HfArgumentParser((InferenceArguments, DataArguments))
    args, data_args, probe_args = parser.parse_args_into_dataclasses()
    set_seed(42)

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if "Qwen" in args.model_name_or_path:
        model = AutoPeftModelForSequenceClassification.from_pretrained(args.model_name_or_path)
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
    model = model.to('cuda')
    model.eval()

    dataset = load_from_disk(os.path.join(data_args.data_dir, data_args.dataset_name))["test"]
    dataset = dataset.filter(lambda x: x["lang"] == data_args.languages[0], num_proc=data_args.preprocessing_num_workers)
    if args.test_ratio != "1:1":
        datasets = DatasetDict({
            f"test_{data_args.languages[0]}": dataset
        })
        datasets = add_samples(datasets, args.test_ratio)
        dataset = datasets[f"test_{data_args.languages[0]}"]
    column_names = list(dataset.features)

    def tokenize_examples(examples):
        tokenized_inputs = tokenizer(examples['function'], truncation=True, max_length=tokenizer.model_max_length)
        tokenized_inputs['labels'] = examples['vulnerable']
        return tokenized_inputs

    tokenized_ds = dataset.map(
        tokenize_examples,
        batched=True,
        remove_columns=column_names,
        num_proc=data_args.preprocessing_num_workers,
    )
    tokenized_ds = tokenized_ds.with_format('torch')

    def collate_fn(batch, tokenizer):
        dict_keys = ['input_ids', 'attention_mask', 'labels']
        d = {k: [dic[k] for dic in batch] for k in dict_keys}
        d['input_ids'] = torch.nn.utils.rnn.pad_sequence(
            d['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id
        )
        d['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
            d['attention_mask'], batch_first=True, padding_value=0
        )
        d['labels'] = torch.stack(d['labels'])
        return d

    precision = evaluate.load("precision")
    recall = evaluate.load("recall")

    def compute_f_scores(precision, recall):
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

        beta_0_3 = 0.3
        beta_0_3_squared = beta_0_3 ** 2
        f03_score = (1 + beta_0_3_squared) * (precision * recall) / (beta_0_3_squared * precision + recall + 1e-10)

        return {'f1': f1_score, 'f03': f03_score}

    test_dataloader = DataLoader(
        dataset=tokenized_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=functools.partial(collate_fn, tokenizer=tokenizer),
    )

    predictions = []
    labels = []
    for batch in test_dataloader:
        batch = {k: v.to('cuda') for k, v in batch.items()}
        with torch.no_grad():
            logits = model(**batch).logits
            preds = logits.argmax(dim=1).tolist()
            predictions.extend(preds)
            labels.extend(batch['labels'].tolist())

    print(f"predictions: {predictions}")
    print(f"labels: {labels}")

    precision_results = precision.compute(predictions=predictions, references=labels)
    recall_results = recall.compute(predictions=predictions, references=labels)

    f_results = compute_f_scores(precision_results['precision'], recall_results['recall'])

    metrics = precision_results | recall_results | f_results
    print(metrics)

    file_name = data_args.languages[0]
    if args.test_ratio != "1:1":
        file_name += f"_{args.test_ratio}"
    output_file = os.path.join(
        args.output_dir,
        f'{file_name}_metrics.json'
    )
    with open(output_file, "w") as f:
        json.dump(metrics | {"predictions": predictions}, f)


if __name__ == '__main__':
    main()
