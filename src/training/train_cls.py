import functools
import os

import evaluate
import numpy as np
import torch
from datasets import load_from_disk
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
    HfArgumentParser,
    Trainer,
    EarlyStoppingCallback
)

from src.arguments import ModelArguments, DataArguments


def load_model_and_tokenizer(model_args):
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, use_fast=True)

    if "codebert" in model_args.model_name_or_path:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            num_labels=2,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        quantization_config = None
        if model_args.use_qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        model_kwargs = {
            "num_labels": 2,
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "quantization_config": quantization_config,
            "device_map": "auto"
        }

        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules='all-linear',
            lora_dropout=model_args.lora_dropout,
            bias='none',
            task_type='SEQ_CLS',
            modules_to_save=["score"],
        )

        model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, **model_kwargs)
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def main():
    parser = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments))
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()
    set_seed(42)

    dataset = load_from_disk(os.path.join(data_args.data_dir, data_args.dataset_name))
    if data_args.languages[0] != "all":
        dataset = dataset.filter(lambda x: x["lang"] in data_args.languages, num_proc=data_args.preprocessing_num_workers)
    model, tokenizer = load_model_and_tokenizer(model_args)

    def tokenize_examples(examples):
        tokenized_inputs = tokenizer(examples['function'], truncation=True, max_length=tokenizer.model_max_length)
        tokenized_inputs['labels'] = examples['vulnerable']
        return tokenized_inputs

    tokenized_ds = dataset.map(
        tokenize_examples,
        batched=True,
        remove_columns=[c for c in dataset['train'].column_names if c not in ["input_ids", "attention_mask", "labels"]]
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
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        precision_results = precision.compute(predictions=predictions, references=labels)
        recall_results = recall.compute(predictions=predictions, references=labels)
        f1_results = f1.compute(predictions=predictions, references=labels)

        return precision_results | recall_results | f1_results

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['validation'],
        tokenizer=tokenizer,
        data_collator=functools.partial(collate_fn, tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback()]
    )
    trainer.train()

    best_model_dir = os.path.join(training_args.output_dir, "best_model_checkpoint")
    model.save_pretrained(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f"Best model saved to {best_model_dir}")


if __name__ == "__main__":
    main()
