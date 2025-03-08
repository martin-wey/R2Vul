import os

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import set_seed, HfArgumentParser, BitsAndBytesConfig, AutoTokenizer, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer, get_kbit_device_map, DataCollatorForCompletionOnlyLM

from src.arguments import ModelArguments, DataArguments
from src.templates import cot_system_prompt, prompt_template


def main():
    parser = HfArgumentParser((SFTConfig, ModelArguments, DataArguments))
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True, use_fast=True
    )
    training_args.max_seq_length = tokenizer.model_max_length

    quantization_config = None
    if model_args.use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    model_kwargs = dict(
        trust_remote_code=True,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch.bfloat16,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_from_disk(os.path.join(data_args.data_dir, data_args.dataset_name))
    if data_args.languages[0] != "all":
        dataset = dataset.filter(lambda x: x["lang"] in data_args.languages, num_proc=data_args.preprocessing_num_workers)
    del dataset["test"]
    column_names = list(dataset["train"].features)

    def apply_chat_template(example):
        assistant_output = "{thought}\n<output>\n{output}\n</output>"
        messages = [
            {"role": "system", "content": cot_system_prompt},
            {"role": "user", "content": prompt_template.format(function=example["function"],
                                                               lang=example["lang"].replace("_", ""))},
            {"role": "assistant", "content": assistant_output.format(thought=example["reasoning"],
                                                                     output="YES" if example["vulnerable"] else "NO")}
        ]
        example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
        return example

    dataset = dataset.map(
        apply_chat_template,
        remove_columns=column_names,
        num_proc=data_args.preprocessing_num_workers,
        desc="Applying chat template"
    )

    collator = None
    if model_args.completion_only:
        response_template = "<|im_start|>assistant"
        collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        peft_config=lora_config,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback()]
    )
    trainer.train()

    best_model_dir = os.path.join(training_args.output_dir, "best_model_checkpoint")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f"Best model saved to {best_model_dir}")


if __name__ == "__main__":
    main()
