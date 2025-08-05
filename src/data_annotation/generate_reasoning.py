import json
import os

import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM
)

from src.arguments import InferenceArguments, DataArguments
from src.templates import (
    reasoning_generation_system_prompt,
    reasoning_generation_structured_template_vulnerable,
    reasoning_generation_structured_template_non_vulnerable
)
from src.data_utils import load_cve_data

TEMPLATE_MAPPING = {
    "vulnerable": reasoning_generation_structured_template_vulnerable,
    "non-vulnerable": reasoning_generation_structured_template_non_vulnerable
}


def main():
    parser = HfArgumentParser((InferenceArguments, DataArguments))
    args, data_args = parser.parse_args_into_dataclasses()
    set_seed(42)

    os.makedirs(args.output_dir, exist_ok=True)

    generation_config = {
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        attn_implementation='flash_attention_2',
    )

    dataset = load_from_disk(os.path.join(data_args.data_dir, data_args.languages[0]))
    cve_data = load_cve_data()

    for split, dataset in dataset.items():
        output_file_name = os.path.join(args.output_dir, f'r2vul_{data_args.languages[0]}_{split}_reasoning.jsonl')
        with open(output_file_name, 'w') as f:
            for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
                is_vulnerable = sample['vulnerable'] != data_args.reverse_labels

                template_key = 'vulnerable' if is_vulnerable else 'non-vulnerable'
                template = TEMPLATE_MAPPING[template_key]

                template_kwargs = {
                    'function': sample['function'],
                    'lang': sample['lang']
                }
                if is_vulnerable:
                    template_kwargs.update({
                        'cwe_list': " ".join(sample['cwe_id']),
                        'cve_id': sample['cve_id'],
                        'cve_desc': cve_data[sample['cve_id']]['description']
                    })

                if "gemini" in args.model_name_or_path:
                    completed = False
                    while not completed:
                        try:
                            chat_session = model.start_chat(history=[])
                            response = chat_session.send_message(template.format(**template_kwargs)).text
                            completed = True
                        except Exception:
                            print("Quota exceeded.")
                else:
                    messages = [
                        {'role': 'system', 'content': reasoning_generation_system_prompt},
                        {'role': 'user', 'content': template.format(**template_kwargs)},
                    ]
                    inputs = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors='pt',
                        return_dict=True
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model.generate(**inputs, **generation_config)
                    response = tokenizer.decode(outputs[:, inputs['input_ids'].shape[1]:][0], skip_special_tokens=True)
                json.dump({"response": response}, f)
                f.write('\n')


if __name__ == "__main__":
    main()
