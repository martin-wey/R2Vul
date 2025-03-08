import re

from .templates import (
    base_system_prompt,
    cot_system_prompt,
    cot_reflection_system_prompt,
    cot_reflection_contrastive_system_prompt,
    prompt_template
)


def get_system_prompt_for_strategy(strategy):
    generation_strategies = {
        'zero-shot': base_system_prompt,
        'cot': cot_system_prompt,
        'cot-consistency': cot_system_prompt,
        'cot-reflection': cot_reflection_system_prompt,
        'cot-contrastive': cot_reflection_contrastive_system_prompt
    }
    return generation_strategies.get(strategy, base_system_prompt)


def generation(conversation, tokenizer, model, generation_kwargs):
    inputs = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors='pt',
        return_dict=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs =  model.generate(**inputs, **generation_kwargs)
    return outputs[:, inputs['input_ids'].shape[1]:]


def base_generation(sample, model, tokenizer, generation_kwargs, args):
    system_prompt = get_system_prompt_for_strategy(args.strategy)
    messages = [
        {'role': 'system', 'content': system_prompt},
        {
            'role': 'user',
            'content': prompt_template.format(
                function=sample['function'],
                lang=sample["lang"]
            )
        },
    ]
    outputs = generation(messages, tokenizer, model, generation_kwargs)

    decoded_sequence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prediction = postprocess_response(decoded_sequence, args.strategy)

    return {
        "response": decoded_sequence,
        "prediction": prediction
    }


def self_consistency_generation(sample, model, tokenizer, generation_kwargs, args):
    system_prompt = get_system_prompt_for_strategy(args.strategy)
    messages = [
        {'role': 'system', 'content': system_prompt},
        {
            'role': 'user',
            'content': prompt_template.format(
                function=sample['function'],
                lang=sample["lang"]
            )
        },
    ]
    generation_kwargs["num_return_sequences"] = args.num_return_sequences
    outputs = generation(messages, tokenizer, model, generation_kwargs)
    decoded_sequences = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    labels = [postprocess_response(s, args.strategy) for s in decoded_sequences]
    prediction = 1 if sum(labels) > args.num_return_sequences // 2 else 0  # majority vote

    return {
        "responses": decoded_sequences,
        "labels": labels,
        "prediction": prediction
    }


def postprocess_response(response, strategy):
    if strategy == 'zero-shot':
        if "NO" in response:
            return 0
        elif "YES" in response:
            return 1
        else:
            return 0
    else:
        output_match = re.search(r'<output>(.*?)(?:</output>|$)', response, re.DOTALL)
        output = output_match.group(1).strip() if output_match else response
        if "NO" in output:
            return 0
        elif "YES" in output:
            return 1
        else:
            # else we do not know ? not ideal.
            return 0