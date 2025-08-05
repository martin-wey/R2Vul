import json
import sys

import anthropic
import instructor
from openai import OpenAI
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

from .templates import judge_system_prompt, template_judging, JudgingExtraction

if __name__ == "__main__":
    model = sys.argv[1]
    if model not in ["openai", "anthropic"]:
        assert False, "Only openai and anthropic are supported"

    with open("data/data_reasoning_labeling.jsonl", "r") as f:
        data = [json.loads(l) for l in f]

    if model == "openai":
        client = OpenAI()
    else:
        client = instructor.from_anthropic(
            anthropic.Anthropic(),
        )
    with open(f"data/data_reasoning_labeling_annotated_{model}.jsonl", "w") as fout:
        with Progress(
                TextColumn(
                    f"Generating judgements •" + "[progress.percentage]{task.percentage:>3.0f}%"
                ),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
        ) as p:
            for sample in p.track(data, total=len(data)):
                function = sample["function"]
                label = "Vulnerable" if sample["vulnerable"] else "Non-vulnerable"
                reasoning_a = sample["reasoning_r2vul"]["response"]
                reasoning_b = sample["reasoning_cot-reflection"]["response"]
                reasoning_c = sample["reasoning_msivd"]["response"]

                input_prompt = template_judging.format(
                    function=function,
                    label=label,
                    reasoning_a=reasoning_a,
                    reasoning_b=reasoning_b,
                    reasoning_c=reasoning_c,
                )

                if model == "openai":
                    completion = client.beta.chat.completions.parse(
                        model="gpt-4o-2024-08-06",
                        messages=[
                            {"role": "system", "content": judge_system_prompt},
                            {"role": "user", "content": input_prompt}
                        ],
                        response_format=JudgingExtraction
                    )
                    output = completion.choices[0].message.parsed
                else:
                    output = client.chat.completions.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=1024,
                        messages=[
                            {"role": "system", "content": judge_system_prompt},
                            {"role": "user", "content": input_prompt}
                        ],
                        response_model=JudgingExtraction
                    )

                json.dump({
                    "comparative_analysis": output.comparative_analysis,
                    "reasoning_a_scores": {c.name: c.score for c in output.reasoning_a_scores},
                    "reasoning_b_scores": {c.name: c.score for c in output.reasoning_b_scores},
                    "reasoning_c_scores": {c.name: c.score for c in output.reasoning_c_scores},
                    "final_selection": output.final_selection,
                }, fout)
                fout.write("\n")
