import json

from openai import OpenAI
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn

from .templates import system_prompt, scoring_template, ExplanationScoreExtraction

LANGUAGES = {"c_sharp", "java", "javascript", "python", "cpp"}

if __name__ == "__main__":
    cve_data_file = "data/cve_data.json"
    dataset = "data/paired_dataset.json"
    output_dir = "data"

    with open(cve_data_file, "r") as f:
        cve_data = json.load(f)

    with open(dataset, "r") as f:
        dataset = json.load(f)

    client = OpenAI()

    for lang in LANGUAGES:
        dataset = list(filter(lambda e: e[0]["lang"] == lang, dataset))
        data = []
        with open(f"data/paired_dataset_{lang}_annotated.json", "w") as f:
            with Progress(
                    TextColumn(
                        f"Generating annotations •" + "[progress.percentage]{task.percentage:>3.0f}%"
                    ),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TextColumn("•"),
                    TimeElapsedColumn(),
            ) as p:
                for function_before, function_after in p.track(dataset, total=len(dataset)):
                    cve_id = function_before["cve_id"]
                    cve_description = cve_data[cve_id]["description"]
                    commit_message = function_before["commit_message"]
                    original_code = function_before["function"]
                    revised_code = function_after["function"]

                    user_prompt = scoring_template.format(
                        cve_id=cve_id,
                        cve_description=cve_description,
                        commit_message=commit_message,
                        original_code=original_code,
                        revised_code=revised_code
                    )

                    completion = client.beta.chat.completions.parse(
                        model="gpt-4o-2024-08-06",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        response_format=ExplanationScoreExtraction
                    )
                    output = completion.choices[0].message.parsed

                    json.dump({
                        "pre": function_before,
                        "post": function_after,
                        "explanation": output.explanation,
                        "score": output.score
                    }, f)
                    f.write("\n")
