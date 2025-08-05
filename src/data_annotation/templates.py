from pydantic import BaseModel


class ExplanationScoreExtraction(BaseModel):
    explanation: str
    score: int

class Criteria(BaseModel):
    name: str
    score: int

class JudgingExtraction(BaseModel):
    comparative_analysis: str
    reasoning_a_scores: list[Criteria]
    reasoning_b_scores: list[Criteria]
    reasoning_c_scores: list[Criteria]
    final_selection: str

system_prompt = """As a cybersecurity expert, analyze the provided "Original" and "Revised" code snippets from a commit, along with the CVE ID, CVE description and commit message.
The "Original" code represents the code before the changes, and the "Revised" code reflects the code after the changes. Your task is to evaluate the changes based on whether they fix any security vulnerabilities.
"""

scoring_template = """Please evaluate the code changes on a scale from 0 to 4 based on the following criteria:
- Score 0: The changes are unrelated to fixing security vulnerabilities.
- Score 1: The changes may involve minor refactoring, but do not focus on fixing security vulnerabilities.
- Score 2: The changes suggest some improvement in code quality, with potential indirect effects on security, but not directly targeting vulnerabilities.
- Score 3: The changes have a moderate focus on addressing security vulnerabilities.
- Score 4: The changes clearly and directly target security vulnerabilities, providing a strong fix.

The length of the code should not influence your evaluation. Focus on the logic of the changes, line by line, and their relevance to vulnerability fixing.
Consider the context provided by other functions in the same commit and the information about the CVE linked to this commit. Think step by step.

### CVE ID:
{cve_id}

### CVE Description:
{cve_description}

### Commit Message:
{commit_message}

### Original code snippet (code before changes):
{original_code}

### Revised code snippet (code after changes):
{revised_code}
"""

judge_system_prompt = """You are an expert in software security and vulnerability detection. Your task is to evaluate the reasoning outputs from three language models for a given code function."""

template_judging = """**Code Function:**
{function}

**Ground-Truth Label:**
{label}

**Reasoning Outputs:**
1. Reasoning A:
{reasoning_a}

2. Reasoning B:
{reasoning_b}

3. Reasoning C:
{reasoning_c}

**Evaluation Instructions:**
1. **Comparative Analysis:** Compare the three reasoning outputs in terms of correctness, completeness, clarity, and actionability. Highlight strengths and weaknesses of each reasoning output relative to one another.
2. **Score Assignment:** After comparing, score each reasoning output on a scale of 1 to 5 for the following criteria:
   - Correctness: Does the reasoning correctly identify if the code is vulnerable or safe? Incorrect predictions should receive low scores.
   - Completeness: Does the reasoning cover the vulnerability mechanism or justify why the code is safe, considering edge cases and attack vectors?
   - Clarity: Is the reasoning logically structured, free of ambiguities, and using precise technical terms?
   - Actionability: Does the reasoning provide actionable insights like highlighting vulnerable lines, suggesting patches, or detailing risks?
3. **Final Selection:** Conclude by selecting the best reasoning output overall, justifying why it stands out based on the comparison and scores.
"""
