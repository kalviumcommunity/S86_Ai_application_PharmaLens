from __future__ import annotations

import json
from pathlib import Path

from src.config import load_settings
from src.llm_client import create_client


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "structured_output.log"


# ============================================================
# JSON PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are PharmaLens, a pharmaceutical research assistant.

Your response MUST be exactly one valid JSON object.

Use this exact structure:

{
  "answer": "string",
  "source": "string"
}

Rules:
- Return JSON only.
- Do not use markdown.
- Do not add explanations outside the JSON.
- Include both required fields.
- Both fields must contain non-empty strings.
- Do not invent study results or citations.
""".strip()


USER_PROMPT = (
    "In one short sentence, explain what a clinical trial is. "
    "Use 'Clinical Research Report' as the source."
)


REQUIRED_FIELDS = ("answer", "source")


# ============================================================
# JSON PARSING AND VALIDATION
# ============================================================

def parse_and_validate(
    raw: str,
    required_fields: tuple[str, ...] = REQUIRED_FIELDS,
):
    """
    Parse JSON and validate the required fields.

    Returns:
        (data, None) when valid.
        (None, error_message) when invalid.
    """

    try:
        data = json.loads(raw)

    except json.JSONDecodeError as error:
        return None, f"malformed JSON: {error.msg}"

    if not isinstance(data, dict):
        return None, "invalid JSON: expected a JSON object"

    missing_fields = [
        field
        for field in required_fields
        if field not in data
    ]

    if missing_fields:
        return None, (
            "missing required fields: "
            + ", ".join(missing_fields)
        )

    for field in required_fields:

        if not isinstance(data[field], str):
            return None, (
                f"invalid field type: '{field}' "
                "must be a string"
            )

        if not data[field].strip():
            return None, (
                f"invalid field value: '{field}' "
                "must not be empty"
            )

    return data, None


# ============================================================
# FIRST STRUCTURED REQUEST
# ============================================================

def request_structured_output(
    client,
    model: str,
    user_prompt: str,
):
    """
    Ask the model to return a structured JSON response.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0,
        max_tokens=300,
        response_format={
            "type": "json_object"
        },
    )

    return response


# ============================================================
# RECOVERY REQUEST
# ============================================================

def recover_from_malformed_json(
    client,
    model: str,
    bad_output: str,
):
    """
    Retry once after receiving malformed JSON.
    """

    recovery_prompt = f"""
The previous response was invalid JSON.

Invalid response:
{bad_output}

Return exactly ONE complete JSON object.

Use this exact example structure:

{{
  "answer": "A clinical trial is a research study involving human participants.",
  "source": "Clinical Research Report"
}}

Rules:
- Return JSON only.
- Do not use markdown.
- Do not add explanations.
- Include both answer and source.
- Both values must be complete strings.
- Close all quotation marks.
- Close the JSON object.
- Do not truncate the response.
""".strip()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": recovery_prompt,
            },
        ],
        temperature=0,
        max_tokens=300,
        response_format={
            "type": "json_object"
        },
    )

    return response


# ============================================================
# MAIN
# ============================================================

def main():

    settings = load_settings()

    client = create_client(settings)
    model = settings["chat_model"]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    print(
        "\n========== PHARMALENS STRUCTURED OUTPUT ==========\n"
    )

    print(f"Model: {model}")

    # ========================================================
    # TASK 1 + TASK 2
    # JSON OUTPUT + PARSING
    # ========================================================

    print("\n" + "=" * 70)
    print("TASK 1 + TASK 2 - JSON OUTPUT AND PARSING")
    print("=" * 70)

    response = request_structured_output(
        client=client,
        model=model,
        user_prompt=USER_PROMPT,
    )

    raw_output = response.choices[0].message.content or ""

    print("\nRaw model response:")
    print(raw_output)

    parsed_data, parse_error = parse_and_validate(
        raw_output
    )

    if parse_error:

        print("\nParsing/validation failed:")
        print(parse_error)

        results.append(
            f"""
VALID JSON RESPONSE
-------------------
Raw response:
{raw_output}

Result:
FAILED

Error:
{parse_error}
"""
        )

    else:

        print("\nParsed Python object:")
        print(parsed_data)

        print("\nAnswer:")
        print(parsed_data["answer"])

        print("\nSource:")
        print(parsed_data["source"])

        results.append(
            f"""
VALID JSON RESPONSE
-------------------
Raw response:
{raw_output}

Parsed object:
{parsed_data}

Answer:
{parsed_data["answer"]}

Source:
{parsed_data["source"]}

Result:
SUCCESS
"""
        )

    # ========================================================
    # TASK 3
    # MALFORMED JSON
    # ========================================================

    print("\n" + "=" * 70)
    print("TASK 3 - MALFORMED JSON HANDLING")
    print("=" * 70)

    # Intentionally invalid JSON.
    # The trailing comma makes this malformed.
    malformed_output = (
        '{"answer": "Clinical trials evaluate medical interventions", '
        '"source": "Clinical Research Report",}'
    )

    print("\nSimulated malformed JSON:")
    print(malformed_output)

    malformed_data, malformed_error = parse_and_validate(
        malformed_output
    )

    if malformed_error:

        print("\nMalformed JSON detected:")
        print(malformed_error)

        results.append(
            f"""
MALFORMED JSON TEST
-------------------
Input:
{malformed_output}

Detection:
{malformed_error}

Result:
SUCCESSFULLY DETECTED
"""
        )

    else:

        print("\nUnexpectedly accepted malformed JSON.")

        results.append(
            f"""
MALFORMED JSON TEST
-------------------
Input:
{malformed_output}

Result:
UNEXPECTEDLY ACCEPTED
"""
        )

    # ========================================================
    # TASK 4
    # REQUIRED FIELD VALIDATION
    # ========================================================

    print("\n" + "=" * 70)
    print("TASK 4 - REQUIRED FIELD VALIDATION")
    print("=" * 70)

    # Missing "source" intentionally.
    missing_field_output = json.dumps(
        {
            "answer": (
                "Clinical trials evaluate medical "
                "interventions."
            )
        }
    )

    print("\nJSON with missing source field:")
    print(missing_field_output)

    missing_data, missing_error = parse_and_validate(
        missing_field_output
    )

    if missing_error:

        print("\nValidation correctly rejected the object:")
        print(missing_error)

        results.append(
            f"""
MISSING FIELD TEST
------------------
Input:
{missing_field_output}

Validation result:
REJECTED

Reason:
{missing_error}
"""
        )

    else:

        print("\nERROR: Missing field was not detected.")

        results.append(
            f"""
MISSING FIELD TEST
------------------
Input:
{missing_field_output}

Validation result:
FAILED TO DETECT MISSING FIELD
"""
        )

    # ========================================================
    # TASK 3 + TASK 5
    # MALFORMED JSON RECOVERY
    # ========================================================

    print("\n" + "=" * 70)
    print("TASK 3 - MALFORMED JSON RECOVERY")
    print("=" * 70)

    print("\nRetrying with a strict JSON reminder...")

    recovery_response = recover_from_malformed_json(
        client=client,
        model=model,
        bad_output=malformed_output,
    )

    recovered_raw = (
        recovery_response.choices[0].message.content or ""
    )

    print("\nRecovered raw response:")
    print(recovered_raw)

    recovered_data, recovered_error = parse_and_validate(
        recovered_raw
    )

    if recovered_error:

        print("\nRecovery failed:")
        print(recovered_error)

        results.append(
            f"""
RECOVERY TEST
-------------
Original malformed input:
{malformed_output}

Recovered response:
{recovered_raw}

Result:
FAILED

Error:
{recovered_error}
"""
        )

    else:

        print("\nRecovery successful!")

        print("\nParsed object:")
        print(recovered_data)

        print("\nRecovered answer:")
        print(recovered_data["answer"])

        print("\nRecovered source:")
        print(recovered_data["source"])

        results.append(
            f"""
RECOVERY TEST
-------------
Original malformed input:
{malformed_output}

Recovered response:
{recovered_raw}

Result:
SUCCESS

Parsed object:
{recovered_data}

Answer:
{recovered_data["answer"]}

Source:
{recovered_data["source"]}
"""
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    summary = """
STRUCTURED OUTPUT VALIDATION SUMMARY
====================================

Required JSON structure:

{
  "answer": "string",
  "source": "string"
}

Implemented:

1. The model is explicitly instructed to return JSON.
2. JSON response mode is used.
3. The response is parsed using json.loads().
4. Malformed JSON is detected safely.
5. Required fields are validated.
6. Missing fields are rejected.
7. A malformed JSON response is retried once.
8. The recovered response is parsed and validated.
9. Sample results are saved for review.

Recommended PharmaLens response contract:

{
  "answer": "The study reported headache and nausea.",
  "source": "Clinical Trial Report - Study ABC - Page 47"
}

This predictable structure allows the backend and frontend
to reliably handle answers and citations.
"""

    print("\n" + "=" * 70)
    print(summary)

    results.append(summary)

    # ========================================================
    # SAVE SAMPLE RESULTS
    # ========================================================

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "PHARMALENS - STRUCTURED OUTPUT & JSON HANDLING\n"
        )

        file.write("=" * 70)

        file.write(
            f"\n\nMODEL: {model}\n"
        )

        for result in results:

            file.write("\n")
            file.write(result)
            file.write("\n")

    print(
        f"\nSample results saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()