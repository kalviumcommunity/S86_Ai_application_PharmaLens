"""
PharmaLens Usage Monitoring Report
"""

import json
from pathlib import Path
from collections import Counter


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "outputs"

LOG_FILE = (
    OUTPUT_DIR
    / "rag_requests.jsonl"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "usage_summary.json"
)


# ============================================================
# LOAD LOG RECORDS
# ============================================================

def load_log_records():
    """
    Load request records from JSONL logs.
    """

    if not LOG_FILE.exists():

        return []

    records = []

    with open(
        LOG_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(
                    line
                )

                records.append(
                    record
                )

            except json.JSONDecodeError:

                continue

    return records


# ============================================================
# SUMMARIZE USAGE
# ============================================================

def summarize_usage(
    records
):
    """
    Create a simple usage monitoring summary.
    """

    total_requests = len(
        records
    )

    if total_requests == 0:

        return {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_hit_rate": 0,
            "total_estimated_cost": 0,
            "average_latency_ms": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "errors": 0,
            "most_common_questions": [],
        }


    # ========================================================
    # CACHE
    # ========================================================

    cache_hits = sum(
        1
        for record in records
        if record.get(
            "cache_hit"
        )
    )

    cache_misses = (
        total_requests
        - cache_hits
    )

    cache_hit_rate = (
        cache_hits
        / total_requests
    )


    # ========================================================
    # COST
    # ========================================================

    total_cost = sum(
        record.get(
            "estimated_cost",
            0,
        )
        for record in records
    )


    # ========================================================
    # LATENCY
    # ========================================================

    average_latency = sum(
        record.get(
            "latency_ms",
            0,
        )
        for record in records
    ) / total_requests


    # ========================================================
    # TOKENS
    # ========================================================

    total_input_tokens = sum(
        record.get(
            "input_tokens",
            0,
        )
        for record in records
    )

    total_output_tokens = sum(
        record.get(
            "output_tokens",
            0,
        )
        for record in records
    )


    # ========================================================
    # ERRORS
    # ========================================================

    errors = sum(
        1
        for record in records
        if record.get(
            "error"
        )
    )


    # ========================================================
    # COMMON QUESTIONS
    # ========================================================

    question_counter = Counter(
        record.get(
            "question",
            ""
        )
        for record in records
    )

    most_common_questions = [
        {
            "question": question,
            "count": count,
        }
        for question, count
        in question_counter.most_common(
            5
        )
    ]


    # ========================================================
    # RETURN SUMMARY
    # ========================================================

    return {

        "total_requests":
            total_requests,

        "cache_hits":
            cache_hits,

        "cache_misses":
            cache_misses,

        "cache_hit_rate":
            round(
                cache_hit_rate,
                2,
            ),

        "total_input_tokens":
            total_input_tokens,

        "total_output_tokens":
            total_output_tokens,

        "total_estimated_cost":
            round(
                total_cost,
                8,
            ),

        "average_latency_ms":
            round(
                average_latency,
                2,
            ),

        "errors":
            errors,

        "most_common_questions":
            most_common_questions,
    }


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(
    summary
):

    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    summary
):

    print()

    print("=" * 70)

    print(
        "PHARMALENS USAGE MONITORING REPORT"
    )

    print("=" * 70)

    print()

    print(
        "Total Requests:",
        summary[
            "total_requests"
        ],
    )

    print(
        "Cache Hits:",
        summary[
            "cache_hits"
        ],
    )

    print(
        "Cache Misses:",
        summary[
            "cache_misses"
        ],
    )

    print(
        "Cache Hit Rate:",
        summary[
            "cache_hit_rate"
        ],
    )

    print(
        "Total Input Tokens:",
        summary[
            "total_input_tokens"
        ],
    )

    print(
        "Total Output Tokens:",
        summary[
            "total_output_tokens"
        ],
    )

    print(
        "Estimated Cost:",
        summary[
            "total_estimated_cost"
        ],
    )

    print(
        "Average Latency:",
        summary[
            "average_latency_ms"
        ],
        "ms",
    )

    print(
        "Errors:",
        summary[
            "errors"
        ],
    )

    print()

    print(
        "Most Common Questions:"
    )

    print(
        "-" * 70
    )

    for item in summary[
        "most_common_questions"
    ]:

        print(
            f"- {item['question']}"
        )

        print(
            f"  Requests: {item['count']}"
        )


    print()

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    records = (
        load_log_records()
    )

    summary = (
        summarize_usage(
            records
        )
    )

    save_summary(
        summary
    )

    print_summary(
        summary
    )

    print()

    print(
        "Usage summary saved to:"
    )

    print(
        SUMMARY_FILE
    )


if __name__ == "__main__":

    main()