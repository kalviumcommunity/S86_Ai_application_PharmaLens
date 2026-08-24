# Chunking Strategy Comparison

Comparison document: `noisy_clinical_report.txt` (cleaned text)

| Strategy | Chunk count | Average size (characters) |
| --- | ---: | ---: |
| Paragraph-aware (max 220 characters) | 3 | 114.3 |
| Fixed-size (180 characters, 30-character overlap) | 3 | 135.0 |

## Choice

Paragraph-aware chunks are the recommended strategy for this corpus. The source files contain short, semantically complete clinical paragraphs and eligibility bullets, so keeping those units intact gives retrieval useful context and avoids splitting a finding or criterion mid-sentence. The fixed-size baseline provides predictable capacity and overlap, but its samples split sentences and can duplicate fragments across retrieval results. The 220-character limit keeps chunks small enough for economical embedding while allowing related sentences to stay together.

## Sample Chunks

Samples use the first three chunks from each strategy so reviewers can inspect boundaries.

### Paragraph-aware (max 220 characters)

**noisy_clinical_report.txt / chunk 1** (118 characters)

> The clinical trial evaluated Drug Y in adult participants. The primary objective was to evaluate the safety of Drug Y.

**noisy_clinical_report.txt / chunk 2** (112 characters)

> Patients receiving Drug Y were monitored for adverse events. Common adverse events included headache and nausea.

**noisy_clinical_report.txt / chunk 3** (113 characters)

> The study also evaluated treatment response and tolerability. Most reported adverse events were mild to moderate.


### Fixed-size (180 characters, 30-character overlap)

**noisy_clinical_report.txt / chunk 1** (180 characters)

> The clinical trial evaluated Drug Y in adult participants. The primary objective was to evaluate the safety of Drug Y. Patients receiving Drug Y were monitored for adverse events. 

**noisy_clinical_report.txt / chunk 2** (180 characters)

> monitored for adverse events. Common adverse events included headache and nausea. The study also evaluated treatment response and tolerability. Most reported adverse events were mi

**noisy_clinical_report.txt / chunk 3** (45 characters)

> eported adverse events were mild to moderate.
