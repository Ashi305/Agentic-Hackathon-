# Person 2 — Safety Report Analysis Agent

## Overview

This module is the **LLM/Agent reasoning layer** of the Safety Report Analysis Agent project.

The overall project analyzes workplace safety reports written in free text and tries to identify early warning signs of serious incidents.

Person 2's responsibility is to take an individual safety report and use an LLM to:

1. Extract structured information from the report
2. Identify hazards and risk factors
3. Identify possible consequences
4. Detect information that is missing from the report
5. Classify the report as `LOW`, `MEDIUM`, or `HIGH` risk
6. Provide a confidence score for the classification
7. Explain why the report received that risk level
8. Detect cases where the model is uncertain and should be reviewed
9. Learn from previous safety-officer corrections through dynamic few-shot examples
10. Retrieve relevant previous corrections when making a new classification

The module is designed to work with the rest of the team project through a simple structured JSON interface.

---

# 1. Role of Person 2

The project is divided into several responsibilities.

### Person 1 — Data Pipeline

Responsible for:

- Safety report dataset
- Synthetic report generation
- Central data schema
- Storage of reports
- Storage of safety-officer corrections

### Person 2 — LLM / Agent Engine

Responsible for:

- LLM prompts
- Information extraction
- Risk classification
- Confidence scoring
- Hazard/consequence reasoning
- Missing-information detection
- Human-review flagging
- Few-shot learning from officer corrections
- Retrieval of relevant previous corrections

**This repository contains Person 2's implementation.**

### Person 3 — Pattern Analytics

Responsible for analyzing multiple reports together.

Examples:

- Recurring hazards
- Recurring risk factors
- Location hotspots
- Risk-level distribution
- Department-level patterns
- Frequently occurring themes

Person 2 does **not** perform cross-report statistical analysis.

### Person 4 — Dashboard / Frontend

Responsible for:

- Streamlit interface
- Report input/upload
- Displaying the analysis
- Risk visualization
- Safety-officer override interface
- Sending officer corrections back to the backend

---

# 2. Overall Architecture

The Person 2 pipeline is:

```text
                    Safety Report
                         |
                         v
              +---------------------+
              |  Information        |
              |  Extraction         |
              +---------------------+
                         |
                         v
              Structured Extraction
                         |
                         v
              +---------------------+
              | Previous Officer    |
              | Corrections         |
              | Retrieval           |
              +---------------------+
                         |
                         v
              +---------------------+
              | Risk Classification |
              | LLM                 |
              +---------------------+
                         |
                         v
              +---------------------+
              | Risk Level          |
              | Confidence          |
              | Reason              |
              | Review Flag         |
              +---------------------+
                         |
                         v
                 Safety Officer
                    Review
                         |
                  +------+------+
                  |             |
                Accept        Override
                  |             |
                  |       +-----------+
                  |       | New Label |
                  |       | + Reason   |
                  |       +-----------+
                  |             |
                  |             v
                  |       Store Correction
                  |             |
                  |             v
                  +------> Future Few-Shot
                           Examples