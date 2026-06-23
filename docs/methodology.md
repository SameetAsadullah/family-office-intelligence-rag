# Dataset And Retrieval Methodology

The dataset is represented as a validated multi-sheet workbook of 50 family office records. The RAG application treats the workbook as the source of truth at query time, which keeps retrieval reproducible and separates data validation from answer generation.

## Workbook Design

The workbook separates structured records, activity evidence, source logging, RAG-ready documents, data dictionary definitions, and QA checks:

- `Family_Offices` stores clean business-facing entity attributes and filterable metadata.
- `Recent_Activities` stores recent investment or market signals.
- `Source_Log` preserves evidence URLs, claim support, and reliability notes.
- `Record_Validation` preserves validation status, confidence scoring, completeness scoring, and processing notes outside the main entity table.
- `RAG_Documents` stores retrieval-ready text and metadata.
- `Data_Dictionary` documents field definitions and derivations.
- `QA_Checks` records validation counts and quality checks.

## Retrieval Layer

Only `RAG_Documents` is embedded. Each row becomes one LangChain document with normalized metadata and source URLs, preserving a direct link between the retrieval unit, the family office record, and its supporting evidence.

## Evidence Layer

Source URLs are stored with every vector document. During answer generation, retrieved family-office records include user-facing profile details and compact source IDs. The Streamlit app renders source chips deterministically after generation.

## Source-Aware Retrieval

The app keeps validation and confidence fields as internal workbook metadata, while the user-facing UI presents business filters and concise answers. The answer layer uses grounded refusal behavior when available records do not support a question.
