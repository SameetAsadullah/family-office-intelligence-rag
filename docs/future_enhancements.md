# Future Enhancements

The current system is designed for a focused 50-record family-office intelligence corpus. The architecture leaves clear room for deeper validation and scale-oriented upgrades:

- Add claim-level evidence rows that map each investment, AUM, contact, or activity claim to a specific source URL and extraction note.
- Add exact entity lookup before semantic search for direct questions about a named family office.
- Expand golden-query evaluation with expected entities, expected source domains, refusal checks, and manual unsupported-claim review.
- Move chat history and vector storage to managed services for authenticated multi-user deployments.
- Add an operator-only ingestion job or CI workflow that validates the workbook, rebuilds Chroma, runs golden queries, and emits a data-quality report.
- Add optional retrieval caching after the core retrieval/evaluation loop is stable, with index-signature invalidation to keep cached evidence aligned with the active dataset.
