# MemInterfere: When Memory Misleads

**When Memory Misleads: Skill Library Interference and Reliability Degradation in Tool-Calling Agents**

## Research Question

Does retrieving from a skill library *decrease* agent reliability when skills conflict, are contextually inappropriate, or inflate confidence without improving accuracy?

Every existing paper shows memory *helps* agents. Nobody systematically studies **when and how it hurts**. This is the "retrieval can hurt" finding that Self-RAG demonstrated for RAG, but applied to agent skill libraries — which nobody has done.

## Core Claim

Retrieving from a skill library can decrease agent reliability through three mechanisms:
1. **Skill conflict**: Two retrieved skills give contradictory advice for similar situations
2. **Contextual mismatch**: Retrieved skills are correct for a different context than the current one
3. **Confidence inflation**: Retrieval increases agent confidence without improving accuracy

## Research Infrastructure

This research assumes a setup similar to ours:
- **Hermes agent**: Tool-calling agent with persistent memory (ByteRover), session search, skill libraries, and cron pipelines
- **ByteRover**: Hierarchical persistent knowledge tree with curation
- **Skills**: Markdown-based procedural memory loaded per task
- **Session search**: Full conversation history across sessions

We aim to improve this setup by identifying and mitigating memory interference.

## Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Literature review + experimental design | 🔄 In progress |
| 2 | Skill library with controlled interference | Pending |
| 3 | Multi-condition agent evaluation | Pending |
| 4 | Interference analysis + mitigation testing | Pending |
| 5 | Paper draft + Grok review | Pending |
| 6 | Revisions + submission | Pending |

## Project Structure

```
meminterfere/
├── README.md                    # This file
├── src/
│   ├── interference_library.py  # Skill library with known interference
│   ├── evaluate_agent.py       # Run agent under controlled conditions
│   ├── metrics.py               # Success rate, calibration, error types
│   └── mitigations.py           # Metadata retrieval, consistency checking
├── data/
│   ├── skills/                  # Interference skill library
│   ├── tasks/                   # Evaluation tasks
│   └── results/                 # Experiment results
├── docs/
│   ├── literature-review.md     # Phase 1 output
│   ├── experimental-design.md   # Phase 1 output
│   └── paper-draft.md           # Phase 5 output
└── tests/
    └── test_interference.py     # Unit tests
```

## Citation

```bibtex
@article{tyler2026meminterfere,
  title={When Memory Misleads: Skill Library Interference and Reliability Degradation in Tool-Calling Agents},
  author={Tyler, David},
  year={2026},
  note={Research in progress}
}
```

## License

MIT