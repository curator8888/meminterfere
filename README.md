# MemInterfere: Skill Library Interference in LLM Agents

**Mostly Harmless, Sometimes Costly**

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper/paper.pdf)
[![Release](https://img.shields.io/badge/Release-v1.0-blue)](https://github.com/curator8888/meminterfere/releases/tag/v1.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## TL;DR

Skill library interference (near-duplicate, stale, conflicting descriptions) does **not** significantly degrade LLM tool selection accuracy (1.2pp difference, p=0.26, d=0.06). But it has real costs: **2.33× token increase** and **confidence drops of 0.6-1.8pp**. When errors do occur, 100% redirect to similar-seeming competitors — near-identical skill names are the real danger.

## Key Findings

| Finding | Detail |
|---------|--------|
| **Null result on accuracy** | 95.8% clean vs 94.6% interference (1.2pp, p=0.26, d=0.06) |
| **TOST equivalence at ±3pp** | p=0.035; Bayes Factor 26.3 favoring null |
| **Ceiling effect** | 83% of tasks show identical outcomes with or without interference |
| **On harder tasks, interference helps** | +4.9pp (p=0.023, not pre-registered) |
| **Errors are systematic** | 100% redirect to similar-seeming competitors, never random |
| **Near-identical names are the danger** | Trap tasks have 2.6× higher error rate |
| **Token cost** | 2.33× increase (4,743 vs 2,032 avg tokens) |
| **RAG solves retrieval** | Top-5 RAG: 92.2% vs gold retrieval 92.5% (0.3pp gap) |
| **Oracle with 1 skill** | 99.8% — the task is retrieval, not invocation |

## Experiments

- **4 models**: Llama 3.1 8B, Grok 3 mini, GPT-4o-mini, Claude Haiku 4.5
- **80 original tasks** + **41 harder tasks** (7.3% keyword cues)
- **5 conditions**: no-memory, oracle, clean-memory, clean+interference, all-memory
- **~12,000 runs** across 6 experiments
- **Total compute cost**: ~$50

## Project Structure

```
meminterfere/
├── paper/                    # LaTeX source and compiled PDF
│   ├── paper.tex
│   ├── paper.pdf
│   ├── references.bib
│   └── figures/              # All 5 figures (PDF + PNG)
├── src/                      # Experiment code
│   ├── interference_library.py   # Skill library with controlled interference
│   ├── evaluate_agent.py         # Run agent under controlled conditions
│   ├── multi_model_runner.py     # Multi-model experiment runner
│   ├── response_parser.py        # Parse model responses
│   ├── metrics.py                # Success rate, calibration, error types
│   ├── phase6_comprehensive_analysis.py  # Publication analysis
│   ├── mixed_effects_analysis.py  # GLMM analysis
│   └── ...                       # Other analysis and utility scripts
├── data/
│   ├── skills/               # Interference skill library (45 skills)
│   ├── tasks/                # 80 original + 41 harder task definitions
│   └── results/              # All experiment results (JSON)
│       ├── phase5_full_4model/    # Primary experiment (4,800 runs)
│       ├── phase6_harder/         # Harder task suite (1,968 runs)
│       ├── phase6_gradient/       # Gradient test (1,920 runs)
│       ├── phase6_oracle/         # Oracle conditions (960 runs)
│       └── phase6_track_a/        # Gold retrieval track (320 runs)
├── tests/                    # Unit tests
├── requirements.txt
└── LICENSE
```

## Reproducing Results

```bash
pip install -r requirements.txt

# Run primary experiment (4 models × 80 tasks × 5 conditions × 3 temps)
python src/multi_model_runner.py --phase 5

# Run Phase 6 follow-up experiments
python src/multi_model_runner.py --phase 6 --experiment harder
python src/multi_model_runner.py --phase 6 --experiment gradient
python src/multi_model_runner.py --phase 6 --experiment oracle
python src/multi_model_runner.py --phase 6 --experiment track_a

# Generate publication tables and figures
python src/phase6_comprehensive_analysis.py
```

## Interference Taxonomy

| Type | Count | Example |
|------|-------|---------|
| Near-identical (sim ≥ 0.8) | 17 | `create_event` vs `add_calendar_event` |
| Version conflict | 6 | `search_web_v1` vs `search_web_v2` |
| Schema conflict | 7 | Different parameter schemas for same task |
| Semantic conflict | 5 | Contradictory descriptions |
| Near-similar (0.4 ≤ sim < 0.8) | 10 | `browse_website` vs `navigate_url` |
| None | 55 | No interference |

## Citation

```bibtex
@article{meminterfere2025,
  title={Skill Library Interference in LLM Agents: Mostly Harmless, Sometimes Costly},
  author={MemInterfere Authors},
  year={2025},
  url={https://github.com/curator8888/meminterfere}
}
```

## License

MIT License — see [LICENSE](LICENSE).