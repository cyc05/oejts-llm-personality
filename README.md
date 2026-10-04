# OEJTS LLM Personality Experiment

Reproducible factorial experiment administering the **Open Extended Jungian Type Scales (OEJTS) 1.2**
to **8 large language models** under controlled conditions, with fully archived data and a Python CLI.

> Uses OEJTS 1.2 (open-source Jungian type scales). **Not affiliated with, and not the official MBTI®**
> assessment. Instrument: © Open Psychometrics, CC BY-NC-SA 4.0.

## What's inside

| Path | Content |
|------|---------|
| `data/all_pilots_summary.csv` | Aggregated results: 144 valid questionnaires (8 models × 2 languages × 3 personas × 3 reps) |
| `results/<run_id>/` | Per-run scored results + config snapshots |
| `artifacts/` | Raw prompts/completions (`raw/`), parsed answers (`results/`), reports (`reports/`) |
| `instruments/oejts/` | Bilingual OEJTS 1.2 items + license |
| `prompts/`, `configs/` | Persona × language prompt templates and experiment config |
| `src/psy_experiment/` | Reproducible pipeline (Python CLI: `doctor`, `run`, `score`, `summarize`) |
| `docs/EXPERIMENT.md` | Full experiment description（中文实验说明） |

## Experiment design

| Factor | Levels |
|--------|--------|
| Model | 8 (DeepSeek-Chat, Kimi, Qwen-Plus, GLM-4-Flash; GPT-5.6-Luna*, Claude-Opus-4.8*, Gemini-3.5-Flash*, Doubao-Seed-2.0-Mini* via gateway) |
| Language | English / Chinese |
| Identity framing | neutral / as-AI / as-human |
| Replicates | 3 |

8 × 2 × 3 × 3 = **144 valid questionnaires** · temperature 0.7 · whole-questionnaire single-call mode · local scoring.

## Key findings

- **Three response-style clusters**: near-deterministic INTJ responders (GPT-5.6-Luna 18/18, Gemini 17/18,
  Claude 14/18); an opposite Qwen-Plus ESTP-style profile (Perceiving 18/18); unstable domestic models
  (DeepSeek-Chat, Kimi, GLM-4-Flash, Doubao).
- **Stability**: mean within-condition type agreement **0.69** (range 0.44–1.00);
  letter-level agreement per dimension **0.87–0.92**.
- **Prompt sensitivity**: cross-language type match 9/9 (GPT-5.6-Luna) vs. 1/9 (four models);
  human-role framing shifts Claude-Opus-4.8 INTJ → ISFJ (3/3).

\* Accessed through a third-party routing gateway; not claimed equivalent to first-party endpoints.

## Reproduce

```bash
pip install -e ".[dev]"        # install the CLI
copy .env.example .env         # then fill API keys (never commit .env)
psy-exp doctor
psy-exp run --models deepseek-chat --languages en --personas neutral --reps 1
psy-exp summarize --run-id <id>
```

## Citation

If you use this data or protocol, please cite the corresponding paper once published,
and attribute OEJTS 1.2 to Open Psychometrics under CC BY-NC-SA 4.0.

## License

- **Code** — pipeline, configurations, prompt templates, analysis scripts: **MIT**, see [`LICENSE`](LICENSE).
- **OEJTS 1.2 instrument** — item wording and scoring formulas: **CC BY-NC-SA 4.0**,
  © Open Psychometrics; non-commercial use only. See [`NOTICE.md`](NOTICE.md).
- Not affiliated with, and does not use, the official MBTI® instrument.
