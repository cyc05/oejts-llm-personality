# Experiment Results Archive

Uses OEJTS 1.2 (open-source Jungian type scales). **Not affiliated with, and not the official MBTI®** assessment. CC BY-NC-SA 4.0.

This directory stores **durable** pilot outputs that should stay in git.
Live run scratch (including bulky `raw/` API transcripts) remains under ignored `artifacts/`.

## Pilots

| run_id | model | cells | status | notes |
|--------|-------|-------|--------|-------|
| `deepseek-pilot-001` | deepseek-chat | 18 | valid | domestic |
| `kimi-pilot-001` | moonshot-v1-8k (`kimi-k2`) | 18 | valid | domestic |
| `qwen-pilot-001` | qwen-plus | 18 | valid | domestic |
| `zhipu-pilot-001` | glm-4-flash (`zhipu-glm4`) | 18 | valid | domestic |
| `ssy-gpt-pilot-001` | openai/gpt-5.6-luna via 胜算云 | 18 | valid | international gateway |
| `ssy-claude-pilot-001` | anthropic/claude-opus-4.8 via 胜算云 | 18 | valid | international gateway |
| `ssy-gemini-pilot-001` | google/gemini-3.5-flash via 胜算云 | 18 | valid | international gateway |
| `ssy-doubao-pilot-001` | bytedance/doubao-seed-2-0-mini via 胜算云 | 18 | valid | ByteDance via gateway |

Failed / smoke runs (not archived): `doubao-smoke-001`, `*-smoke-001`.

Combined table: [`all_pilots_summary.csv`](all_pilots_summary.csv) (**144** rows = 8 models × 18).

## Shengsuan (国际三模型) 一致性摘要

| 模型 | 条件众数（一致率） | 备注 |
|------|-------------------|------|
| GPT Luna | **全部 6 条件均为 INTJ（3/3）** | 同条件三次分数几乎完全相同，极稳 |
| Claude Opus | en/as_ai INTJ 2/3；en/as_human **ISFJ 3/3**；en/neutral INTJ 3/3；zh/as_ai INTJ 3/3；zh/as_human ISTJ 2/3；zh/neutral INTJ 3/3 | 扮演人类时更偏 S/F |
| Gemini Flash | 五条件 INTJ 3/3；zh/as_human **INFJ 2/3** | 与 GPT 接近，人类角色略软化 |
| Doubao Seed Mini | en/as_human **ISFP 2/3**；zh/as_ai **ESTJ 2/3**；zh/as_human **ENFJ 2/3**；其余 1/3 | 漂移大，整体更偏 **E/J**，不像 GPT 系稳定 INTJ |

相对国内模型：国际三模型整体更偏 **I/N/T/J**；千问此前偏 **E/S/T/P**，反差很大。

## Layout per pilot

```text
results/<run_id>/
  summary.csv
  config.snapshot.yaml
  cell_results/*.json
```

## Factorial design

- Languages: `en`, `zh`
- Personas: `neutral`, `as_ai`, `as_human`
- Replicates: 3
- Instrument: OEJTS 1.2 (32 items), whole-questionnaire mode
- Default temperature: 0.7 (see each `config.snapshot.yaml`)
- 胜算云 base: `https://router.shengsuanyun.com/api/v1`（`SSY_API_KEY`）
