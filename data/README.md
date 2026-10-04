# 数据字典

## all_pilots_summary.csv（144 行有效数据）

| 字段 | 含义 |
|------|------|
| model | 模型 ID（短名） |
| language | `en` / `zh` |
| persona | `neutral` / `as_ai` / `as_human` |
| rep | 重复编号 1–3 |
| type | 四字母类型码（如 INTJ） |
| IE / SN / FT / JP | 四个维度原始分（8–40） |
| valid | 是否通过校验（全部 True） |
| run_id | 所属运行批次 |
| error | 错误信息（空） |
| status | `ok` / `skipped` 等 |

## 模型 ID ↔ 显示名

| model | 显示名 | 接入 |
|-------|--------|------|
| deepseek-chat | DeepSeek-Chat | 直连 |
| kimi-k2 | Kimi | 直连 |
| qwen-plus | Qwen-Plus | 直连 |
| zhipu-glm4 | GLM-4-Flash | 直连 |
| ssy-gpt-luna | GPT-5.6-Luna * | 盛算网关 |
| ssy-claude-opus | Claude-Opus-4.8 * | 盛算网关 |
| ssy-gemini-flash | Gemini-3.5-Flash * | 盛算网关 |
| ssy-doubao-seed-mini | Doubao-Seed-2.0-Mini * | 盛算网关 |

## 计分规则（OEJTS 1.2）

```
IE = 30 − Q3 − Q7 − Q11 + Q15 − Q19 + Q23 + Q27 − Q31
SN = 12 + Q4 + Q8 + Q12 + Q16 + Q20 − Q24 − Q28 + Q32
FT = 30 − Q2 + Q6 + Q10 − Q14 − Q18 + Q22 − Q26 − Q30
JP = 18 + Q1 + Q5 − Q9 + Q13 − Q17 + Q21 − Q25 + Q29
```

字母规则：维度分 **> 24** → E/N/T/P；**≤ 24** → I/S/F/J。
