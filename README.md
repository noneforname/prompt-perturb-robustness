# prompt-perturb-robustness
# 大模型提示词鲁棒性研究 · 实验脚手架

> 国创项目《基于提示词扰动与统计分析的大语言模型回答鲁棒性研究》的最小可运行工程。
> 设计原则：**零第三方依赖（只要 numpy + pandas）、不联网也能跑、每一条结论都能从 `runs.csv` 重算出来。**

## 5 分钟跑起来（不需要 API Key，不花钱）

```powershell
# 1) 仿真模式跑完整个流程（40 道示例题 × 7 个条件）
python src/run_experiment.py --mock --out data/runs.csv --overwrite

# 2) 出统计结果表（含 95% CI、McNemar、BH 校正）
python src/metrics.py --input data/runs.csv

# 3) 验证统计代码本身没写错
python src/metrics.py --selftest
```

看到 `reports/summary.md` 里"选项重排（零扰动对照）Δacc = 0.0"就说明链路是对的——
仿真模型是配对设计的，零扰动对照必须严格为 0；真实实验里它不会正好是 0，
那个数就是你这张表的**噪声地板**。

## 接入真实模型（花几百元那一步）

```powershell
$env:LLM_BASE_URL="https://api.deepseek.com/v1"
$env:LLM_API_KEY="sk-f8b2f7bb9d3f466f8f8cc8437d789e5a"
$env:LLM_MODEL="deepseek-chat"

python src/llm_client.py                 # 先测连通
python src/run_experiment.py --limit 5 --dry-run   # 估算请求数
python src/run_experiment.py --out data/runs_deepseek.csv
```

其他平台（百炼 / 智谱 / 火山）只改 `LLM_BASE_URL` 和 `LLM_MODEL`，见 `.env.example`。

## 从 C-Eval 导入题目

从魔搭（ModelScope）或 Hugging Face 下载 `ceval-exam.zip`（约 1.5MB），然后：

```powershell
python src/import_ceval.py --zip D:\下载\ceval-exam.zip --n 500 --split val
```

会输出 `data/questions.csv`（我们的 schema，按学科分层抽 500 题），并打印学科分布。

- C-Eval 的 `test` 划分**没有公开答案**，本地实验一律用 `val`；
- 想先避开 LaTeX 公式题，加 `--drop-latex`；
- 脚本也能直接吃 CMMLU（`--dir` 指向它的 `data/` 目录即可，它没有表头，脚本会自动判断）。

## 目录结构

```
├─ docs/                     研究方案、执行手册、申报书要点、新手训练营
├─ data/
│  ├─ questions.template.csv 40 道示例题（schema 示范，正式实验换成 C-Eval 抽样）
│  ├─ variants.template.csv  扰动变体表（人工复核用）
│  ├─ runs.template.csv      结果表样例（一行 = 一次调用）
│  └─ runs.csv               仿真跑出来的结果（git 忽略）
├─ src/
│  ├─ llm_client.py          OpenAI 兼容客户端：重试 / 缓存 / 版本记录
│  ├─ perturbations.py       6 类扰动算子 + prompt 模板
│  ├─ run_experiment.py      批量实验，--mock 仿真模式
│  └─ metrics.py             Δacc / CI / McNemar / BH / 一致性 + 自检
├─ reports/                  统计输出（git 忽略）
├─ requirements.txt
└─ .env.example
```

## 三张表（全组统一，不许各写各的）

| 表 | 一行代表 | 关键字段 |
|---|---|---|
| `questions.csv` | 一道题 | `qid, subject, difficulty, question, option_a..d, answer, split` |
| `variants.csv` | 一道题的一个扰动版本 | `vid, qid, operator, variant_text, option_order, semantic_score, status` |
| `runs.csv` | 一次模型调用 | `run_id, vid, qid, operator, model, model_version, temperature, seed, timestamp, prompt_text, raw_response, parsed_answer, gold_answer, is_correct, parse_ok, latency_ms, tokens_in, tokens_out` |

`run_id` 规则：`{model}__{vid}__T{temperature}__S{seed}__R{重复序号}`，重跑时自动跳过已存在的行（断点续跑）。

## 七个条件

| operator | 中文 | 类型 |
|---|---|---|
| `original` | 原始题 | 基线 |
| `synonym` | 同义替换 | 词汇级 |
| `word_order` | 语序调整 | 句法级 |
| `irrelevant` | 无关信息插入 | 语境级 |
| `option_reorder` | 选项重排 | **零扰动对照** |
| `role_prefix` | 角色前缀 | 指令级 |
| `format_constraint` | 格式约束 | 指令级 |

## 验收清单

- [x] `--mock` 跑出 ≥200 行 `runs.csv`，全程不联网
- [x] `metrics.py` 输出含 95% CI、p 值、BH q 值的表
- [x] `metrics.py --selftest` 全部通过（已知效应能检出、真无效应不误报）
- [x] `.env.example` 里没有任何真实 Key
- [x] 零扰动对照在仿真数据上严格为 0（证明统计流程无系统偏差）

## 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `没有找到 API Key` | 环境变量没设 | 见上文，或先用 `--mock` |
| `HTTP 401` | Key 错/过期 | 重新生成，注意别把 Key 提交进 Git |
| `HTTP 429` | 限流 | 加 `--sleep 0.5`，或换平台 |
| `parse_ok=0` 很多 | 模型没按格式输出 | 改 prompt 模板并升 `prompt_template` 版本号 |
| 某算子 `n` 特别小 | 该题不适用（skip） | 正常；skip 率 >10% 要回头改进算子 |
| 中文乱码 | 终端编码 | 脚本已强制 UTF-8 输出；CSV 用 `utf-8-sig` 打开 |

## 团队文档

1. [研究方案](docs/01_研究方案.md) —— 为什么这么做、每个决策的依据
2. [执行手册](docs/02_执行手册.md) —— 逐周任务、质检规则、报错处理
3. [申报书与答辩要点](docs/03_申报书与答辩要点.md) —— 可直接改写的申报文本 + 12 个答辩问答
4. [新手训练营](docs/04_新手训练营.md) —— 4 周从零到跑通
