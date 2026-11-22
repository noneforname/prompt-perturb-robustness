# -*- coding: utf-8 -*-
"""批量实验脚本：题目 + 扰动算子 × 模型 → runs.csv（一行 = 一次调用）。

三种运行模式：
1. --mock        仿真模式：不联网、不花钱，用"带真值的假模型"模拟已知效应。
                  用来验证整条流水线和统计代码是否正确（真值：无关信息插入 -12pp 等）。
2. 提供 API Key   真实模式：调用 OpenAI 兼容接口，产出正式实验数据。
3. --dry-run     只打印会发出多少个请求，一个也不发。

示例：
    python src/run_experiment.py --mock --out data/runs.csv --limit 60
    python src/run_experiment.py --model deepseek-chat --limit 5 --dry-run
"""

from __future__ import annotations

import argparse
import atexit
import csv
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:  # Windows 控制台默认 GBK，中文/符号打印会崩，这里统一成 UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import ChatResult, LLMClient, MockClient  # noqa: E402
from perturbations import (  # noqa: E402
    ALL_OPERATORS,
    OPERATOR_CN,
    PROMPT_TEMPLATE_VERSION,
    Question,
    Variant,
    build_messages,
    make_variant,
)

RUN_FIELDS = [
    "run_id", "vid", "qid", "operator", "model", "model_version", "temperature", "seed",
    "max_tokens",
    "timestamp", "prompt_template", "prompt_text", "raw_response", "parsed_answer",
    "gold_answer", "is_correct", "parse_ok", "latency_ms", "tokens_in", "tokens_out",
    "variant_ok", "note",
]

# 仿真模式的"真值"：故意让不同扰动有不同降幅，用来检验统计代码能否把它们找回来。
SIM_ACCURACY = {
    "original": 0.78,
    "synonym": 0.75,
    "word_order": 0.72,
    "irrelevant": 0.66,   # 预期降幅最大
    "option_reorder": 0.78,  # 零扰动：应无变化
    "role_prefix": 0.76,
    "format_constraint": 0.70,
}
SIM_MODEL_FACTOR = {"sim-strong": 1.0, "sim-weak": 0.88}


# ---------------------------------------------------------------- 数据读取

def load_questions(path: str | Path) -> list[Question]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"qid", "question", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少必需列：{sorted(missing)}")

    opt_cols = sorted([c for c in df.columns if re.fullmatch(r"option_[a-f]", c)])
    if not opt_cols and "options" not in df.columns:
        raise ValueError(f"{path} 里既没有 option_a..option_f 列，也没有 options 列")

    questions: list[Question] = []
    for _, row in df.iterrows():
        if "options" in df.columns and not opt_cols:
            options = [x.strip() for x in str(row["options"]).split("|") if x.strip()]
        else:
            options = [str(row[c]).strip() for c in opt_cols if str(row[c]).strip()]
        options = options[:4]  # 本项目只做四选一
        questions.append(
            Question(
                qid=str(row["qid"]).strip(),
                subject=str(row.get("subject", "")).strip(),
                difficulty=str(row.get("difficulty", "")).strip(),
                question=str(row["question"]).strip(),
                options=options,
                answer=str(row["answer"]).strip(),
                source=str(row.get("source", "")).strip(),
                split=str(row.get("split", "test")).strip(),
            )
        )
    return questions


# ---------------------------------------------------------------- 答案解析

_ANSWER_RE = re.compile(r"答案\s*[:：]\s*[（(]?\s*([A-Fa-f])\s*[)）]?")
_LAST_LINE_RE = re.compile(r"^\s*[（(]?\s*([A-Fa-f])\s*[)）]?\s*$", re.MULTILINE)


def parse_answer(text: str) -> str | None:
    """从模型回复里抽选项字母。抽不到返回 None（记为 parse_ok=False）。"""
    if not text:
        return None
    hits = _ANSWER_RE.findall(text)
    if hits:
        return hits[-1].upper()
    tail = _LAST_LINE_RE.findall(text)
    if tail:
        return tail[-1].upper()
    return None


# ---------------------------------------------------------------- 仿真模型

class SimulatedResponder:
    """带真值的假模型：用于验证流水线与统计代码，不产生任何研究结论。"""

    def __init__(self, model: str = "sim-strong", noise_seed: int = 7) -> None:
        self.model = model
        self.factor = SIM_MODEL_FACTOR.get(model, 1.0)

    def respond(self, variant: Variant, seed: int) -> ChatResult:
        # 关键：用"题目难度"这个与算子无关的随机数 u 决定对错，
        # 这样同一道题在不同条件下是强相关的（真实实验也是配对设计）。
        # 于是"选项重排"（acc 与基线相同）会严格得到 Δacc = 0，正好可以检验流程没有系统偏差。
        acc = min(0.99, SIM_ACCURACY.get(variant.operator, 0.75) * self.factor)
        u = (hash_int(f"{variant.qid}|{self.model}|{seed}") % 1_000_000) / 1_000_000
        gold = variant.answer
        if u < acc:
            letter = gold
        else:
            candidates = [c for c in "ABCD"[: len(variant.options)] if c != gold]
            letter = candidates[hash_int(f"{variant.qid}|{variant.operator}|{seed}|wrong") % len(candidates)]
        body = f"（仿真）依据题干信息逐项排除。\n答案：{letter}"
        return ChatResult(
            text=body,
            model=self.model,
            model_version="simulated-v1",
            prompt_tokens=len(variant.question) // 2 + 20,
            completion_tokens=len(body) // 2,
            latency_ms=0,
        )


def hash_int(text: str) -> int:
    """把字符串映射成稳定的整数（内置 hash 对字符串是随机化的，不能用）。"""
    import hashlib

    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


class _Tee:
    """把 stdout 同时写到控制台与日志文件，便于事后审计。"""

    def __init__(self, stream, fh) -> None:
        self.stream = stream
        self.fh = fh

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.fh.write(text)
        self.flush()
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.fh.flush()


def _setup_run_log(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = log_path.open("a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, fh)
    print(f"\n===== {datetime.now().astimezone().isoformat(timespec='seconds')} 运行开始 =====")
    print(f"命令：python {' '.join(sys.argv)}")

    def _finish() -> None:
        sys.stdout = sys.__stdout__
        fh.close()

    atexit.register(_finish)


# ---------------------------------------------------------------- 主流程

def build_run_id(model: str, vid: str, temperature: float, seed: int, repeat: int) -> str:
    return f"{model}__{vid}__T{temperature:g}__S{seed}__R{repeat}"


def load_done_ids(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        done = pd.read_csv(path, usecols=["run_id"], dtype=str)
        return set(done["run_id"].astype(str))
    except Exception:  # noqa: BLE001 - 半个文件也能尽量续跑
        return set()


def main() -> int:
    parser = argparse.ArgumentParser(description="批量实验：题目 × 扰动 × 模型 → runs.csv")
    parser.add_argument("--input", default="data/questions.template.csv")
    parser.add_argument("--out", default="data/runs.csv")
    parser.add_argument("--model", default=None, help="模型名，默认读 LLM_MODEL 环境变量")
    parser.add_argument("--conditions", default="all",
                        help="逗号分隔的算子，默认 all：" + ",".join(ALL_OPERATORS))
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 道题（0=全部）")
    parser.add_argument("--repeat", type=int, default=1, help="每个条件重复采样次数（T>0 时用）")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="单次回复最大输出 token 数（默认 8192；含隐藏推理 token）")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--sleep", type=float, default=0.0, help="每次调用之间暂停秒数（限流用）")
    parser.add_argument("--mock", action="store_true", help="仿真模式，不联网")
    parser.add_argument("--dry-run", action="store_true", help="只统计请求数，不发请求")
    parser.add_argument("--overwrite", action="store_true", help="删掉已有 runs.csv 重新开始")
    parser.add_argument("--log-file", default=None,
                        help="运行日志路径（默认与 --out 同名 .log，追加写入）")
    args = parser.parse_args()

    conditions = ALL_OPERATORS if args.conditions.strip() == "all" else [
        c.strip() for c in args.conditions.split(",") if c.strip()
    ]
    unknown = [c for c in conditions if c not in ALL_OPERATORS]
    if unknown:
        raise SystemExit(f"未知算子：{unknown}，可选：{ALL_OPERATORS}")

    questions = load_questions(args.input)
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        raise SystemExit("没有读到任何题目")

    out_path = Path(args.out)
    log_path = Path(args.log_file) if args.log_file else out_path.with_suffix(".log")
    _setup_run_log(log_path)
    if args.overwrite and out_path.exists():
        out_path.unlink()
    done = load_done_ids(out_path)

    if out_path.exists() and out_path.stat().st_size > 0:
        with out_path.open("r", encoding="utf-8-sig", newline="") as f:
            existing_fields = next(csv.reader(f), [])
        if existing_fields != RUN_FIELDS:
            raise SystemExit(
                f"{out_path} 的列与当前代码不一致（多半是旧版本生成的），"
                "请换一个 --out 文件名，或加 --overwrite 从头重跑。"
            )

    model = args.model or os.getenv("LLM_MODEL") or ("sim-strong" if args.mock else "unknown-model")

    client = None
    simulator = None
    if args.mock:
        simulator = SimulatedResponder(model=model if model.startswith("sim-") else "sim-strong")
        model = simulator.model
    elif not args.dry_run:
        client = LLMClient(model=model)

    total_plan = len(questions) * len(conditions) * args.repeat
    print(f"题目 {len(questions)} 道 × 条件 {len(conditions)} 个 × 重复 {args.repeat} 次 = 计划 {total_plan} 次调用")
    print(f"模型：{model}｜温度：{args.temperature}｜输出：{out_path}")
    if args.dry_run:
        print("（dry-run：以上只是计划，没有发出任何请求）")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_path.exists() or out_path.stat().st_size == 0
    fh = out_path.open("a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(fh, fieldnames=RUN_FIELDS)
    if new_file:
        writer.writeheader()

    done_count = skipped = failed = truncated = variant_skipped = 0
    started_at = time.time()
    try:
        for qi, q in enumerate(questions, 1):
            for op in conditions:
                variant = make_variant(q, op, seed=args.seed)
                if not variant.ok:
                    # 算子判定该题无法安全扰动：不发请求、不花钱、不写入结果表
                    variant_skipped += 1
                    continue
                messages = build_messages(variant)
                prompt_text = messages[-1]["content"]
                for rep in range(args.repeat):
                    run_id = build_run_id(model, variant.vid, args.temperature, args.seed, rep)
                    if run_id in done:
                        skipped += 1
                        continue

                    sample_seed = args.seed + rep
                    error_note = ""
                    try:
                        if simulator is not None:
                            result = simulator.respond(variant, sample_seed)
                        else:
                            assert client is not None
                            result = client.chat(
                                messages,
                                temperature=args.temperature,
                                seed=sample_seed if args.temperature > 0 else None,
                                max_tokens=args.max_tokens,
                            )
                    except Exception as exc:  # noqa: BLE001 - 单条失败不能中断整个实验
                        failed += 1
                        error_note = f"ERROR: {type(exc).__name__}: {exc}"[:300]
                        result = ChatResult("", model, "unknown", 0, 0, 0)

                    parsed = parse_answer(result.text)
                    trunc_note = (
                        "TRUNCATED: finish_reason=length（回复被 max_tokens 截断，答案行缺失）"
                        if result.finish_reason == "length" else ""
                    )
                    if trunc_note:
                        truncated += 1
                    row = {
                        "run_id": run_id,
                        "vid": variant.vid,
                        "qid": q.qid,
                        "operator": variant.operator,
                        "model": model,
                        "model_version": result.model_version,
                        "temperature": args.temperature,
                        "seed": sample_seed,
                        "max_tokens": args.max_tokens,
                        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                        "prompt_template": PROMPT_TEMPLATE_VERSION,
                        "prompt_text": prompt_text,
                        "raw_response": result.text[:4000],
                        "parsed_answer": parsed or "",
                        "gold_answer": variant.answer,
                        "is_correct": int(parsed == variant.answer),
                        "parse_ok": int(parsed is not None),
                        "latency_ms": result.latency_ms,
                        "tokens_in": result.prompt_tokens,
                        "tokens_out": result.completion_tokens,
                        "variant_ok": int(variant.ok),
                        "note": "；".join(
                            x for x in (error_note, trunc_note, variant.note) if x
                        )[:300],
                    }
                    writer.writerow(row)
                    fh.flush()
                    done_count += 1

                    if args.sleep:
                        time.sleep(args.sleep)
            if qi % 10 == 0 or qi == len(questions):
                elapsed = time.time() - started_at
                print(f"  进度 {qi}/{len(questions)} 题｜新增 {done_count}｜跳过 {skipped}｜失败 {failed}｜{elapsed:.0f}s")
    finally:
        fh.close()

    print(
        f"\n完成：新增 {done_count} 行，跳过已存在 {skipped} 行，"
        f"算子不适用 {variant_skipped} 条，失败 {failed} 行，截断 {truncated} 行 → {out_path}"
    )
    if variant_skipped:
        print("  提示：算子不适用的题会在 metrics 里自动丢弃，不影响配对比较；"
              "但若某算子 skip 率 >10%，请回看 perturbations.py 的策略是否需要改进。")

    # 跑完立刻给一个粗表，方便现场判断有没有跑歪
    try:
        df = pd.read_csv(out_path, dtype=str)
        df["is_correct"] = df["is_correct"].astype(int)
        summary = (
            df[df["parse_ok"].astype(int) == 1]
            .groupby("operator")["is_correct"]
            .agg(["count", "mean"])
            .reindex([c for c in ALL_OPERATORS if c in set(df["operator"])])
        )
        if not summary.empty:
            print("\n各条件准确率（未做统计校正，仅供肉眼检查）：")
            for op, row in summary.iterrows():
                print(f"  {OPERATOR_CN.get(op, op):<18} n={int(row['count']):>5}  acc={row['mean']:.3f}")
    except Exception as exc:  # noqa: BLE001
        print(f"（汇总表生成失败：{exc}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
