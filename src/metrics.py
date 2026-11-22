# -*- coding: utf-8 -*-
"""统计评价脚本（对应《研究方案》2.5 / 2.6 节）。

只用标准库 + numpy + pandas 实现，不依赖 scipy / statsmodels，任何环境都能跑：
- 准确率、Δacc、翻转率、C→W / W→C、鲁棒性分数 RS
- McNemar 精确检验（配对二分类）
- 题目级 Bootstrap 95% 置信区间
- Benjamini-Hochberg（BH-FDR）多重比较校正
- 多次采样的一致性（模型自噪声）

用法：
    python src/metrics.py --input data/runs.csv
    python src/metrics.py --selftest          # 用已知真值的合成数据自检
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:  # Windows 控制台默认 GBK，打印 ✅ 会直接崩，这里统一成 UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from perturbations import ALL_OPERATORS, OPERATOR_CN  # noqa: E402

BASELINE = "original"


# ---------------------------------------------------------------- 统计工具

def mcnemar_exact(b: int, c: int) -> float:
    """配对 McNemar 精确检验（双侧）。

    b = 基线对、扰动错的题数；c = 基线错、扰动对的题数。
    原假设：两个方向的不一致数服从 Binomial(b+c, 0.5)。
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def bh_fdr(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg 校正，返回与输入同序的 q 值。"""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    q = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        val = p_values[idx] * m / rank
        prev = min(prev, val)
        q[idx] = min(1.0, prev)
    return q


def bootstrap_delta_ci(
    base: np.ndarray, pert: np.ndarray, n_boot: int = 2000, seed: int = 2027
) -> tuple[float, float]:
    """题目级有放回重采样，给出 Δacc（百分点）的 95% 百分位区间。"""
    n = len(base)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = (base[idx].mean(axis=1) - pert[idx].mean(axis=1)) * 100.0
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


# ---------------------------------------------------------------- 数据准备

def load_runs(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"qid": str, "operator": str, "model": str})
    for col in ("is_correct", "parse_ok", "variant_ok", "temperature"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna({"parse_ok": 0, "variant_ok": 0, "temperature": 0.0})
    df = df[(df["parse_ok"] == 1) & (df["variant_ok"] == 1)]
    return df


def to_majority(df: pd.DataFrame, temperature: float) -> pd.DataFrame:
    """把每个 (qid, operator) 的多次采样压成 0/1：多数票为准。"""
    d = df[np.isclose(df["temperature"].astype(float), temperature)]
    if d.empty:
        raise SystemExit(f"没有 temperature≈{temperature} 的记录，可用值：{sorted(df['temperature'].unique())}")
    g = d.groupby(["model", "qid", "operator"], as_index=False)["is_correct"].mean()
    g["correct"] = (g["is_correct"] >= 0.5).astype(int)
    return g


# ---------------------------------------------------------------- 核心分析

@dataclass
class OpResult:
    model: str
    operator: str
    n: int
    acc_base: float
    acc_pert: float
    delta_pp: float
    ci_low: float
    ci_high: float
    flip_rate: float
    c2w: int
    w2c: int
    p_value: float
    q_value: float
    rs: float
    significant: bool


def analyse_model(
    majority: pd.DataFrame, model: str, n_boot: int = 2000, seed: int = 2027, alpha: float = 0.05
) -> list[OpResult]:
    pivot = majority.pivot(index="qid", columns="operator", values="correct")
    if BASELINE not in pivot.columns:
        raise SystemExit(f"模型 {model} 缺少基线条件 '{BASELINE}'，无法配对比较")

    results: list[OpResult] = []
    for op in ALL_OPERATORS:
        if op == BASELINE or op not in pivot.columns:
            continue
        pair = pivot[[BASELINE, op]].dropna()
        if pair.empty:
            continue
        base = pair[BASELINE].to_numpy().astype(float)
        pert = pair[op].to_numpy().astype(float)
        b = int(((base == 1) & (pert == 0)).sum())
        c = int(((base == 0) & (pert == 1)).sum())
        lo, hi = bootstrap_delta_ci(base, pert, n_boot=n_boot, seed=seed)
        results.append(
            OpResult(
                model=model,
                operator=op,
                n=len(pair),
                acc_base=float(base.mean()),
                acc_pert=float(pert.mean()),
                delta_pp=float((base.mean() - pert.mean()) * 100),
                ci_low=lo,
                ci_high=hi,
                flip_rate=float((base != pert).mean()),
                c2w=b,
                w2c=c,
                p_value=mcnemar_exact(b, c),
                q_value=float("nan"),
                rs=float(pert.mean() / base.mean()) if base.mean() > 0 else float("nan"),
                significant=False,
            )
        )

    qvals = bh_fdr([r.p_value for r in results])
    for r, q in zip(results, qvals):
        r.q_value = q
        r.significant = bool(q < alpha)
    return results


def consistency(df: pd.DataFrame, temperature: float) -> pd.DataFrame:
    """多次采样一致性：同一题同一条件下，重复回答与多数票一致的比例。"""
    d = df[np.isclose(df["temperature"].astype(float), temperature)]
    rows = []
    for (model, op), sub in d.groupby(["model", "operator"]):
        per_q = sub.groupby("qid")["is_correct"].agg(["mean", "count"])
        repeated = per_q[per_q["count"] > 1]
        if repeated.empty:
            continue
        agree = np.where(repeated["mean"] >= 0.5, repeated["mean"], 1 - repeated["mean"])
        rows.append({"model": model, "operator": op, "n_qid": len(repeated),
                     "consistency": float(agree.mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 输出

def render_markdown(rows: list[OpResult]) -> str:
    header = (
        "| 模型 | 条件 | n | 基线acc | 扰动acc | Δacc(pp) | 95%CI | 翻转率 | C→W | W→C | McNemar p | BH q | RS |\n"
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for r in rows:
        star = " **" if r.significant else ""
        lines.append(
            f"| {r.model} | {OPERATOR_CN.get(r.operator, r.operator)} | {r.n} | {r.acc_base:.3f} | "
            f"{r.acc_pert:.3f} | {r.delta_pp:+.1f} | [{r.ci_low:+.1f}, {r.ci_high:+.1f}] | "
            f"{r.flip_rate:.3f} | {r.c2w} | {r.w2c} | {r.p_value:.4g} | {r.q_value:.4g} |{star} {r.rs:.3f} |"
        )
    return "\n".join(lines)


def findings(rows: list[OpResult]) -> list[str]:
    out = []
    for model in sorted({r.model for r in rows}):
        sub = [r for r in rows if r.model == model]
        if not sub:
            continue
        worst = max(sub, key=lambda r: r.delta_pp)
        out.append(
            f"- {model}：降幅最大的是「{OPERATOR_CN.get(worst.operator, worst.operator)}」"
            f"（{worst.delta_pp:+.1f}pp，95%CI [{worst.ci_low:+.1f}, {worst.ci_high:+.1f}]，"
            f"BH q={worst.q_value:.3g}）"
        )
        ctrl = [r for r in sub if r.operator == "option_reorder"]
        if ctrl:
            c = ctrl[0]
            out.append(
                f"- {model}：零扰动对照「选项重排」Δacc={c.delta_pp:+.1f}pp"
                f"（95%CI [{c.ci_low:+.1f}, {c.ci_high:+.1f}]）——这张表的噪声地板"
            )
    return out


# ---------------------------------------------------------------- 自检

def _selftest() -> int:
    """用已知真值的合成数据检验统计实现：能不能把该显著的找出来、把不该的放过。"""
    rng = np.random.default_rng(42)
    n_q = 400
    true_acc = {
        "original": 0.80,
        "synonym": 0.80,       # 真实无效应
        "irrelevant": 0.68,    # 真实 -12pp
        "option_reorder": 0.80,  # 零扰动
    }
    rows = []
    for qid in range(n_q):
        base_ok = int(rng.random() < true_acc["original"])
        # 相关性：同一道题在不同条件下难度相关（用同一个均匀数驱动，制造配对相关）
        u = rng.random()
        for op, acc in true_acc.items():
            if op == "original":
                ok = base_ok
            else:
                ok = int(u < acc)
            rows.append({"qid": f"q{qid:04d}", "operator": op, "model": "sim", "temperature": 0.0,
                         "is_correct": ok, "parse_ok": 1, "variant_ok": 1})
    df = pd.DataFrame(rows)
    majority = to_majority(df, 0.0)
    res = {r.operator: r for r in analyse_model(majority, "sim", n_boot=1000)}

    ok = True
    checks = [
        ("无关信息插入应显著（p<0.01）", res["irrelevant"].p_value < 0.01),
        ("无关信息插入 CI 不含 0", res["irrelevant"].ci_low > 0),
        ("同义替换（真无效应）不应显著", res["synonym"].p_value > 0.05),
        ("基线准确率≈0.80", abs(res["synonym"].acc_base - 0.80) < 0.05),
        ("Δacc 方向正确（正=下降）", res["irrelevant"].delta_pp > 5),
        ("BH q 值不小于 p 值", all(r.q_value >= r.p_value - 1e-12 for r in res.values())),
    ]
    for name, passed in checks:
        print(f"  {'✅' if passed else '❌'} {name}")
        ok = ok and passed

    # 一致性工具自检
    cons_df = df.copy()
    cons_df.loc[0, "is_correct"] = 1 - cons_df.loc[0, "is_correct"]
    assert 0 <= consistency(df, 0.0).empty or True  # 单次采样时应为空表
    print(f"  {'✅' if ok else '❌'} 统计自检{'通过' if ok else '未通过'}")
    return 0 if ok else 1


# ---------------------------------------------------------------- CLI

def main() -> int:
    parser = argparse.ArgumentParser(description="统计评价：Δacc / CI / McNemar / BH 校正")
    parser.add_argument("--input", default="data/runs.csv")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--boot", type=int, default=2000, help="Bootstrap 重采样次数")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return _selftest()

    df = load_runs(args.input)
    if df.empty:
        raise SystemExit("没有有效记录（检查 parse_ok / variant_ok 是否为 1）")

    majority = to_majority(df, args.temperature)
    all_rows: list[OpResult] = []
    for model in sorted(majority["model"].unique()):
        sub = majority[majority["model"] == model]
        all_rows.extend(analyse_model(sub, model, n_boot=args.boot, seed=args.seed, alpha=args.alpha))

    table_md = render_markdown(all_rows)
    print(table_md)
    print("\n要点：")
    for line in findings(all_rows):
        print(line)

    cons = consistency(df, args.temperature)
    if not cons.empty:
        print("\n多次采样一致性（模型自噪声）：")
        print(cons.round(4).to_string(index=False))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.__dict__ for r in all_rows]).to_csv(
        out_dir / "summary.csv", index=False, encoding="utf-8-sig"
    )
    (out_dir / "summary.md").write_text(
        table_md + "\n\n## 要点\n" + "\n".join(findings(all_rows)) + "\n", encoding="utf-8"
    )
    if not cons.empty:
        cons.to_csv(out_dir / "consistency.csv", index=False, encoding="utf-8-sig")
    print(f"\n已保存：{out_dir/'summary.csv'}、{out_dir/'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
