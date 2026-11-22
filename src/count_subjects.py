# -*- coding: utf-8 -*-
"""第一课：读一张表，数一数，顺便做最基础的数据检查。

这是给完全没写过代码的同学的第一个任务：不涉及模型、不涉及统计，
只确认三件事——Python 装好了、pandas 能用、文件路径没写错。

用法：
    python src/count_subjects.py
    python src/count_subjects.py --input data/questions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

LETTERS = "ABCD"


def main() -> int:
    ap = argparse.ArgumentParser(description="数一数题库里每个学科有几道题")
    ap.add_argument("--input", default="data/questions.template.csv")
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"找不到文件：{path}")
        print("先确认你在项目根目录（有 src/ 和 data/ 的那一层），或者用 --input 指定路径")
        return 1

    # 关键就这一行：把 csv 读成一张表
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    print(f"文件：{path}")
    print(f"共 {len(df)} 行 × {len(df.columns)} 列")
    print(f"列名：{df.columns.tolist()}\n")

    if "subject" not in df.columns:
        print("这张表没有 subject 列，没法按学科数")
        return 1

    # 按学科分组，数每组的行数
    counts = df["subject"].value_counts()
    print("每个学科有几道题：")
    for subject, n in counts.items():
        print(f"  {subject:<6} {n} 道")
    print(f"  {'合计':<6} {counts.sum()} 道")

    # ---------------- 顺手做三项检查 ----------------
    print("\n数据检查：")
    ok = True

    dup = df["qid"].duplicated().sum() if "qid" in df.columns else 0
    ok &= dup == 0
    print(f"  {'✅' if dup == 0 else '❌'} 题号不重复（重复 {dup} 个）")

    if "answer" in df.columns:
        bad = (~df["answer"].str.strip().str.upper().str[:1].isin(list(LETTERS))).sum()
        ok &= bad == 0
        print(f"  {'✅' if bad == 0 else '❌'} 答案都在 A–D 内（异常 {bad} 道）")
    else:
        print("  ⚠️  没有 answer 列")
        ok = False

    opt_cols = [c for c in df.columns if c.startswith("option_")]
    if opt_cols:
        empty = (df[opt_cols].apply(lambda s: s.str.strip() == "")).any(axis=1).sum()
        ok &= empty == 0
        print(f"  {'✅' if empty == 0 else '❌'} 选项没有空的（空 {empty} 道）")
    else:
        print("  ⚠️  没有 option_ 开头的列")
        ok = False

    print("\n结论：" + ("这张表可以直接用" if ok else "有几处要修，修完再用"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
