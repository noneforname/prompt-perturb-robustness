# -*- coding: utf-8 -*-
"""把 C-Eval / CMMLU 的原始题库转成本项目的 questions.csv。

支持三种输入：
    --zip  ceval-exam.zip        （官方/魔搭下载的压缩包）
    --dir  ./ceval-exam          （已解压的目录）
    --csv  某个具体的 csv         （单文件）

会自动做四件事：
1. 判断有没有表头（C-Eval 有、CMMLU 没有）；
2. 统一列名（question/A/B/C/D/answer，大小写和中文列名都能认）；
3. 只保留"四个选项齐全 + 答案是 A–D"的干净题目；
4. 按学科分层抽样到指定题数。

用法：
    python src/import_ceval.py --zip D:\\下载\\ceval-exam.zip --n 500
    python src/import_ceval.py --dir D:\\下载\\ceval-exam --n 500 --split val
    python src/import_ceval.py --selftest        # 用假数据验证脚本本身
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

LETTERS = "ABCD"
SPLIT_WORDS = {"dev": "dev", "val": "val", "validation": "val", "test": "test"}
OUT_COLUMNS = ["qid", "subject", "difficulty", "question",
               "option_a", "option_b", "option_c", "option_d",
               "answer", "source", "split"]

# 列名归一化：原始字段 -> 我们的字段
NAME_MAP = {
    "question": "question", "题目": "question", "题干": "question", "问题": "question",
    "a": "A", "option_a": "A", "选项a": "A",
    "b": "B", "option_b": "B", "选项b": "B",
    "c": "C", "option_c": "C", "选项c": "C",
    "d": "D", "option_d": "D", "选项d": "D",
    "answer": "answer", "答案": "answer", "label": "answer",
    "id": "id", "qid": "id", "编号": "id",
}


# ---------------------------------------------------------------- 读取
def read_table(path: Path) -> pd.DataFrame:
    """读一个 csv，自动判断有没有表头。"""
    with path.open(encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
    low = first.lower()
    has_header = ("question" in low) or ("answer" in low) or low.startswith("id")
    if has_header:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False,
                         header=None, names=["question", "A", "B", "C", "D", "answer"])
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """列名统一成 question / A / B / C / D / answer。"""
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in NAME_MAP:
            rename[col] = NAME_MAP[key]
    df = df.rename(columns=rename)
    need = {"question", "A", "B", "C", "D", "answer"}
    if not need.issubset(df.columns):
        raise ValueError(f"缺少必要列，实际列名：{list(df.columns)}")
    keep = [c for c in ["id", "question", "A", "B", "C", "D", "answer"] if c in df.columns]
    return df[keep].copy()


def guess_split(path: Path) -> str:
    # 既看目录名（.../val/xxx.csv），也看文件名（.../test.csv）
    for part in [*path.parts[:-1], path.stem]:
        key = part.lower().strip()
        if key in SPLIT_WORDS:
            return SPLIT_WORDS[key]
    return "unknown"


def guess_subject(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"_(val|valid|validation|test|dev)$", "", stem, flags=re.I)
    # CMMLU 的目录结构是 data/<学科>/test.csv，学科名在上一级目录
    if stem.lower() in SPLIT_WORDS:
        stem = path.parent.name
    return stem


def collect(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            df = normalize(read_table(p))
        except Exception as exc:  # noqa: BLE001 - 单个文件坏了不影响整体
            print(f"  [跳过] {p.name}: {exc}")
            continue
        df["subject"] = guess_subject(p)
        df["split"] = guess_split(p)
        df["_source_file"] = p.name
        frames.append(df)
    if not frames:
        raise SystemExit("没有读到任何可用 csv，检查 --zip / --dir 指向对不对")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- 清洗
def clean(df: pd.DataFrame, drop_latex: bool) -> pd.DataFrame:
    for col in ["question", "A", "B", "C", "D"]:
        df[col] = df[col].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip().str.upper().str[:1]
    df = df[(df["answer"].isin(list(LETTERS))) & (df["question"] != "")]
    df = df[(df["A"] != "") & (df["B"] != "") & (df["C"] != "") & (df["D"] != "")]
    if drop_latex:
        mask = df[["question", "A", "B", "C", "D"]].apply(
            lambda col: col.str.contains(r"\$|\\frac|\\sqrt|\\int", regex=True)).any(axis=1)
        df = df[~mask]
    return df.drop_duplicates(subset=["question"]).reset_index(drop=True)


def to_schema(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = pd.DataFrame({
        "qid": [f"{source}_{i:05d}" for i in range(len(df))],
        "subject": df["subject"].values,
        "difficulty": "未标注",
        "question": df["question"].values,
        "option_a": df["A"].values,
        "option_b": df["B"].values,
        "option_c": df["C"].values,
        "option_d": df["D"].values,
        "answer": df["answer"].values,
        "source": source,
        "split": df["split"].values,
    })
    return out[OUT_COLUMNS]


# ---------------------------------------------------------------- 抽样
def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """按学科比例分配名额，余数给题最多的学科，保证总数正好是 n。"""
    if n >= len(df):
        return df
    counts = df["subject"].value_counts()
    total = counts.sum()
    quota = {s: int(n * c / total) for s, c in counts.items()}
    remain = n - sum(quota.values())
    for s, _ in counts.items():
        if remain <= 0:
            break
        quota[s] += 1
        remain -= 1

    rng = random.Random(seed)
    picked = []
    for subject, k in quota.items():
        sub = df[df["subject"] == subject]
        idx = sorted(rng.sample(range(len(sub)), min(k, len(sub))))
        picked.append(sub.iloc[idx])
    return pd.concat(picked, ignore_index=True)


# ---------------------------------------------------------------- 自检
def scratch_dir(name: str) -> Path:
    """当前工作目录下的临时目录（不用 tempfile：它建的目录在某些环境下无法再写子目录）。"""
    p = Path(name)
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)
    return p


def selftest() -> int:
    base = scratch_dir(".selftest_ceval")
    try:
        root = base / "ceval-exam"
        for split in ("val", "test"):
            d = root / split
            d.mkdir(parents=True, exist_ok=True)
            rows = ["id,question,A,B,C,D,answer,explanation"]
            for i in range(30):
                rows.append(f'{i},"[{split}]第{i}题：以下说法正确的是？",选项1,选项2,选项3,选项4,{LETTERS[i % 4]},')
            (d / f"computer_network_{split}.csv").write_text("\n".join(rows), encoding="utf-8")

        df = collect(sorted(root.rglob("*.csv")))
        df = clean(df, drop_latex=False)
        assert len(df) == 60, len(df)
        assert set(df["split"]) == {"val", "test"}, set(df["split"])
        assert (df["answer"].isin(list(LETTERS))).all()

        out = to_schema(df[df["split"] == "val"], "ceval")
        assert list(out.columns) == OUT_COLUMNS
        assert len(out) == 30

        sample = stratified_sample(to_schema(df, "ceval"), 20, seed=2027)
        assert len(sample) == 20, len(sample)

        # 无表头 + 学科在目录名上（CMMLU 风格：data/<学科>/test.csv）
        d2 = base / "cmmlu" / "data" / "agronomy"
        d2.mkdir(parents=True, exist_ok=True)
        p = d2 / "test.csv"
        p.write_text("\n".join([f"题干{i},a,b,c,d,{LETTERS[i % 4]}" for i in range(5)]), encoding="utf-8")
        df2 = normalize(read_table(p))
        assert list(df2.columns)[:6] == ["question", "A", "B", "C", "D", "answer"]
        assert guess_subject(p) == "agronomy", guess_subject(p)
        assert guess_split(p) == "test", guess_split(p)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print("✅ import_ceval 自检通过：有表头 / 无表头 / 分层抽样 / 输出 schema 都正常")
    return 0


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="C-Eval / CMMLU → questions.csv")
    ap.add_argument("--zip")
    ap.add_argument("--dir")
    ap.add_argument("--csv")
    ap.add_argument("--n", type=int, default=500, help="抽样题数（0 = 全要）")
    ap.add_argument("--split", default="val", help="val / test / dev / all")
    ap.add_argument("--out", default="data/questions.csv")
    ap.add_argument("--source", default="ceval")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--drop-latex", action="store_true", help="剔除含 $ \\frac 等公式的题目")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    tmpdir = None
    try:
        if args.zip:
            tmpdir = scratch_dir(".tmp_import")
            with zipfile.ZipFile(args.zip) as zf:
                zf.extractall(tmpdir)
            paths = sorted(Path(tmpdir).rglob("*.csv"))
            source_root = Path(args.zip).stem
        elif args.dir:
            paths = sorted(Path(args.dir).rglob("*.csv"))
            source_root = Path(args.dir).name
        elif args.csv:
            paths = [Path(args.csv)]
            source_root = Path(args.csv).stem
        else:
            raise SystemExit("必须指定 --zip / --dir / --csv 之一（或 --selftest）")

        print(f"找到 {len(paths)} 个 csv，开始读取…")
        df = clean(collect(paths), drop_latex=args.drop_latex)
        print(f"清洗后可用题目：{len(df)}")

        if args.split != "all":
            df = df[df["split"] == args.split]
            print(f"只保留 split={args.split}：{len(df)} 题")
            if df.empty:
                raise SystemExit(f"没有 split={args.split} 的题，实际有："
                                 f"{sorted(set(collect(paths)['split']))}")

        out = to_schema(df, args.source)
        if args.n:
            out = stratified_sample(out, args.n, args.seed)
        out = out.sort_values(["subject", "qid"]).reset_index(drop=True)
        out["qid"] = [f"{args.source}_{i:05d}" for i in range(len(out))]

        dst = Path(args.out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(dst, index=False, encoding="utf-8-sig", quoting=1)
        print(f"\n✅ 已写出 {len(out)} 题 → {dst}")
        print("\n学科分布：")
        print(out["subject"].value_counts().to_string())
        return 0
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
