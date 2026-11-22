# -*- coding: utf-8 -*-
"""提示词扰动算子库（对应《研究方案》2.3 节的 6 类算子）。

一句话原则：**每个算子只改一处**，构成单因素对照；
"选项重排"在理论上对准确率零影响，用来给所有结果表定一个"噪声地板"。

用法：
    from perturbations import Question, make_variant, ALL_OPERATORS
    q = Question("q001", "数学", "中等", "下列说法正确的是？", ["1", "2", "3", "4"], "B")
    v = make_variant(q, "synonym", seed=2027)
"""

from __future__ import annotations

import hashlib
import random
import re
import sys
from dataclasses import dataclass, field

try:  # Windows 控制台默认 GBK，打印 ✅ 会直接崩，这里统一成 UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

# ---------------------------------------------------------------- 基础结构

LETTERS = "ABCD"

SYSTEM_PROMPT = "你是一名严谨的考试答题助手。请仔细阅读题目，理解题意后作答。"

BASE_INSTRUCTION = (
    "请从 A / B / C / D 中选择唯一正确的一项，"
    "并在最后一行只输出：答案：X（该行不要写任何其他内容）"
)

COT_INSTRUCTION = (
    "请先简要写出你的推理过程，"
    "然后在最后一行单独输出：答案：X（该行不要写任何其他内容）"
)

# 算子清单：第一个是基线，其余 6 个是扰动
BASELINE_OPERATOR = "original"
ALL_OPERATORS = [
    "original",
    "synonym",          # 词汇级：同义替换
    "word_order",       # 句法级：语序调整
    "irrelevant",       # 语境级：无关信息插入
    "option_reorder",   # 结构级：选项重排（零扰动对照）
    "role_prefix",      # 指令级：角色前缀
    "format_constraint",  # 指令级：输出格式约束（要求先推理）
]

OPERATOR_CN = {
    "original": "原始题",
    "synonym": "同义替换",
    "word_order": "语序调整",
    "irrelevant": "无关信息插入",
    "option_reorder": "选项重排(零扰动对照)",
    "role_prefix": "角色前缀",
    "format_constraint": "格式约束",
}

OPERATOR_LAYER = {
    "original": "基线",
    "synonym": "词汇",
    "word_order": "句法",
    "irrelevant": "语境",
    "option_reorder": "结构",
    "role_prefix": "指令",
    "format_constraint": "指令",
}

# 理论上"语义应保持"的算子（选项重排单独处理，它连标点都不该变）
SEMANTIC_PRESERVING = {"synonym", "word_order", "irrelevant", "option_reorder"}


@dataclass
class Question:
    """一道原始题。"""

    qid: str
    subject: str
    difficulty: str
    question: str
    options: list[str]
    answer: str  # 大写字母 A/B/C/D
    source: str = ""
    split: str = "test"

    def __post_init__(self) -> None:
        self.answer = str(self.answer).strip().upper()[:1]
        if self.answer not in LETTERS[: len(self.options)]:
            raise ValueError(f"{self.qid}: 答案 {self.answer!r} 不在选项范围内")


@dataclass
class Variant:
    """一个"题目条件"：某道题的某个扰动版本。"""

    vid: str
    qid: str
    operator: str
    question: str
    options: list[str]
    answer: str
    instruction: str = BASE_INSTRUCTION
    note: str = ""
    ok: bool = True
    meta: dict = field(default_factory=dict)

    def corrected_options(self) -> list[str]:
        return [f"{LETTERS[i]}. {opt}" for i, opt in enumerate(self.options)]


def build_messages(variant: Variant) -> list[dict[str, str]]:
    """统一的 prompt 模板：模板版本号写进 runs.csv，改模板必须升版本。"""
    lines = [variant.question.strip(), ""]
    lines.extend(variant.corrected_options())
    lines.append("")
    lines.append(variant.instruction)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


# v1.1：强制"最后一行只输出：答案：X"，配合更大的 max_tokens（默认 8192）修复回复被截断导致的解析失败
PROMPT_TEMPLATE_VERSION = "v1.1"


# ---------------------------------------------------------------- 工具函数

def _stable_seed(*parts: object) -> int:
    """跨进程稳定的随机种子（Python 内置 hash 对字符串是随机化的，不能用）。"""
    raw = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8], 16)


def _split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[。！？])", text) if s.strip()]


# ---------------------------------------------------------------- 各算子

# 只收录"换成同义词后题意不变"的高置信词对。宁少勿滥：
# 换错一个词，整个实验的结论就错了，所以这里刻意保守。
SYNONYMS = {
    "下列": "以下",
    "以下": "下列",
    "正确": "准确",
    "错误": "不正确",
    "主要": "核心",
    "原因": "缘由",
    "因为": "由于",
    "所以": "因此",
    "导致": "引起",
    "影响": "作用",
    "提高": "提升",
    "降低": "下降",
    "选择": "挑选",
    "描述": "表述",
    "计算": "算出",
    "判断": "判别",
    "说明": "阐明",
    "表明": "显示",
    "属于": "归入",
    "包括": "涵盖",
    "通常": "一般",
    "必须": "务必",
    "能够": "可以",
    "具有": "具备",
}

_IRRELEVANT_SENTENCES = [
    "（背景补充）这份材料在不同年份的印刷版本中，页边距曾做过微调。",
    "（背景补充）出题时所参考的教材有多个版本，各版本封面颜色不一。",
    "（背景补充）本题所属的题库在整理时按学科字母顺序排列。",
    "（背景补充）类似的题目在往年的试卷中也出现过，分值略有不同。",
    "（背景补充）命题组在讨论本题时，会议记录用的是纸质手稿。",
]

_ROLE_TEMPLATES = [
    "你是一位资深的{subject}领域专家，请以专业视角审慎作答。",
    "你是一名{subject}方向的大学教师，正在批改一份考卷，请认真作答。",
    "假设你是一位严谨的{subject}研究者，请给出你最有把握的判断。",
]


def op_synonym(q: Question, rng: random.Random) -> Variant:
    text = q.question
    hits: list[str] = []
    produced: set[str] = set()  # 防止 A→B 之后又被 B→A 换回来
    for word, repl in SYNONYMS.items():
        if word in produced or repl in produced:
            continue
        if word in text:
            text = text.replace(word, repl, 1)
            produced.add(repl)
            hits.append(f"{word}→{repl}")
            if len(hits) >= 3:
                break
    if not hits:
        return Variant(f"{q.qid}__synonym", q.qid, "synonym", q.question, list(q.options), q.answer,
                       ok=False, note="题库中没有可替换的同义词，判为 skip")
    return Variant(f"{q.qid}__synonym", q.qid, "synonym", text, list(q.options), q.answer,
                   note="替换：" + "、".join(hits))


def op_word_order(q: Question, rng: random.Random) -> Variant:
    sentences = _split_sentences(q.question)
    if not sentences:
        return Variant(f"{q.qid}__word_order", q.qid, "word_order", q.question, list(q.options),
                       q.answer, ok=False, note="无法切分句子")
    first = sentences[0]
    parts = [p for p in re.split(r"(?<=，)", first) if p.strip()]
    if len(parts) >= 2:
        new_first = "".join([parts[1], parts[0]] + parts[2:])
        new_text = new_first + "".join(sentences[1:])
        if new_text.strip() and new_text != q.question:
            return Variant(f"{q.qid}__word_order", q.qid, "word_order", new_text,
                           list(q.options), q.answer, note="首句前两个分句对调")
    if len(sentences) >= 2:
        new_text = sentences[1] + sentences[0] + "".join(sentences[2:])
        if new_text.strip() and new_text != q.question:
            return Variant(f"{q.qid}__word_order", q.qid, "word_order", new_text,
                           list(q.options), q.answer, note="前两句对调")
    return Variant(f"{q.qid}__word_order", q.qid, "word_order", q.question, list(q.options),
                   q.answer, ok=False, note="句子结构不支持安全换序，判为 skip")


def op_irrelevant(q: Question, rng: random.Random) -> Variant:
    filler = _IRRELEVANT_SENTENCES[rng.randrange(len(_IRRELEVANT_SENTENCES))]
    return Variant(f"{q.qid}__irrelevant", q.qid, "irrelevant", f"{filler}\n\n{q.question}",
                   list(q.options), q.answer, note="前置 1 句无关背景")


def op_option_reorder(q: Question, rng: random.Random) -> Variant:
    """零扰动对照：选项顺序变化，题意与答案等价。"""
    n = len(q.options)
    order = list(range(n))
    for _ in range(10):  # 避免洗出原顺序（那就不是扰动了）
        rng.shuffle(order)
        if order != list(range(n)):
            break
    new_options = [q.options[i] for i in order]
    old_idx = LETTERS.index(q.answer)
    new_idx = order.index(old_idx)
    return Variant(f"{q.qid}__option_reorder", q.qid, "option_reorder", q.question,
                   new_options, LETTERS[new_idx],
                   note=f"选项顺序 {'ABCD'[:n]} → " + "".join(LETTERS[i] for i in order))


def op_role_prefix(q: Question, rng: random.Random) -> Variant:
    template = _ROLE_TEMPLATES[rng.randrange(len(_ROLE_TEMPLATES))]
    prefix = template.format(subject=q.subject or "该学科")
    return Variant(f"{q.qid}__role_prefix", q.qid, "role_prefix", f"{prefix}\n\n{q.question}",
                   list(q.options), q.answer, note="前置角色设定句")


def op_format_constraint(q: Question, rng: random.Random) -> Variant:
    return Variant(f"{q.qid}__format_constraint", q.qid, "format_constraint", q.question,
                   list(q.options), q.answer, instruction=COT_INSTRUCTION,
                   note="要求先写推理过程再给答案（CoT 式格式约束）")


_OPERATOR_FUNCS = {
    "original": lambda q, rng: Variant(f"{q.qid}__original", q.qid, "original", q.question,
                                        list(q.options), q.answer, note="基线，未扰动"),
    "synonym": op_synonym,
    "word_order": op_word_order,
    "irrelevant": op_irrelevant,
    "option_reorder": op_option_reorder,
    "role_prefix": op_role_prefix,
    "format_constraint": op_format_constraint,
}


def make_variant(q: Question, operator: str, seed: int = 2027) -> Variant:
    """生成某道题的某个扰动版本。同参数必然得到同结果（可复现）。"""
    if operator not in _OPERATOR_FUNCS:
        raise KeyError(f"未知算子：{operator}，可选：{ALL_OPERATORS}")
    rng = random.Random(_stable_seed(q.qid, operator, seed))
    return _OPERATOR_FUNCS[operator](q, rng)


def make_all_variants(q: Question, seed: int = 2027) -> list[Variant]:
    return [make_variant(q, op, seed) for op in ALL_OPERATORS]


# ---------------------------------------------------------------- 自检

def _selftest() -> None:
    q = Question("q001", "数学", "中等", "某公司2024年的营收为100万元，利润为20万元，则利润率下列描述正确的是？",
                 ["20%", "25%", "80%", "5%"], "A")
    variants = make_all_variants(q)
    assert len(variants) == 7
    for v in variants:
        if not v.ok:
            print(f"  skip: {v.operator} ({v.note})")
            continue
        assert v.answer in LETTERS[: len(v.options)], v.operator
        if v.operator == "option_reorder":
            # 零扰动对照：重排后正确选项的文本必须与原题一致
            assert v.options[LETTERS.index(v.answer)] == q.options[LETTERS.index(q.answer)]
        if v.operator in ("synonym", "word_order", "irrelevant", "role_prefix"):
            assert v.options == q.options, v.operator
        msgs = build_messages(v)
        assert msgs[-1]["content"].strip()
    # 可复现性
    assert make_variant(q, "synonym").question == make_variant(q, "synonym").question
    print("✅ perturbations 自检通过")


if __name__ == "__main__":
    _selftest()
