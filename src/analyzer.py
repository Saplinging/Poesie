from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import logging
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analyzer")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

HAS_NUMPY = False
HAS_MPL = False
HAS_SKLEARN = False
HAS_TORCH = False
HAS_TRANSFORMERS = False
HAS_PERPLEXFERN = False

try:
    import numpy as np  # type: ignore

    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt  # type: ignore

    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    import torch  # type: ignore

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

try:
    import perplexfern  # type: ignore

    HAS_PERPLEXFERN = True
except Exception:
    HAS_PERPLEXFERN = False


DEFAULT_PPL_MODEL = "dbmdz/german-gpt2"
DEFAULT_PERPLEXFERN_METRICS = ["perplexity", "entropy", "ttr"]
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", re.UNICODE)

PROMPT_STOPWORDS = {
    "du",
    "bist",
    "ein",
    "eine",
    "im",
    "in",
    "und",
    "der",
    "die",
    "das",
    "von",
    "zu",
    "mit",
    "waehrend",
    "dein",
    "deine",
    "deiner",
    "deinem",
    "deinen",
    "deines",
    "deutscher",
    "deutsche",
    "dichter",
    "dichterin",
    "schriftsteller",
    "schreibe",
    "gedicht",
    "jahr",
    "lebst",
    "erlebst",
    "entsteht",
    "einer",
    "einem",
    "einen",
    "dem",
    "den",
    "des",
    "unter",
    "gegen",
    "als",
    "nach",
    "nimmt",
    "durch",
    "nur",
    "innerer",
    "inneren",
    "innerlich",
    "starkem",
    "starke",
    "starken",
    "schreiben",
    "zeit",
    "situation",
    "sich",
    "sind",
    "ist",
    "wird",
    "vom",
    "zur",
    "zum",
    "oder",
    "auf",
    "an",
    "klar",
    "klare",
    "aus",
    "maennlicher",
    "politischer",
    "politisch",
    "engagierter",
    "junger",
    "oesterreichischer",
    "deutsch-juedische",
    "deutsch-juedischer",
    "juedischer",
    "schreibst",
    "setzt",
    "dich",
    "auseinander",
    "entspringt",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def write_text(path: str, text: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    if not rows:
        write_text(path, "")
        return
    cols = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def maybe_fix_mojibake(text: str) -> str:
    # Some source texts arrived with mojibake, so we fix the obvious case once here.
    if "Ã" in text or "Â" in text:
        try:
            return text.encode("latin1").decode("utf-8")
        except Exception:
            return text
    return text


def normalize_text(text: str) -> str:
    t = maybe_fix_mojibake(text or "")
    t = (
        t.replace("\u00e4", "ae")
        .replace("\u00f6", "oe")
        .replace("\u00fc", "ue")
        .replace("\u00df", "ss")
        .replace("\u00c4", "Ae")
        .replace("\u00d6", "Oe")
        .replace("\u00dc", "Ue")
    )
    return t


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_lines(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def tokenize_words(text: str) -> List[str]:
    t = normalize_text(text).lower()
    return WORD_RE.findall(t)


def ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None and not math.isnan(v) and not math.isinf(v)]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def std(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None and not math.isnan(v) and not math.isinf(v)]
    if len(vals) < 2:
        return 0.0 if len(vals) == 1 else None
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return float(math.sqrt(var))


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def moving_ttr(tokens: List[str], window: int = 20) -> float:
    if not tokens:
        return 0.0
    if len(tokens) < window:
        return len(set(tokens)) / len(tokens)
    vals = []
    for i in range(len(tokens) - window + 1):
        win = tokens[i : i + window]
        vals.append(len(set(win)) / window)
    return float(sum(vals) / len(vals)) if vals else 0.0


def token_entropy(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    c = Counter(tokens)
    total = len(tokens)
    out = 0.0
    for freq in c.values():
        p = freq / total
        out -= p * math.log2(p)
    return float(out)


def repetition_rate(lines: List[str]) -> float:
    if not lines:
        return 0.0
    norm = [normalize_ws(ln).lower() for ln in lines]
    c = Counter(norm)
    repeated = sum(v - 1 for v in c.values() if v > 1)
    return float(repeated / len(norm))


def compression_ratio(text: str) -> float:
    raw = (text or "").encode("utf-8")
    raw_len = max(len(raw), 1)
    return float(len(gzip.compress(raw)) / raw_len)


def punctuation_density(text: str) -> float:
    t = text or ""
    chars = re.sub(r"\s+", "", t)
    denom = max(len(chars), 1)
    punct_chars = set(".,;:!?-()[]\"'…")
    punct_count = sum(1 for ch in chars if ch in punct_chars)
    return float(punct_count / denom)


def compute_text_metrics(text: str) -> Dict[str, float]:
    tokens = tokenize_words(text)
    lines = split_lines(text)
    bigrams = ngrams(tokens, 2)
    n_words = len(tokens)
    n_types = len(set(tokens))
    # Plain TTR is short-text sensitive, so we keep MATTR alongside it.
    return {
        "n_chars": float(len(text or "")),
        "n_words": float(n_words),
        "n_types": float(n_types),
        "n_lines": float(len(lines)),
        "avg_words_per_line": float(n_words / len(lines)) if lines else 0.0,
        "ttr": float(n_types / n_words) if n_words else 0.0,
        "mattr": moving_ttr(tokens, window=20),
        "distinct_1": float(len(set(tokens)) / n_words) if n_words else 0.0,
        "distinct_2": float(len(set(bigrams)) / len(bigrams)) if bigrams else 0.0,
        "token_entropy": token_entropy(tokens),
        "repetition_rate": repetition_rate(lines),
        "compression_ratio": compression_ratio(text),
        "punct_density": punctuation_density(text),
    }


def extract_prompt_keywords(prompt_text: str, limit: int = 12) -> List[str]:
    tokens = tokenize_words(prompt_text)
    out: List[str] = []
    seen = set()
    for tok in tokens:
        if tok in PROMPT_STOPWORDS:
            continue
        if len(tok) < 5 and not tok.isdigit():
            continue
        if tok in seen:
            continue
        out.append(tok)
        seen.add(tok)
        if len(out) >= limit:
            break
    return out


def prompt_keyword_coverage(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    wset = set(tokenize_words(text))
    hits = sum(1 for k in keywords if k in wset)
    return float(hits / len(keywords))


def jaccard_wordset(a: str, b: str) -> float:
    sa = set(tokenize_words(a))
    sb = set(tokenize_words(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def tfidf_cosine(a: str, b: str) -> Optional[float]:
    if not HAS_SKLEARN:
        return None
    vec = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform([normalize_text(a), normalize_text(b)])
    sim = (X[0] @ X[1].T).toarray()[0][0]
    return float(sim)


def modified_precision(hyp: List[str], refs: List[List[str]], n: int) -> float:
    hyp_ng = Counter(ngrams(hyp, n))
    if not hyp_ng:
        return 0.0
    max_ref_counts: Dict[Tuple[str, ...], int] = {}
    for ref in refs:
        ref_ng = Counter(ngrams(ref, n))
        for ng, cnt in ref_ng.items():
            max_ref_counts[ng] = max(max_ref_counts.get(ng, 0), cnt)
    clipped = sum(min(cnt, max_ref_counts.get(ng, 0)) for ng, cnt in hyp_ng.items())
    total = sum(hyp_ng.values())
    return clipped / total if total else 0.0


def brevity_penalty(hyp_len: int, ref_lens: List[int]) -> float:
    if hyp_len == 0:
        return 0.0
    if not ref_lens:
        return 1.0
    closest = min(ref_lens, key=lambda r: (abs(r - hyp_len), r))
    if hyp_len > closest:
        return 1.0
    return math.exp(1.0 - (closest / hyp_len))


def sentence_bleu2(hyp: List[str], refs: List[List[str]]) -> float:
    if not hyp or not refs:
        return 0.0
    p1 = modified_precision(hyp, refs, 1)
    p2 = modified_precision(hyp, refs, 2)
    eps = 1e-12
    bp = brevity_penalty(len(hyp), [len(r) for r in refs])
    return float(bp * math.exp(0.5 * math.log(max(p1, eps)) + 0.5 * math.log(max(p2, eps))))


def self_bleu2(texts: List[str]) -> float:
    tokenized = [tokenize_words(t) for t in texts if normalize_ws(t)]
    if len(tokenized) < 2:
        return 0.0
    vals = []
    for i, hyp in enumerate(tokenized):
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i]
        vals.append(sentence_bleu2(hyp, refs))
    return float(sum(vals) / len(vals)) if vals else 0.0


def set_distinct2(texts: List[str]) -> float:
    bigrams_all: List[Tuple[str, ...]] = []
    for text in texts:
        toks = tokenize_words(text)
        bigrams_all.extend(ngrams(toks, 2))
    if not bigrams_all:
        return 0.0
    return float(len(set(bigrams_all)) / len(bigrams_all))


@dataclass
class HumanPoem:
    id: str
    title: Optional[str]
    author: Optional[str]
    year: Optional[int]
    context: Dict[str, Any]
    text: str
    source_path: str


@dataclass
class AISample:
    ai_poem_id: str
    human_poem_id: str
    sample_id: str
    sample_index: int
    text: str
    generator: Dict[str, Any]
    source_path: str


class PerplexityScorer:
    def __init__(self, model_name: str = DEFAULT_PPL_MODEL):
        if not (HAS_TRANSFORMERS and HAS_TORCH):
            raise RuntimeError("transformers/torch not installed")
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading perplexity model %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.cache: Dict[str, Optional[float]] = {}

    @torch.no_grad()
    def perplexity(self, text: str) -> Optional[float]:
        key = normalize_ws(text)
        if not key:
            return None
        if key in self.cache:
            return self.cache[key]
        try:
            enc = self.tokenizer(key, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = enc["input_ids"].to(self.device)
            out = self.model(input_ids=input_ids, labels=input_ids)
            loss = float(out.loss.item())
            ppl = float(math.exp(loss))
            if math.isnan(ppl) or math.isinf(ppl):
                ppl = None
        except Exception as exc:
            logger.warning("Perplexity failed: %s", exc)
            ppl = None
        self.cache[key] = ppl
        return ppl


def generate_perplexfern_image(text: str, out_path: str, metric: str) -> bool:
    if not HAS_PERPLEXFERN:
        return False
    try:
        ensure_dir(os.path.dirname(out_path))
        perplexfern.analyse(text, save_path=out_path, metric=metric)  # type: ignore
        return True
    except Exception as exc:
        logger.warning("perplexFern failed for %s (%s): %s", out_path, metric, exc)
        return False


def load_humans(human_dir: str) -> Dict[str, HumanPoem]:
    out: Dict[str, HumanPoem] = {}
    for fp in sorted(glob.glob(os.path.join(human_dir, "*.json"))):
        try:
            data = read_json(fp)
            hid = str(data.get("id") or "").strip()
            if not hid:
                logger.warning("Skipping human poem without id: %s", fp)
                continue
            year = data.get("year")
            out[hid] = HumanPoem(
                id=hid,
                title=data.get("title"),
                author=data.get("author"),
                year=int(year) if str(year).isdigit() else None,
                context=data.get("context") or {},
                text=data.get("text") or "",
                source_path=fp,
            )
        except Exception as exc:
            logger.warning("Failed reading human poem %s: %s", fp, exc)
    return out


def load_prompt_keywords(prompt_dir: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for fp in sorted(glob.glob(os.path.join(prompt_dir, "*.json"))):
        try:
            data = read_json(fp)
            hid = data.get("poem-id")
            if not hid:
                continue
            prompt_text = data.get("prompt-text") or ""
            out[str(hid)] = {
                "prompt_id": data.get("prompt-id"),
                "prompt_text": prompt_text,
                "keywords": extract_prompt_keywords(prompt_text),
                "source_path": fp,
            }
        except Exception as exc:
            logger.warning("Failed reading prompt %s: %s", fp, exc)
    return out


def merge_generator_meta(global_meta: Dict[str, Any], sample_meta: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(global_meta or {})
    merged.update(sample_meta or {})
    return merged


def load_ai_samples(ai_dir: str) -> List[AISample]:
    out: List[AISample] = []
    for fp in sorted(glob.glob(os.path.join(ai_dir, "*.json"))):
        try:
            data = read_json(fp)
            ai_poem_id = str(data.get("ai-poem-id") or data.get("id") or os.path.splitext(os.path.basename(fp))[0])
            human_poem_id = str(data.get("human-poem-id") or data.get("poem-id") or "")
            if not human_poem_id:
                logger.warning("Skipping AI file without human-poem-id: %s", fp)
                continue
            global_meta = data.get("generator") or {}
            samples = data.get("samples")
            # Prefer the multi-sample schema, but keep a soft landing for older files.
            if isinstance(samples, list) and samples:
                for idx, sample in enumerate(samples, start=1):
                    text = sample.get("text") or ""
                    sid = str(sample.get("sample-id") or f"{ai_poem_id}_{idx:02d}")
                    sidx = int(sample.get("index") or idx)
                    smeta = merge_generator_meta(global_meta, sample.get("generator") or {})
                    out.append(
                        AISample(
                            ai_poem_id=ai_poem_id,
                            human_poem_id=human_poem_id,
                            sample_id=sid,
                            sample_index=sidx,
                            text=text,
                            generator=smeta,
                            source_path=fp,
                        )
                    )
            else:
                text = data.get("text") or ""
                out.append(
                    AISample(
                        ai_poem_id=ai_poem_id,
                        human_poem_id=human_poem_id,
                        sample_id=f"{ai_poem_id}_01",
                        sample_index=1,
                        text=text,
                        generator=global_meta,
                        source_path=fp,
                    )
                )
        except Exception as exc:
            logger.warning("Failed reading AI file %s: %s", fp, exc)
    return out


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("$", r"\$")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def export_ch4_4_table(pair_rows: List[Dict[str, Any]], out_csv: str, out_tex: str) -> None:
    table: List[Dict[str, Any]] = []
    for row in pair_rows:
        human_ppl = row.get("human_ppl")
        ai_mean = row.get("ai_mean_ppl")
        ai_std = row.get("ai_std_ppl")
        if human_ppl is None or ai_mean is None:
            comment = "n/a"
        elif ai_mean > human_ppl:
            comment = "KI > Mensch"
        elif ai_mean < human_ppl:
            comment = "KI < Mensch"
        else:
            comment = "gleich"
        table.append(
            {
                "poem_id": row.get("human_id"),
                "human_ppl": human_ppl,
                "ai_ppl_mean": ai_mean,
                "ai_ppl_std": ai_std,
                "comment": comment,
            }
        )
    write_csv(out_csv, table)

    header = (
        "\\begin{tabular}{lrrrrl}\n"
        "\\hline\n"
        "Gedicht-ID & Mensch-PPL & KI-PPL (M) & KI-PPL (SD) & \\; & Kommentar\\\\\n"
        "\\hline\n"
    )

    def fmt(x: Any) -> str:
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "n/a"

    body = []
    for row in table:
        body.append(
            f"{latex_escape(str(row['poem_id']))} & {fmt(row['human_ppl'])} & {fmt(row['ai_ppl_mean'])} & {fmt(row['ai_ppl_std'])} &  & {latex_escape(str(row['comment']))}\\\\\n"
        )
    footer = "\\hline\n\\end{tabular}\n"
    write_text(out_tex, header + "".join(body) + footer)


@dataclass
class MetricSpec:
    key: str
    label: str
    lower_is_better: bool = False


def configure_plot_style() -> None:
    if not HAS_MPL:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )


def _values_for_human(sample_rows: List[Dict[str, Any]], human_id: str, key: str) -> List[float]:
    vals = []
    for r in sample_rows:
        if r.get("human_id") != human_id:
            continue
        v = r.get(key)
        if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v):
            vals.append(float(v))
    return vals


def plot_metric_distribution(
    sample_rows: List[Dict[str, Any]],
    pair_rows: List[Dict[str, Any]],
    spec: MetricSpec,
    out_path: str,
) -> Optional[str]:
    if not HAS_MPL:
        return None
    order = [r["human_id"] for r in pair_rows]
    ai_series = [_values_for_human(sample_rows, hid, f"ai_{spec.key}") for hid in order]
    human_vals = [r.get(f"human_{spec.key}") for r in pair_rows]
    if not any(ai_series):
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    positions = list(range(1, len(order) + 1))
    bp = ax.boxplot(ai_series, positions=positions, widths=0.56, patch_artist=True, showfliers=False)
    for box in bp["boxes"]:
        box.set(facecolor="#8ecae6", alpha=0.45, edgecolor="#1d3557")
    for median in bp["medians"]:
        median.set(color="#1d3557", linewidth=1.4)

    rnd = random.Random(7)
    for pos, vals in zip(positions, ai_series):
        jitter = [(rnd.random() - 0.5) * 0.25 for _ in vals]
        xs = [pos + j for j in jitter]
        ax.scatter(xs, vals, s=22, color="#1d3557", alpha=0.55, linewidths=0)

    human_x, human_y = [], []
    for pos, val in zip(positions, human_vals):
        if isinstance(val, (int, float)) and not math.isnan(val) and not math.isinf(val):
            human_x.append(pos)
            human_y.append(float(val))
    if human_x:
        ax.scatter(human_x, human_y, marker="D", s=36, color="#d62828", label="Human")

    ax.set_xticks(positions)
    ax.set_xticklabels(order, rotation=40, ha="right")
    ax.set_title(f"{spec.label}: AI-Sample-Verteilung pro Gedicht")
    ax.set_ylabel(spec.label)
    ax.legend(loc="upper right")
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_metric_delta(pair_rows: List[Dict[str, Any]], spec: MetricSpec, out_path: str) -> Optional[str]:
    if not HAS_MPL:
        return None
    labels, deltas, errs = [], [], []
    for row in pair_rows:
        hv = row.get(f"human_{spec.key}")
        am = row.get(f"ai_mean_{spec.key}")
        sd = row.get(f"ai_std_{spec.key}")
        if hv is None or am is None:
            continue
        labels.append(row["human_id"])
        deltas.append(float(am) - float(hv))
        errs.append(float(sd) if isinstance(sd, (int, float)) else 0.0)
    if not deltas:
        return None

    x = list(range(len(deltas)))
    colors = ["#e76f51" if d >= 0 else "#2a9d8f" for d in deltas]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x, deltas, yerr=errs, color=colors, alpha=0.85, capsize=3)
    ax.axhline(0.0, color="#111111", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel(f"Delta (AI mean - Human) in {spec.label}")
    ax.set_title(f"{spec.label}: Mittelwertdifferenz pro Gedicht")
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_metric_scatter(pair_rows: List[Dict[str, Any]], spec: MetricSpec, out_path: str) -> Optional[str]:
    if not HAS_MPL:
        return None
    xvals, yvals, yerr, labels = [], [], [], []
    for row in pair_rows:
        hv = row.get(f"human_{spec.key}")
        am = row.get(f"ai_mean_{spec.key}")
        sd = row.get(f"ai_std_{spec.key}")
        if hv is None or am is None:
            continue
        xvals.append(float(hv))
        yvals.append(float(am))
        yerr.append(float(sd) if isinstance(sd, (int, float)) else 0.0)
        labels.append(row["human_id"])
    if not xvals:
        return None
    lo, hi = min(min(xvals), min(yvals)), max(max(xvals), max(yvals))

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.errorbar(xvals, yvals, yerr=yerr, fmt="o", color="#264653", ecolor="#457b9d", alpha=0.9)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#6c757d", linewidth=1)
    for x, y, label in zip(xvals, yvals, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel(f"Human {spec.label}")
    ax.set_ylabel(f"AI mean {spec.label}")
    ax.set_title(f"{spec.label}: Human vs AI-Mittelwert")
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def zscore(values: List[float]) -> List[float]:
    if not values:
        return []
    m = sum(values) / len(values)
    var = sum((x - m) ** 2 for x in values) / max(len(values) - 1, 1)
    sd = math.sqrt(var)
    if sd == 0:
        return [0.0 for _ in values]
    return [(x - m) / sd for x in values]


def plot_delta_heatmap(pair_rows: List[Dict[str, Any]], specs: List[MetricSpec], out_path: str) -> Optional[str]:
    if not (HAS_MPL and HAS_NUMPY):
        return None
    labels = [r["human_id"] for r in pair_rows]
    if not labels:
        return None
    cols = []
    valid_specs = []
    for spec in specs:
        deltas = []
        ok = True
        for row in pair_rows:
            hv = row.get(f"human_{spec.key}")
            am = row.get(f"ai_mean_{spec.key}")
            if hv is None or am is None:
                ok = False
                break
            deltas.append(float(am) - float(hv))
        if ok and deltas:
            cols.append(zscore(deltas))
            valid_specs.append(spec)
    if not cols:
        return None
    mat = np.array(cols).T
    fig, ax = plt.subplots(figsize=(1.9 * len(valid_specs) + 3, 0.45 * len(labels) + 2.5))
    im = ax.imshow(mat, cmap="coolwarm", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(list(range(len(valid_specs))))
    ax.set_xticklabels([s.key for s in valid_specs], rotation=35, ha="right")
    ax.set_yticks(list(range(len(labels))))
    ax.set_yticklabels(labels)
    ax.set_title("Z-standardisierte Delta-Matrix (AI mean - Human)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("z-score")
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_self_bleu(pair_rows: List[Dict[str, Any]], out_path: str) -> Optional[str]:
    if not HAS_MPL:
        return None
    labels, values = [], []
    for row in pair_rows:
        v = row.get("ai_self_bleu2")
        if v is None:
            continue
        labels.append(row["human_id"])
        values.append(float(v))
    if not values:
        return None
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.bar(range(len(values)), values, color="#6a4c93", alpha=0.85)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Self-BLEU-2 (higher = less diverse)")
    ax.set_title("Diversitaet pro Prompt-Set (AI) via Self-BLEU-2")
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path))
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def build_metric_specs() -> List[MetricSpec]:
    specs = [
        MetricSpec("ppl", "Perplexity (dbmdz/german-gpt2)", lower_is_better=True),
        MetricSpec("distinct_2", "Distinct-2"),
        MetricSpec("mattr", "MATTR"),
        MetricSpec("token_entropy", "Token-Entropy (bits)"),
        MetricSpec("repetition_rate", "Repetition Rate", lower_is_better=True),
        MetricSpec("compression_ratio", "Compression Ratio (gzip/raw)", lower_is_better=True),
        MetricSpec("prompt_keyword_coverage", "Prompt Keyword Coverage"),
    ]
    if HAS_SKLEARN:
        specs.append(MetricSpec("tfidf_cosine", "TF-IDF Cosine to Human"))
    return specs


def generate_plots(
    sample_rows: List[Dict[str, Any]],
    pair_rows: List[Dict[str, Any]],
    results_dir: str,
    specs: List[MetricSpec],
) -> Dict[str, Any]:
    plots_dir = os.path.join(results_dir, "plots")
    ensure_dir(plots_dir)
    configure_plot_style()

    index: Dict[str, Any] = {
        "plots_dir": "results/plots",
        "generated": [],
        "notes": [
            "dist: AI sample distribution (box+jitter) with human reference marker",
            "delta: AI mean minus human per poem",
            "scatter: Human value vs AI mean per poem",
        ],
    }

    for spec in specs:
        base = os.path.join(plots_dir, spec.key)
        files = [
            plot_metric_distribution(sample_rows, pair_rows, spec, base + "__dist.png"),
            plot_metric_delta(pair_rows, spec, base + "__delta.png"),
            plot_metric_scatter(pair_rows, spec, base + "__scatter.png"),
        ]
        existing_files = [f for f in files if f]
        if existing_files:
            index["generated"].append(
                {
                    "metric": spec.key,
                    "label": spec.label,
                    "lower_is_better": spec.lower_is_better,
                    "files": existing_files,
                }
            )

    heatmap_path = plot_delta_heatmap(pair_rows, specs, os.path.join(plots_dir, "delta_heatmap.png"))
    self_bleu_path = plot_self_bleu(pair_rows, os.path.join(plots_dir, "self_bleu2__bar.png"))
    if heatmap_path:
        index["generated"].append({"metric": "delta_heatmap", "files": [heatmap_path]})
    if self_bleu_path:
        index["generated"].append({"metric": "self_bleu2", "files": [self_bleu_path]})

    write_json(os.path.join(plots_dir, "index.json"), index)
    readme = [
        "Plots (Kap. 4.4)\n",
        "- *_dist.png: AI-Verteilung je Gedicht (Box + Jitter), Human als Marker\n",
        "- *_delta.png: Mittelwertdifferenz (AI mean - Human) je Gedicht\n",
        "- *_scatter.png: Human vs AI-Mittelwert je Gedicht\n",
        "- delta_heatmap.png: Z-standardisierte Delta-Uebersicht ueber mehrere Metriken\n",
        "- self_bleu2__bar.png: Diversitaet innerhalb der AI-Samples je Prompt\n",
        "\n",
        "Hinweis: Perplexity ist modellabhaengig. Quantitative Werte sind Indikatoren.\n",
    ]
    write_text(os.path.join(plots_dir, "README.txt"), "".join(readme))
    return index


def analyze(
    human_dir: str,
    ai_dir: str,
    prompt_dir: str,
    results_dir: str,
    ppl_model: str = DEFAULT_PPL_MODEL,
    compute_ppl: bool = True,
    generate_perplexfern_images: bool = True,
    perplexfern_metrics: Optional[List[str]] = None,
) -> int:
    ensure_dir(results_dir)
    ensure_dir(os.path.join(results_dir, "perplexfern_outputs"))

    humans = load_humans(human_dir)
    prompts = load_prompt_keywords(prompt_dir)
    ai_samples = load_ai_samples(ai_dir)
    if not humans:
        logger.error("No human poems found in %s", human_dir)
        return 2
    if not ai_samples:
        logger.error("No AI samples found in %s", ai_dir)
        return 2

    ai_by_human: Dict[str, List[AISample]] = defaultdict(list)
    for sample in ai_samples:
        ai_by_human[sample.human_poem_id].append(sample)

    scorer: Optional[PerplexityScorer] = None
    if compute_ppl:
        try:
            scorer = PerplexityScorer(model_name=ppl_model)
        except Exception as exc:
            logger.warning("Perplexity disabled: %s", exc)
            scorer = None

    sample_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    pf_generated = 0
    pf_failed = 0
    pf_metrics = perplexfern_metrics or list(DEFAULT_PERPLEXFERN_METRICS)

    ordered_humans = sorted(humans.values(), key=lambda h: (h.year if h.year is not None else 9999, h.id))
    for human in ordered_humans:
        # This is the core pairing step: one human poem against all linked AI samples.
        samples = sorted(ai_by_human.get(human.id, []), key=lambda s: (s.sample_index, s.sample_id))
        if not samples:
            logger.warning("No AI samples for human poem %s", human.id)
            continue

        human_metrics = compute_text_metrics(human.text)
        human_ppl = scorer.perplexity(human.text) if scorer is not None else None
        prompt_info = prompts.get(human.id) or {}
        prompt_keywords = prompt_info.get("keywords") or []
        human_kw_cov = prompt_keyword_coverage(human.text, prompt_keywords)

        pair_id = f"{human.id}__{samples[0].ai_poem_id}"
        pair_sample_rows: List[Dict[str, Any]] = []

        for sample in samples:
            ai_metrics = compute_text_metrics(sample.text)
            ai_ppl = scorer.perplexity(sample.text) if scorer is not None else None
            row: Dict[str, Any] = {
                "pair_id": pair_id,
                "human_id": human.id,
                "human_title": human.title,
                "human_author": human.author,
                "human_year": human.year,
                "ai_poem_id": sample.ai_poem_id,
                "ai_sample_id": sample.sample_id,
                "ai_sample_index": sample.sample_index,
                "ai_model": sample.generator.get("model"),
                "ai_backend": sample.generator.get("backend"),
                "ai_temperature": safe_float(sample.generator.get("temperature")),
                "ai_created_at": sample.generator.get("created_at"),
                "prompt_id": sample.generator.get("prompt_id")
                or sample.generator.get("prompt-id")
                or prompt_info.get("prompt_id"),
                "prompt_keyword_count": len(prompt_keywords),
                "human_prompt_keyword_coverage": human_kw_cov,
                "ai_prompt_keyword_coverage": prompt_keyword_coverage(sample.text, prompt_keywords),
                "jaccard_wordset": jaccard_wordset(human.text, sample.text),
                "tfidf_cosine": tfidf_cosine(human.text, sample.text),
                "human_ppl": human_ppl,
                "ai_ppl": ai_ppl,
                "human_source": human.source_path,
                "ai_source": sample.source_path,
            }
            for k, v in human_metrics.items():
                row[f"human_{k}"] = v
            for k, v in ai_metrics.items():
                row[f"ai_{k}"] = v
            sample_rows.append(row)
            pair_sample_rows.append(row)

        ai_texts = [s.text for s in samples]
        ai_self_bleu = self_bleu2(ai_texts)
        ai_set_d2 = set_distinct2(ai_texts)
        for r in pair_sample_rows:
            r["ai_self_bleu2"] = ai_self_bleu
            r["ai_set_distinct2"] = ai_set_d2

        pair_row: Dict[str, Any] = {
            "pair_id": pair_id,
            "human_id": human.id,
            "human_title": human.title,
            "human_author": human.author,
            "human_year": human.year,
            "ai_poem_id": samples[0].ai_poem_id,
            "prompt_id": pair_sample_rows[0].get("prompt_id"),
            "n_ai_samples": len(samples),
            "ai_self_bleu2": ai_self_bleu,
            "ai_set_distinct2": ai_set_d2,
        }

        keys_for_pair = [
            "ppl",
            "n_words",
            "n_lines",
            "ttr",
            "mattr",
            "distinct_1",
            "distinct_2",
            "token_entropy",
            "repetition_rate",
            "compression_ratio",
            "punct_density",
            "prompt_keyword_coverage",
            "tfidf_cosine",
            "jaccard_wordset",
        ]

        for key in keys_for_pair:
            if key in {"tfidf_cosine", "jaccard_wordset"}:
                hcol = None
                acol = key
            elif key == "prompt_keyword_coverage":
                hcol = "human_prompt_keyword_coverage"
                acol = "ai_prompt_keyword_coverage"
            elif key == "ppl":
                hcol = "human_ppl"
                acol = "ai_ppl"
            else:
                hcol = f"human_{key}"
                acol = f"ai_{key}"

            pair_row[f"human_{key}"] = pair_sample_rows[0].get(hcol) if hcol else None
            vals = []
            for r in pair_sample_rows:
                v = r.get(acol)
                if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v):
                    vals.append(float(v))
            pair_row[f"ai_mean_{key}"] = mean(vals)
            pair_row[f"ai_std_{key}"] = std(vals)
        pair_rows.append(pair_row)

        if generate_perplexfern_images and HAS_PERPLEXFERN:
            rep = pair_sample_rows[0]
            ppl_vals = [(r, r.get("ai_ppl")) for r in pair_sample_rows if isinstance(r.get("ai_ppl"), (int, float))]
            # Use the sample nearest to the mean perplexity as the representative plot candidate.
            if ppl_vals and pair_row.get("ai_mean_ppl") is not None:
                target = float(pair_row["ai_mean_ppl"])
                rep = min(ppl_vals, key=lambda t: abs(float(t[1]) - target))[0]
            rep_sample = next((s for s in samples if s.sample_id == rep.get("ai_sample_id")), samples[0])
            pf_base = os.path.join(results_dir, "perplexfern_outputs", pair_id)
            for metric in pf_metrics:
                out_h = os.path.join(pf_base, f"human__{metric}.png")
                out_a = os.path.join(pf_base, f"ai_rep__{metric}.png")
                ok_h = generate_perplexfern_image(human.text, out_h, metric)
                ok_a = generate_perplexfern_image(rep_sample.text, out_a, metric)
                if ok_h and ok_a:
                    pf_generated += 1
                else:
                    pf_failed += 1

    if not sample_rows or not pair_rows:
        logger.error("No analyzable pairs found")
        return 3

    write_csv(os.path.join(results_dir, "metrics_samples.csv"), sample_rows)
    write_json(os.path.join(results_dir, "metrics_samples.json"), sample_rows)
    write_csv(os.path.join(results_dir, "metrics_pairs.csv"), pair_rows)
    write_json(os.path.join(results_dir, "metrics_pairs.json"), pair_rows)
    write_csv(os.path.join(results_dir, "metrics.csv"), pair_rows)
    write_json(os.path.join(results_dir, "metrics.json"), pair_rows)

    export_ch4_4_table(
        pair_rows,
        os.path.join(results_dir, "ch4_4_table.csv"),
        os.path.join(results_dir, "ch4_4_table.tex"),
    )

    specs = build_metric_specs()
    plot_index = generate_plots(sample_rows, pair_rows, results_dir, specs) if HAS_MPL else {"generated": []}

    summary: Dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "n_human_poems": len(humans),
        "n_humans_with_ai": len(pair_rows),
        "n_ai_samples": len(sample_rows),
        "samples_per_human_mean": len(sample_rows) / max(len(pair_rows), 1),
        "ppl_model": ppl_model if scorer is not None else None,
        "ppl_enabled": scorer is not None,
        "perplexfern_installed": HAS_PERPLEXFERN,
        "perplexfern_images_generated": pf_generated,
        "perplexfern_images_failed": pf_failed,
        "plots_generated_metrics": [x.get("metric") for x in plot_index.get("generated", [])],
        "note": "Quantitative Werte sind Indikatoren und werden mit Close Reading kombiniert.",
    }
    with_ppl = [r for r in pair_rows if r.get("human_ppl") is not None and r.get("ai_mean_ppl") is not None]
    if with_ppl:
        higher = sum(1 for r in with_ppl if r["ai_mean_ppl"] > r["human_ppl"])
        lower = sum(1 for r in with_ppl if r["ai_mean_ppl"] < r["human_ppl"])
        equal = len(with_ppl) - higher - lower
        summary.update(
            {
                "pairs_with_ppl": len(with_ppl),
                "ai_mean_ppl_higher_than_human": higher,
                "ai_mean_ppl_lower_than_human": lower,
                "ai_mean_ppl_equal_to_human": equal,
            }
        )
    write_json(os.path.join(results_dir, "summary.json"), summary)

    logger.info("Done. Wrote results to %s", results_dir)
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze human vs AI poems (multi-sample aware)")
    p.add_argument("--human-dir", default="data/human", help="Directory with human JSON files")
    p.add_argument("--ai-dir", default="data/ai", help="Directory with AI JSON files")
    p.add_argument("--prompt-dir", default="data/prompt", help="Directory with prompt JSON files")
    p.add_argument("--results-dir", default="results", help="Output directory")
    p.add_argument("--ppl-model", default=DEFAULT_PPL_MODEL, help="HF model for perplexity scoring")
    p.add_argument("--no-ppl", action="store_true", help="Disable perplexity computation")
    p.add_argument("--no-perplexfern-images", action="store_true", help="Disable perplexFern exports")
    p.add_argument(
        "--perplexfern-metrics",
        default=",".join(DEFAULT_PERPLEXFERN_METRICS),
        help="Comma-separated perplexFern metrics (e.g. perplexity,entropy,ttr)",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    pf_metrics = [m.strip() for m in args.perplexfern_metrics.split(",") if m.strip()]
    code = analyze(
        human_dir=args.human_dir,
        ai_dir=args.ai_dir,
        prompt_dir=args.prompt_dir,
        results_dir=args.results_dir,
        ppl_model=args.ppl_model,
        compute_ppl=(not args.no_ppl),
        generate_perplexfern_images=(not args.no_perplexfern_images),
        perplexfern_metrics=pf_metrics,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
