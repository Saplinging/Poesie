import argparse
import glob
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

HAS_OPENAI = False

try:
    from openai import OpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_NUM_SAMPLES = 10
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_TOKENS = 500


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_prompts(prompt_dir: str) -> List[Tuple[str, Dict[str, Any]]]:
    # Keep prompt loading boring and predictable: sorted files, UTF-8 only.
    pattern = os.path.join(prompt_dir, "*.json")
    files = sorted(glob.glob(pattern))
    prompts: List[Tuple[str, Dict[str, Any]]] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                prompts.append((fp, json.load(fh)))
        except Exception as exc:
            logger.warning("Failed to read prompt file %s: %s", fp, exc)
    return prompts


def generate_with_openai(
    prompt_text: str,
    model: str = DEFAULT_OPENAI_MODEL,
    api_key: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    if not HAS_OPENAI:
        raise RuntimeError("OpenAI SDK not installed. Install 'openai' first.")

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment or .env file")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Antworte ausschliesslich mit einem Gedicht."},
            {"role": "user", "content": prompt_text},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def build_sample_meta(
    sample_index: int,
    prompt_id: Optional[str],
    model: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "created_at": utc_now_iso(),
        "sample_index": sample_index,
    }
    if prompt_id:
        meta["prompt-id"] = prompt_id
        meta["prompt_id"] = prompt_id
    meta["backend"] = "openai"
    meta["model"] = model
    meta["temperature"] = temperature
    meta["max_tokens"] = max_tokens
    return meta


def build_generator_meta(
    prompt_id: Optional[str],
    model: str,
    num_samples: int,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "created_at": utc_now_iso(),
        "sampling_mode": "independent_single_prompt_calls",
    }
    if prompt_id:
        meta["prompt-id"] = prompt_id
        meta["prompt_id"] = prompt_id
    meta["num_samples"] = num_samples
    meta["num-samples"] = num_samples
    meta["backend"] = "openai"
    meta["model"] = model
    meta["temperature"] = temperature
    meta["max_tokens"] = max_tokens
    return meta


def main(
    prompt_dir: str = "data/prompt",
    ai_dir: str = "data/ai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    overwrite: bool = False,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> None:
    if not HAS_OPENAI:
        raise RuntimeError("OpenAI SDK not installed. Install 'openai' first.")

    prompts = load_prompts(prompt_dir)
    if not prompts:
        logger.info("No prompts found in %s", prompt_dir)
        return

    if num_samples < 1:
        logger.warning("num_samples=%s is invalid; forcing to 1", num_samples)
        num_samples = 1

    ensure_dir(ai_dir)

    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError("OPENAI_API_KEY not set in environment or .env file")

    model_to_use = model or DEFAULT_OPENAI_MODEL
    temperature = DEFAULT_TEMPERATURE
    max_tokens = DEFAULT_MAX_TOKENS

    for filepath, pdata in prompts:
        human_id = pdata.get("poem-id")
        if not human_id:
            logger.warning("Missing required 'poem-id' in %s -> skipping", os.path.basename(filepath))
            continue

        prompt_id = pdata.get("prompt-id")
        prompt_text = (pdata.get("prompt-text") or "").strip()
        if not prompt_text:
            logger.warning("Empty 'prompt-text' in %s -> skipping", os.path.basename(filepath))
            continue

        ai_id = f"ai_{human_id}"
        out_path = os.path.join(ai_dir, f"{ai_id}.json")

        if os.path.exists(out_path) and not overwrite:
            logger.info("Skipping existing %s (use --overwrite to replace)", out_path)
            continue

        logger.info("Generating %d samples for %s", num_samples, human_id)
        samples: List[Dict[str, Any]] = []

        # Every sample is its own API call, so per-sample metadata stays honest.
        for idx in range(1, num_samples + 1):
            try:
                text = generate_with_openai(
                    prompt_text=prompt_text,
                    model=model_to_use,
                    api_key=resolved_api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                logger.error("OpenAI generation failed for %s sample %d: %s", human_id, idx, exc)
                break

            samples.append(
                {
                    "sample-id": f"{ai_id}_{idx:02d}",
                    "index": idx,
                    "generator": build_sample_meta(
                        sample_index=idx,
                        prompt_id=prompt_id,
                        model=model_to_use,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ),
                    "text": text,
                }
            )

        if not samples:
            logger.warning("No samples generated for %s", human_id)
            continue

        # Keep this aligned with the multi-sample ai_*.json structure used in the repo.
        out_obj: Dict[str, Any] = {
            "ai-poem-id": ai_id,
            "human-poem-id": human_id,
            "type": "ai",
            "generator": build_generator_meta(
                prompt_id=prompt_id,
                model=model_to_use,
                num_samples=len(samples),
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            "samples": samples,
            "text": samples[0]["text"],
            "texts": [sample["text"] for sample in samples],
        }

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(out_obj, fh, ensure_ascii=False, indent=2)
        logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AI poems from prompt JSON files via the OpenAI API.")
    parser.add_argument("--prompt-dir", default="data/prompt", help="Directory with prompt JSON files")
    parser.add_argument("--ai-dir", default="data/ai", help="Output directory for AI JSON poems")
    parser.add_argument(
        "--model",
        default=DEFAULT_OPENAI_MODEL,
        help="OpenAI model to use, for example gpt-4o-mini or gpt-4o",
    )
    parser.add_argument("--api-key", default=None, help="OpenAI API key (if not set in .env)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing AI JSON files")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of independent poems per prompt (default: 10)",
    )
    args = parser.parse_args()

    main(
        prompt_dir=args.prompt_dir,
        ai_dir=args.ai_dir,
        model=args.model,
        api_key=args.api_key,
        overwrite=args.overwrite,
        num_samples=args.num_samples,
    )
