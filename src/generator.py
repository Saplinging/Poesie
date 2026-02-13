import os
import json
import glob
import argparse
import logging
from datetime import datetime, timezone

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

HAS_OPENAI = False
HAS_TRANSFORMERS = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from transformers import pipeline, set_seed
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
# 4) Deutsches Modell als Default für lokalen Fallback:
DEFAULT_LOCAL_MODEL = "dbmdz/german-gpt2"
DEFAULT_LOCAL_SEED = 42


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_with_openai(prompt_text: str, model: str = DEFAULT_OPENAI_MODEL, api_key: str = None,
                         temperature: float = 0.8, max_tokens: int = 500) -> str:
    """Generate poem using OpenAI API."""
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment or .env file")

    client = OpenAI(api_key=api_key)

    system_msg = (
        "Antworte ausschließlich mit einem Gedicht."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt_text}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content.strip()


def build_transformer_generator(model_name: str):
    """Create and return a HF text-generation pipeline (once)."""
    device = 0 if (HAS_TRANSFORMERS and torch.cuda.is_available()) else -1
    logger.info("Loading transformers model: %s (device=%s)", model_name, device)
    return pipeline("text-generation", model=model_name, device=device)


def generate_with_transformers(gen, prompt_text: str, seed: int = DEFAULT_LOCAL_SEED,
                               max_new_tokens: int = 200, top_p: float = 0.9,
                               temperature: float = 0.8) -> str:
    """Generate poem using local HuggingFace model."""
    set_seed(seed)

    out = gen(
        prompt_text,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        num_return_sequences=1,
        truncation=True
    )

    text = out[0]["generated_text"]
    if text.startswith(prompt_text):
        text = text[len(prompt_text):].strip()
    return text.strip()


def fallback_generate(prompt_text: str) -> str:
    """Simple rule-based fallback poem generator."""
    words = [w.strip(".,;:!?") for w in prompt_text.split()]
    keywords = []
    for w in words:
        if w.isalpha() and len(w) > 4:
            keywords.append(w.lower())
        if len(keywords) >= 4:
            break
    if not keywords:
        keywords = words[:4]

    lines = []
    lines.append(f"Inmitten von {keywords[0] if keywords else 'Nacht'}, schlägt mein Herz.")
    lines.append(f"Ein Name: {keywords[1] if len(keywords) > 1 else 'Stimme'}, ein ferner Schmerz.")
    lines.append("Die Straßen sind leer, die Worte schwer.")
    lines.append("Ich sammle Licht in meinen Händen.")
    if len(keywords) > 2:
        lines.append(f"Und denke an {keywords[2]}, und an {keywords[3] if len(keywords) > 3 else 'die Zeit'}.")
    lines.append("So endet nichts — es ist nur ein Atemzug.")
    return "\n".join(lines)


def load_prompts(prompt_dir: str):
    """Load all prompt JSON files from a directory."""
    pattern = os.path.join(prompt_dir, "*.json")
    files = sorted(glob.glob(pattern))
    prompts = []
    for p in files:
        try:
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            prompts.append((p, data))
        except Exception as e:
            logger.warning("Failed to read prompt file %s: %s", p, e)
    return prompts


def ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def main(
    prompt_dir="data/prompt",
    ai_dir="data/ai",
    model=None,
    use_openai=False,
    api_key=None,
    overwrite=False
):
    """Main generation function."""
    prompts = load_prompts(prompt_dir)
    if not prompts:
        logger.info("No prompts found in %s", prompt_dir)
        return

    ensure_dir(ai_dir)

    # Prepare local generator once (if needed)
    local_model = model if (model and not use_openai) else DEFAULT_LOCAL_MODEL
    local_gen = None
    if HAS_TRANSFORMERS:
        try:
            local_gen = build_transformer_generator(local_model)
        except Exception as e:
            logger.warning("Failed to load transformers model '%s': %s", local_model, e)
            local_gen = None

    for filepath, pdata in prompts:
        # 2) poem-id ist Pflicht (sonst skip)
        human_id = pdata.get("poem-id")
        if not human_id:
            logger.warning("Missing required 'poem-id' in %s -> skipping", os.path.basename(filepath))
            continue

        prompt_id = pdata.get("prompt-id")

        prompt_text = pdata.get("prompt-text", "").strip()
        if not prompt_text:
            logger.warning("Empty 'prompt-text' in %s -> skipping", os.path.basename(filepath))
            continue

        logger.info("Processing prompt for human poem-id: %s (%s)", human_id, os.path.basename(filepath))

        generated = ""
        generator_meta = {
            "created_at": utc_now_iso()
        }
        if prompt_id:
            generator_meta["prompt-id"] = prompt_id

        if use_openai:
            if not HAS_OPENAI:
                logger.error("OpenAI SDK not installed. Install 'openai' or run without --use-openai.")
                continue
            try:
                model_to_use = model or DEFAULT_OPENAI_MODEL
                temp = 0.8
                max_toks = 500
                logger.info("Using OpenAI model: %s", model_to_use)
                generated = generate_with_openai(
                    prompt_text,
                    model=model_to_use,
                    api_key=api_key,
                    temperature=temp,
                    max_tokens=max_toks
                )
                generator_meta.update({
                    "backend": "openai",
                    "model": model_to_use,
                    "temperature": temp,
                    "max_tokens": max_toks
                })
            except Exception as e:
                logger.error("OpenAI generation failed for %s: %s", human_id, e)
                logger.info("Falling back to local generation...")

                if HAS_TRANSFORMERS and local_gen is not None:
                    try:
                        seed = DEFAULT_LOCAL_SEED
                        max_new = 500
                        top_p = 0.9
                        temp = 0.8
                        generated = generate_with_transformers(
                            local_gen,
                            prompt_text,
                            seed=seed,
                            max_new_tokens=max_new,
                            top_p=top_p,
                            temperature=temp
                        )
                        generator_meta.update({
                            "backend": "transformers",
                            "model": local_model,
                            "seed": seed,
                            "max_new_tokens": max_new,
                            "top_p": top_p,
                            "temperature": temp
                        })
                    except Exception as e2:
                        logger.warning("Transformer generation also failed for %s: %s", human_id, e2)
                        generated = fallback_generate(prompt_text)
                        generator_meta.update({
                            "backend": "fallback"
                        })
                else:
                    generated = fallback_generate(prompt_text)
                    generator_meta.update({
                        "backend": "fallback"
                    })
        else:
            # Local generation path
            if HAS_TRANSFORMERS and local_gen is not None:
                try:
                    seed = DEFAULT_LOCAL_SEED
                    max_new = 500
                    top_p = 0.9
                    temp = 0.8
                    generated = generate_with_transformers(
                        local_gen,
                        prompt_text,
                        seed=seed,
                        max_new_tokens=max_new,
                        top_p=top_p,
                        temperature=temp
                    )
                    generator_meta.update({
                        "backend": "transformers",
                        "model": local_model,
                        "seed": seed,
                        "max_new_tokens": max_new,
                        "top_p": top_p,
                        "temperature": temp
                    })
                except Exception as e:
                    logger.warning("Transformer generation failed for %s: %s", human_id, e)
                    generated = fallback_generate(prompt_text)
                    generator_meta.update({
                        "backend": "fallback"
                    })
            else:
                generated = fallback_generate(prompt_text)
                generator_meta.update({
                    "backend": "fallback"
                })

        # 1) Stabiler Output-Name: ai_{human_id}.json (statt poem_1, poem_2, ...)
        ai_id = f"ai_{human_id}"
        out_path = os.path.join(ai_dir, f"{ai_id}.json")

        out_obj = {
            "ai-poem-id": ai_id,
            "human-poem-id": human_id,
            "type": "ai",
            # 5) Metadaten für Reproduzierbarkeit
            "generator": generator_meta,
            "text": generated
        }

        if os.path.exists(out_path) and not overwrite:
            logger.info("Skipping existing %s (use --overwrite to replace)", out_path)
            continue

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(out_obj, fh, ensure_ascii=False, indent=2)
        logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AI poems from prompt JSONs and save to data/ai/")
    parser.add_argument("--prompt-dir", default="data/prompt", help="Directory with prompt JSON files")
    parser.add_argument("--ai-dir", default="data/ai", help="Output directory for AI JSON poems")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI API instead of local models")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model to use. "
            "OpenAI: e.g. gpt-4o-mini, gpt-4o | "
            "Local (default: dbmdz/german-gpt2): any HF text-generation model id"
        )
    )
    parser.add_argument("--api-key", default=None, help="OpenAI API key (if not set in .env)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing AI JSON files")
    args = parser.parse_args()

    main(
        prompt_dir=args.prompt_dir,
        ai_dir=args.ai_dir,
        model=args.model,
        use_openai=args.use_openai,
        api_key=args.api_key,
        overwrite=args.overwrite
    )
