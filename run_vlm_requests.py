#!/usr/bin/env python3
"""
Run VLM requests from JSONL and save outputs.

Input:  dataset_v2/vlm/vlm_requests.jsonl
Output: dataset_v2/vlm/vlm_outputs.jsonl
"""

import os
import json
import base64
import time
from typing import Dict, Any, Optional
from pathlib import Path

from tqdm import tqdm
from openai import OpenAI

from physai_vlm.taxonomy import JUDGE_SCHEMA, TAG_SCHEMA


MODEL = os.getenv("VLM_MODEL", "gpt-4o-mini")
INPUT_JSONL = os.getenv("INPUT_JSONL", "dataset_v2/vlm/vlm_requests.jsonl")
OUTPUT_JSONL = os.getenv("OUTPUT_JSONL", "dataset_v2/vlm/vlm_outputs.jsonl")
MAX_ITEMS = int(os.getenv("MAX_ITEMS", "0"))
SKIP_EXISTING = True
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))


def img_to_data_url(path: str) -> str:
    p = Path(path)
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    ext = p.suffix.lower().replace(".", "")
    mime = "image/png" if ext == "png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                return None
    return None


def validate_output(rtype: str, out: Dict[str, Any]) -> bool:
    if rtype == "judge":
        keys = {"p_success", "progress", "uncertainty", "judge_notes"}
    else:
        keys = {"stage", "failure_type", "confidence", "notes"}
    return isinstance(out, dict) and keys.issubset(out.keys())


def load_existing_keys(out_path: str):
    keys = set()
    if not os.path.exists(out_path):
        return keys
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                keys.add((r.get("type"), r.get("episode_id"), r.get("t"), r.get("image_path")))
            except Exception:
                continue
    return keys


def extract_output_text(resp) -> str:
    out_text = getattr(resp, "output_text", None)
    if out_text is not None:
        return out_text
    text = ""
    try:
        for item in resp.output:
            if item.type == "message":
                for c in item.content:
                    if c.type in ("output_text", "text"):
                        text += c.text
    except Exception:
        text = str(resp)
    return text


def call_with_retry(client, rtype: str, prompt: str, data_url: str) -> Dict[str, Any]:
    schema = JUDGE_SCHEMA if rtype == "judge" else TAG_SCHEMA
    name = "judge" if rtype == "judge" else "tagging"

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=MODEL,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": data_url},
                        ],
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "strict": True,
                        "schema": schema,
                        "name": name,
                    }
                },
            )
            out_text = extract_output_text(resp)
            parsed = safe_json_parse(out_text)
            if parsed and validate_output(rtype, parsed):
                return parsed
            last_error = "schema_validation_failed"
        except Exception as e:
            last_error = str(e)

        time.sleep(2 ** (attempt - 1))

    return {"raw_text": f"error: {last_error}"}


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI()
    existing = load_existing_keys(OUTPUT_JSONL) if SKIP_EXISTING else set()
    Path(os.path.dirname(OUTPUT_JSONL)).mkdir(parents=True, exist_ok=True)

    n = 0
    with open(INPUT_JSONL, "r", encoding="utf-8") as fin, open(OUTPUT_JSONL, "a", encoding="utf-8") as fout:
        lines = fin.readlines()
        for line in tqdm(lines, desc="VLM requests"):
            req = json.loads(line)
            rtype = req["type"]
            episode_id = req["episode_id"]
            t = req["t"]
            image_path = req["image_path"]
            prompt = req["prompt"]

            key = (rtype, episode_id, t, image_path)
            if SKIP_EXISTING and key in existing:
                continue

            data_url = img_to_data_url(image_path)
            parsed = call_with_retry(client, rtype, prompt, data_url)

            out_rec = {
                "type": rtype,
                "episode_id": episode_id,
                "t": t,
                "image_path": image_path,
                "model": MODEL,
                "output": parsed,
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()

            n += 1
            if MAX_ITEMS > 0 and n >= MAX_ITEMS:
                break

    print(f"[DONE] wrote outputs to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
