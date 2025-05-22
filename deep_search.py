#!/usr/bin/env python3
"""
Deep contextual search → DuckDuckGo + MarkItDown + Ollama
─────────────────────────────────────────────────────────
AUTO-PLAN MODE (v0.4+) + JSON Schema support

Fixes:
• 404 from Ollama now gives a clear hint.
• Correct import for UnsupportedFormatException.
• Now supports --schema for strict JSON output from Ollama.

USAGE:
python deep_search.py "Tell me about AI safety" --model llama3.2 --auto --schema '{"type":"object","properties":{"risks":{"type":"array","items":{"type":"string"}},"recommendations":{"type":"array","items":{"type":"string"}}},"required":["risks","recommendations"]}'
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import mimetypes
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

import aiohttp
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from markitdown import MarkItDown,UnsupportedFormatException  # corrected import

OLLAMA_URL = "http://localhost:11434/api/generate"
MKD = MarkItDown()

# ────────────── helpers ────────────── #

def _fallback_clean(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript", "iframe", "svg"]):
        t.decompose()
    return " ".join(soup.get_text(" ").split())

async def _fetch_and_convert(session: aiohttp.ClientSession, url: str, timeout: int = 20) -> Tuple[str, str]:
    try:
        async with session.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}) as resp:
            resp.raise_for_status()
            raw = await resp.read()
            suffix = Path(url).suffix or mimetypes.guess_extension(resp.content_type) or ".html"
            filename = f"download{suffix}"
            try:
                md = MKD.convert_stream(io.BytesIO(raw), filename=filename, url=url).markdown
            except UnsupportedFormatException:
                md = _fallback_clean(raw.decode(errors="ignore"))
            return url, md[:8000]
    except Exception:
        return url, ""

async def _gather(urls: List[str]) -> Dict[str, str]:
    texts: Dict[str, str] = {}
    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_and_convert(session, u) for u in urls]
        for coro in asyncio.as_completed(tasks):
            url, md = await coro
            if md:
                texts[url] = md
    return texts

def _load_schema(src: str) -> dict:
    """Load JSON schema from file path or raw JSON string."""
    try:
        # treat as file path
        return json.loads(Path(src).read_text())
    except Exception:
        # treat as raw JSON
        return json.loads(src)

# ────────────── Ollama client ────────────── #

def _ask_ollama(
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    fmt: dict | str | None = None,
    stream: bool = False,
    timeout: int = 300,
) -> str:
    payload = {"model": model, "prompt": prompt, "stream": stream}

    # --- COMPAT shim for pre-0.1.34 servers -------------------------------
    if isinstance(fmt, dict):                 # we have a JSON-Schema
        payload["format"] = "json"            # old servers want a string
        # push the schema into the system prompt so the model sees it
        schema_txt = json.dumps(fmt, separators=(',', ':'))
        system = (system or "") + (
            "\n\n# Follow this JSON schema exactly\n" + schema_txt
        )
    elif fmt is not None:
        payload["format"] = fmt
    # ----------------------------------------------------------------------

    if system:
        payload["system"] = system

    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError:
        raise RuntimeError(
            f"Ollama HTTP {resp.status_code} on {OLLAMA_URL}\n{resp.text}"
        ) from None

    return resp.json().get("response", "").strip()


# ────────────── search core ────────────── #

def _run_ddg(query: str, k: int) -> List[Dict]:
    with DDGS() as ddgs:
        return list(ddgs.text(query, safesearch="Off", max_results=k))

def _deep_search_single(query: str, k: int, model: str, schema: dict | None = None) -> Tuple[str, List[Dict]]:
    results = _run_ddg(query, k)
    urls = [r["href"] for r in results]
    docs = asyncio.run(_gather(urls))

    ctx = "\n\n".join([f"URL: {u}\n\n{textwrap.shorten(t, 8000)}" for u, t in docs.items()])
    prompt = (
        "Answer the QUERY below using ONLY the information in the DOCUMENTS.\n"
        "If the answer is missing, reply 'I don't know'.\n\n"
        f"# QUERY\n{query}\n\n# DOCUMENTS\n{ctx}\n"
    )
    answer = _ask_ollama(model, prompt, fmt=schema)
    return answer, results

# ────────────── auto-plan ────────────── #

import re

def _extract_json_array(text: str) -> str | None:
    """Extract the first JSON array from any string (even with extra text/markdown)."""
    match = re.search(r"\[[\s\S]*\]", text)
    return match.group(0) if match else None



def _auto_plan(
    question: str,
    model: str,
    max_steps: int = 5,
) -> List[Tuple[str, int]]:
    
    sys_prompt = (
        f"You are a research strategist. For the user's question, respond "
        f"with a JSON **array** containing *at most* {max_steps} objects.  "
        f"Each object MUST have:\n"
        f"  • 'query'   : string\n"
        f"  • 'results' : integer (1-10)\n"
        f"Return **ONLY** valid JSON - no markdown, no extra keys."
    )
    raw = _ask_ollama(model, question, system=sys_prompt, fmt=schema)

    # Fast path – model behaved
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        extracted = _extract_json_array(raw)
        if not extracted:
            raise ValueError(f"Expected JSON array, got:\n{raw}")
        data = json.loads(extracted)

    # ── NEW: normalise to a list ─────────────────────────────────────────── #
    if not isinstance(data, list):
        # 1) Wrapper object with a list field?
        if isinstance(data, dict):
            for key in ("plan", "queries", "steps"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                # 2) Single-object response → wrap into a list
                data = [data]
        else:
            raise ValueError(
                f"Model response is neither a list nor a dict:\n{raw}"
            )
    # ─────────────────────────────────────────────────────────────────────── #

    # Normalise & clamp
    plan: List[Tuple[str, int]] = []
    for item in data[:max_steps]:
        if not isinstance(item, dict):
            continue
        q = str(item.get("query", "")).strip()
        try:
            n = int(item.get("results", 1))
        except (TypeError, ValueError):
            n = 1
        n = max(1, min(10, n))
        if q:
            plan.append((q, n))

    return plan
import time

def deep_search(question: str, model: str, num_results: int, auto: bool, schema: dict | None = None):
    if not auto:
        return _deep_search_single(question, num_results, model, schema)
    plan = _auto_plan(question, model)
    total_urls = 0
    combined_ctx: Dict[str, str] = {}

    for subquery, k in plan:
        if total_urls >= 10:
            break
        hits = _run_ddg(subquery, k)
        urls = [h["href"] for h in hits]
        remaining = 10 - total_urls
        limited_urls = urls[:remaining]
        combined_ctx.update(asyncio.run(_gather(limited_urls)))
        total_urls += len(limited_urls)
        time.sleep(2)

    ctx = "\n\n".join([f"URL: {u}\n\n{textwrap.shorten(t, 8000)}" for u, t in combined_ctx.items()])
    prompt = (
        "Answer the QUESTION below using ONLY the information in the DOCUMENTS.\n"
        "If the answer is missing, reply 'I don't know'.\n\n"
        f"# QUESTION\n{question}\n\n# DOCUMENTS\n{ctx}\n"
    )
    answer = _ask_ollama(model, prompt, fmt=schema)
    return answer, plan

# ────────────── CLI ────────────── #
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Deep search via DuckDuckGo + MarkItDown + Ollama")
    p.add_argument("question", help="Natural language question or search query")
    p.add_argument("--model", default="llama3.2", help="Ollama model to use")
    p.add_argument("--num_results", "-k", type=int, default=5, help="DDG results if --auto is off")
    p.add_argument("--auto", action="store_true", help="Let LLM generate sub-queries + k")
    p.add_argument("--schema", help="Path to JSON schema file or raw JSON string for structured output")
    args = p.parse_args()

    schema = _load_schema(args.schema) if args.schema else None
    answer, meta = deep_search(args.question, args.model, args.num_results, args.auto, schema)
    print("\n=== ANSWER ===\n")
    print(answer)

    if args.auto:
        print("\n=== LLM SEARCH PLAN ===")
        print(json.dumps([{"query": q, "results": k} for q, k in meta], indent=2))
    else:
        print("\n=== SOURCES ===")
        for r in meta:
            print(f"{r['title']} - {r['href']}")
