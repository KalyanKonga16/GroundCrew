"""
Enterprise CrewAI Research Pipeline — LiteLLM + Groq (Free-Tier Safe)
----------------------------------------------------------------------
- UI-safe import: nothing heavy runs at module load
- 4 sequential crews (Planner → Validator → Extractor → Writer)
- Groq via LiteLLM: model = "groq/llama-3.3-70b-versatile"
- All LLM calls tuned to stay under Groq free-tier 12,000 TPM limit
- URL fetching is parallelized with ThreadPoolExecutor to minimize I/O wait
"""

import os
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from crewai import Agent, Task, Crew, Process, LLM
from tavily import TavilyClient

load_dotenv()

# =========================
# CONFIGURATION (Free-Tier TPM Safe)
# =========================
_RAW_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
LLM_MODEL = f"groq/{_RAW_MODEL.removeprefix('groq/').removeprefix('openai/')}"

MAX_QUERIES = 5
MAX_SEARCH_RESULTS_TOTAL = 10
MAX_VALIDATED = 4
MAX_FETCH_CHARS = 4000
PDF_MAX_PAGES = 3
TAVILY_RESULTS_PER_QUERY = 2
FETCH_TIMEOUT = 20

PLANNER_TOKENS = 600
VALIDATOR_TOKENS = 1400
EXTRACTOR_TOKENS = 2500
WRITER_TOKENS = 4000  # slightly reduced to fit under 12K TPM window

# Seconds to sleep between heavy LLM stages to let Groq's sliding TPM window reset
GROQ_INTER_STAGE_SLEEP = 20

# =========================
# PYDANTIC SCHEMAS (manual parse only)
# =========================
class QueryList(BaseModel):
    queries: List[str] = Field(default_factory=list)

class SourceItem(BaseModel):
    rank: int = Field(default=0)
    title: str = Field(default="")
    url: str = Field(default="")
    score: float = Field(default=0.0)
    reason: str = Field(default="")

class ValidatedSources(BaseModel):
    sources: List[SourceItem] = Field(default_factory=list)

class QuoteItem(BaseModel):
    quote: str = Field(default="")
    location_hint: str = Field(default="")

class EvidenceItem(BaseModel):
    id: int = Field(default=0)
    title: str = Field(default="")
    url: str = Field(default="")
    datasets: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    key_findings: List[str] = Field(default_factory=list)
    quotes: List[QuoteItem] = Field(default_factory=list)

    @field_validator("datasets", "metrics", "key_findings", mode="before")
    @classmethod
    def _coerce_to_strings(cls, v):
        """If the LLM returns dicts/objects, coerce them to JSON strings so Pydantic stays happy."""
        if not isinstance(v, list):
            return []
        out: List[str] = []
        for item in v:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                parts = [f"{kk}: {vv}" for kk, vv in item.items()]
                out.append(" | ".join(parts))
            else:
                out.append(str(item))
        return out

class EvidenceCollection(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)

class ReportOutput(BaseModel):
    report: str = Field(default="")

# =========================
# LAZY LLM FACTORY (LiteLLM → Groq)
# =========================
@lru_cache(maxsize=None)
def _get_llm(max_tokens: int = 1200) -> LLM:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing or empty. Check your .env file and restart the app."
        )
    return LLM(
        model=LLM_MODEL,
        api_key=api_key,
        temperature=0.2,
        max_tokens=max_tokens,
    )

# =========================
# LAZY AGENT FACTORIES
# =========================
def _get_planner_agent() -> Agent:
    return Agent(
        role="Senior Research Planner",
        goal="Decompose a research topic into 3–5 precise academic search queries.",
        backstory=(
            "You are an expert research strategist. You break topics into "
            "foundational concepts, recent methods, benchmarks, applications, and open problems. "
            "You always return compact, valid JSON and never add conversational text."
        ),
        llm=_get_llm(PLANNER_TOKENS),
        allow_delegation=False,
        verbose=False,
    )

def _get_validator_agent() -> Agent:
    return Agent(
        role="Academic Source Validator",
        goal="Select the top 4 most credible and technically deep sources from search results.",
        backstory=(
            "You are a discerning research librarian. You score sources on credibility, recency, "
            "and technical depth. You always return compact, valid JSON and never add conversational text."
        ),
        llm=_get_llm(VALIDATOR_TOKENS),
        allow_delegation=False,
        verbose=False,
    )

def _get_extractor_agent() -> Agent:
    return Agent(
        role="Evidence Extractor",
        goal="Extract only grounded facts, datasets, metrics, and verbatim quotes from raw text.",
        backstory=(
            "You are a meticulous analytical reader. You never hallucinate. You extract exactly "
            "what is present in the text and return compact, valid JSON. Never add conversational text."
        ),
        llm=_get_llm(EXTRACTOR_TOKENS),
        allow_delegation=False,
        verbose=False,
    )

def _get_writer_agent() -> Agent:
    return Agent(
        role="Senior Research Synthesizer",
        goal="Write a rigorous Markdown research report with inline citations using only the provided evidence.",
        backstory=(
            "You are a senior technology analyst. You write clear, structured Markdown reports. "
            "Every claim is cited with [1], [2], etc. If evidence is insufficient, you say so explicitly. "
            "Return ONLY valid JSON with a single key 'report'. Never add conversational text."
        ),
        llm=_get_llm(WRITER_TOKENS),
        allow_delegation=False,
        verbose=False,
    )

# =========================
# HELPERS
# =========================
def _now_ms() -> int:
    return int(time.time() * 1000)


def _extract_json(text: str) -> Any:
    """Robust JSON extraction from markdown/string output."""
    if not text:
        raise ValueError("Empty response")

    text = text.strip()
    # Strip markdown fences
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    raise ValueError(f"Could not parse JSON from response: {text[:400]}")


def _safe_parse_crew_result(raw_result: Any, pydantic_cls):
    """Unwrap CrewOutput.raw if present, parse JSON, validate with Pydantic."""
    if isinstance(raw_result, pydantic_cls):
        return raw_result

    if hasattr(raw_result, "raw"):
        s = str(raw_result.raw)
    else:
        s = str(raw_result)

    try:
        data = _extract_json(s)
        if isinstance(data, dict):
            return pydantic_cls(**data)
        raise ValueError("Parsed JSON is not a dict")
    except Exception as e:
        raise ValueError(
            f"Failed to parse {pydantic_cls.__name__}: {e}\nRaw snippet: {s[:600]}"
        ) from e


def _to_dict(obj: BaseModel) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()  # type: ignore
    return obj.dict()  # type: ignore


def _domain_score(url: str) -> float:
    u = (url or "").lower()
    score = 0.0
    if "arxiv.org" in u:
        score += 2.5
    if "aclanthology.org" in u or "aclweb.org" in u:
        score += 2.0
    if "ieee.org" in u:
        score += 2.0
    if "springer.com" in u or "sciencedirect.com" in u:
        score += 1.5
    if ".edu" in u:
        score += 1.0
    if "github.com" in u:
        score += 1.0
    return score


def _tavily_search(queries: List[str]) -> List[Dict[str, Any]]:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY is missing. Check your .env file.")

    client = TavilyClient(api_key=key)
    all_results: List[Dict[str, Any]] = []

    for q in queries:
        try:
            resp = client.search(
                query=q,
                search_depth="basic",
                max_results=TAVILY_RESULTS_PER_QUERY,
                include_answer=False,
            )
            all_results.extend(resp.get("results", []))
            time.sleep(0.3)
        except Exception as e:
            print("Tavily error:", e)

    seen: set = set()
    dedup: List[Dict[str, Any]] = []
    for r in all_results:
        url = r.get("url", "").strip()
        if url and url not in seen:
            seen.add(url)
            dedup.append({
                "title": r.get("title", "Untitled"),
                "url": url,
                "snippet": (r.get("content", "") or "")[:600],
                "domain_score": round(_domain_score(url), 2),
            })

    dedup.sort(key=lambda x: x.get("domain_score", 0.0), reverse=True)
    return dedup[:MAX_SEARCH_RESULTS_TOTAL]


def _fetch_url_text(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchPipeline/1.0)"}
    try:
        r = requests.get(url, headers=headers, timeout=FETCH_TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        return f"[Fetch error: {str(e)}]"

    ctype = (r.headers.get("Content-Type") or "").lower()
    is_pdf = url.lower().endswith(".pdf") or "application/pdf" in ctype

    if is_pdf:
        try:
            import pdfplumber
            with pdfplumber.open(BytesIO(r.content)) as pdf:
                text = "\n".join(
                    (page.extract_text() or "") for page in pdf.pages[:PDF_MAX_PAGES]
                )
            return text[:MAX_FETCH_CHARS]
        except Exception as e:
            return f"[PDF parse error: {str(e)}]"

    soup = BeautifulSoup(r.content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return text[:MAX_FETCH_CHARS]


# =========================
# CREW TASK BUILDERS
# =========================
def _build_planner_task(topic: str) -> Task:
    return Task(
        description=(
            f"Topic: {topic}\n\n"
            "Generate exactly 3 to 5 precise academic search queries covering:\n"
            "1. Foundational concepts\n"
            "2. Recent methods / algorithms\n"
            "3. Benchmarks / datasets\n"
            "4. Applications\n"
            "5. Open problems / limitations\n\n"
            "IMPORTANT RULES:\n"
            "- Return ONLY a JSON object. No markdown, no explanations, no greetings.\n"
            "- The JSON must match this exact schema:\n"
            '{\n  "queries": ["q1", "q2", "q3", "q4", "q5"]\n}'
        ),
        expected_output="A JSON object containing a queries array.",
        agent=_get_planner_agent(),
    )


def _build_validator_task(search_results: List[Dict[str, Any]]) -> Task:
    payload = json.dumps(search_results, ensure_ascii=False, indent=2)
    return Task(
        description=(
            f"Evaluate these {len(search_results)} search results and select the top {MAX_VALIDATED} "
            "best sources based on credibility, recency, and technical depth.\n\n"
            "Score each source 0–10. Provide a concise reason.\n\n"
            f"Input results:\n{payload}\n\n"
            "IMPORTANT RULES:\n"
            "- Return ONLY a JSON object. No markdown, no explanations, no greetings.\n"
            "- The JSON must match this exact schema:\n"
            '{\n  "sources": [\n    {"rank": 1, "title": "...", "url": "...", "score": 9.2, "reason": "..."}\n  ]\n}'
        ),
        expected_output=f"A JSON object with a sources array of at most {MAX_VALIDATED} items.",
        agent=_get_validator_agent(),
    )


def _build_extractor_task(sources_with_text: List[Dict[str, Any]]) -> Task:
    payload = json.dumps(sources_with_text, ensure_ascii=False, indent=2)
    return Task(
        description=(
            "Extract grounded evidence from each source below.\n\n"
            "For every source, extract thoroughly:\n"
            "- datasets (up to 5, plain strings only — e.g., 'Natural Questions (n=300K)'. NEVER return objects/dicts.)\n"
            "- metrics (up to 5, plain strings only — e.g., 'Exact Match: 45–52%'. NEVER return objects/dicts.)\n"
            "- key_findings (up to 5, detailed sentences, not one-liners)\n"
            "- quotes (up to 2; each quote max 300 chars, verbatim)\n\n"
            "IMPORTANT RULES:\n"
            "- Return ONLY a JSON object. No markdown, no explanations, no greetings.\n"
            "- The JSON must match this exact schema:\n"
            '{\n  "evidence": [\n    {\n      "id": 1,\n      "title": "...",\n      "url": "...",\n      "datasets": ["string1", "string2"],\n      "metrics": ["string1", "string2"],\n      "key_findings": ["string1", "string2"],\n      "quotes": [{"quote": "...", "location_hint": "..."}]\n    }\n  ]\n}\n\n'
            "- Do NOT hallucinate. Extract exactly what is present.\n"
            "- datasets and metrics MUST be plain strings, never JSON objects.\n\n"
            f"Sources:\n{payload}"
        ),
        expected_output="A JSON object with an evidence array.",
        agent=_get_extractor_agent(),
    )


def _build_writer_task(topic: str, evidence_json: str) -> Task:
    return Task(
        description=(
            f"Write a comprehensive, detailed Markdown research report on: {topic}\n\n"
            "Use ONLY the provided evidence.\n"
            "Every important claim must have an inline citation like [1], [2], etc.\n\n"
            "Structure (write multiple paragraphs per section where evidence supports it). "
            "Do NOT include a References or Bibliography section — it is appended automatically after your output.\n\n"
            "- Executive Summary (2–3 paragraphs)\n"
            "- Background (2–4 paragraphs with definitions, history, and context)\n"
            "- Methods & Approaches (3–5 paragraphs; describe specific architectures, algorithms, and training regimes cited in evidence)\n"
            "- Benchmarks / Datasets / Metrics (2–4 paragraphs with exact numbers, score ranges, and comparison tables if supported)\n"
            "- Key Findings (3–5 paragraphs; synthesize trends, trade-offs, and reproducibility notes)\n"
            "- Open Problems & Risks (2–3 paragraphs; be specific about failure modes and gaps)\n"
            "- Recommendations (1–2 paragraphs of actionable, evidence-backed next steps)\n\n"
            f"Evidence JSON:\n{evidence_json}\n\n"
            "If evidence is insufficient for a section, explicitly say so rather than inventing content.\n\n"
            "IMPORTANT RULES:\n"
            "- Return ONLY a JSON object. No markdown around the JSON, no explanations, no greetings.\n"
            "- The report content goes inside the JSON string; use \\n for newlines.\n"
            "- Be detailed and thorough. Do not artificially shorten sections.\n"
            "- The JSON must match this exact schema:\n"
            '{\n  "report": "# Executive Summary\\n\\n..."\n}'
        ),
        expected_output="A comprehensive Markdown research report with inline citations.",
        agent=_get_writer_agent(),
    )


# =========================
# MAIN PIPELINE
# =========================
def run_research_pipeline(topic: str, status_cb=None) -> Tuple[str, Dict[str, Any]]:
    def _status(msg: str, pct: int):
        if status_cb:
            status_cb(msg, pct)

    debug: Dict[str, Any] = {"topic": topic, "timestamps": {}}

    # 1) PLANNER CREW
    _status("🧠 Planning your research...", 10)
    queries: List[str] = []
    debug["timestamps"]["planner_start_ms"] = _now_ms()
    try:
        crew_plan = Crew(
            agents=[_get_planner_agent()],
            tasks=[_build_planner_task(topic)],
            process=Process.sequential,
            verbose=False,
        )
        plan_raw = crew_plan.kickoff()
        plan_obj: QueryList = _safe_parse_crew_result(plan_raw, QueryList)
        queries = [str(q).strip() for q in plan_obj.queries if str(q).strip()][:MAX_QUERIES]
        if len(queries) < 3:
            raise RuntimeError(f"Planner returned too few queries: {queries}")
    except Exception as e:
        raise RuntimeError(f"Planner stage failed: {e}") from e
    finally:
        debug["timestamps"]["planner_end_ms"] = _now_ms()
        debug["queries"] = queries

    time.sleep(0.5)

    # 2) SEARCH (deterministic, no LLM tokens)
    _status("🔍 Searching the web for sources...", 30)
    debug["timestamps"]["search_start_ms"] = _now_ms()
    try:
        search_results = _tavily_search(queries)
        if not search_results:
            raise RuntimeError("No search results returned from Tavily.")
    except Exception as e:
        raise RuntimeError(f"Search stage failed: {e}") from e
    finally:
        debug["timestamps"]["search_end_ms"] = _now_ms()
        debug["search_results"] = search_results

    time.sleep(0.5)

    # 3) VALIDATOR CREW
    _status("✅ Reviewing and picking the best sources...", 50)
    validated: List[SourceItem] = []
    debug["timestamps"]["validator_start_ms"] = _now_ms()
    try:
        crew_val = Crew(
            agents=[_get_validator_agent()],
            tasks=[_build_validator_task(search_results)],
            process=Process.sequential,
            verbose=False,
        )
        val_raw = crew_val.kickoff()
        val_obj: ValidatedSources = _safe_parse_crew_result(val_raw, ValidatedSources)
        validated = val_obj.sources[:MAX_VALIDATED]
        if not validated:
            raise RuntimeError("Validator returned zero sources.")
    except Exception as e:
        raise RuntimeError(f"Validator stage failed: {e}") from e
    finally:
        debug["timestamps"]["validator_end_ms"] = _now_ms()
        debug["validated_sources"] = [_to_dict(s) for s in validated]

    time.sleep(0.5)

    # 4) FETCH — parallelized to eliminate sequential network wait
    _status("📄 Reading the selected sources...", 65)

    def _fetch_one(idx: int, src: SourceItem) -> Tuple[int, Optional[Dict[str, Any]]]:
        url = src.url.strip()
        title = src.title.strip() or "Untitled"
        if not url:
            return idx, None
        raw_text = _fetch_url_text(url)
        return idx, {"id": idx, "title": title, "url": url, "text": raw_text}

    sources_with_text: List[Dict[str, Any]] = []
    if validated:
        with ThreadPoolExecutor(max_workers=min(4, len(validated))) as executor:
            future_to_idx = {
                executor.submit(_fetch_one, i, src): i
                for i, src in enumerate(validated, start=1)
            }
            results_map: Dict[int, Optional[Dict[str, Any]]] = {}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    _, result = future.result()
                    results_map[idx] = result
                except Exception as exc:
                    src = validated[idx - 1]
                    results_map[idx] = {
                        "id": idx,
                        "title": src.title.strip() or "Untitled",
                        "url": src.url.strip(),
                        "text": f"[Fetch exception: {exc}]",
                    }
            # Reassemble in original order, skipping empty URLs
            for i in range(1, len(validated) + 1):
                item = results_map.get(i)
                if item is not None:
                    sources_with_text.append(item)

    # 5) EXTRACTOR CREW
    _status("📄 Pulling out key facts and quotes...", 75)
    evidence: List[EvidenceItem] = []
    debug["timestamps"]["extractor_start_ms"] = _now_ms()
    try:
        crew_ext = Crew(
            agents=[_get_extractor_agent()],
            tasks=[_build_extractor_task(sources_with_text)],
            process=Process.sequential,
            verbose=False,
        )
        ext_raw = crew_ext.kickoff()
        ext_obj: EvidenceCollection = _safe_parse_crew_result(ext_raw, EvidenceCollection)
        evidence = ext_obj.evidence
        if not evidence:
            raise RuntimeError("Extractor returned zero evidence items.")
    except Exception as e:
        raise RuntimeError(f"Extractor stage failed: {e}") from e
    finally:
        debug["timestamps"]["extractor_end_ms"] = _now_ms()
        debug["extracted_evidence"] = [_to_dict(e) for e in evidence]

    # --- FREE-TIER TPM COOLDOWN ---
    # Groq's sliding 1-min window counts the Extractor's ~9K tokens.
    # The Writer needs ~6K tokens. We must wait for the window to clear
    # enough headroom before firing the next heavy LLM call.
    _status("⏳ Making sure everything runs smoothly...", 82)
    time.sleep(GROQ_INTER_STAGE_SLEEP)

    # 6) WRITER CREW
    _status("✍️ Writing your research report...", 90)
    report = ""
    debug["timestamps"]["synth_start_ms"] = _now_ms()
    try:
        evidence_json = json.dumps(
            [_to_dict(e) for e in evidence], ensure_ascii=False, indent=2
        )
        crew_write = Crew(
            agents=[_get_writer_agent()],
            tasks=[_build_writer_task(topic, evidence_json)],
            process=Process.sequential,
            verbose=False,
        )
        write_raw = crew_write.kickoff()
        write_obj: ReportOutput = _safe_parse_crew_result(write_raw, ReportOutput)
        report = write_obj.report.strip()
        if not report:
            raise RuntimeError("Writer returned an empty report.")
    except Exception as e:
        raise RuntimeError(f"Writer stage failed: {e}") from e
    finally:
        debug["timestamps"]["synth_end_ms"] = _now_ms()

    # Append verified references block
    verified_refs = "\n\n## Verified References\n"
    for item in evidence:
        verified_refs += f"- [{item.id}] {item.title} — {item.url}\n"

    _status("✅ All done!", 100)
    return report + verified_refs, debug
