"""Structured-data Q&A for tabular files (.csv / .xlsx / .xls / .db / .sqlite).

RAG retrieves text chunks and is great for lookups, descriptions and summaries,
but it cannot reliably compute over many rows (totals, counts, max/min) because
it only sees the chunks that were retrieved. This module loads tabular files into
pandas DataFrames and lets the LLM run a real pandas expression for computational
questions — exact answers that scale. Non-computational questions return None so
the normal RAG path handles them unchanged.
"""
import os
import io
import re
import ast
import json
import builtins
from typing import Dict, Optional

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from logger import GLOBAL_LOGGER as log

TABULAR_EXTS = {".csv", ".xlsx", ".xls", ".db", ".sqlite", ".sqlite3"}

# Strong signals that a question needs the table data (aggregation or a labeled
# row/value lookup). When one of these matches and tabular files exist, we force
# the compute branch so the LLM cannot intermittently answer use=false and let
# the question fall back to (unreliable) RAG — the cause of "sometimes it
# answers, sometimes it says I don't know" for the same question.
_COMPUTE_HINTS = re.compile(
    r"(?i)\b("
    r"total|totals|subtotal|grand\s+total|sum|sums|count|counts|how\s+many|"
    r"number\s+of|average|avg|mean|median|max(?:imum)?|min(?:imum)?|"
    r"highest|lowest|largest|smallest|most|least|"
    r"opening\s+balance|closing\s+balance|balance|net\s+amount|aggregate"
    r")\b"
)


def looks_computational(question: str) -> bool:
    """True if the question shows an aggregation/lookup signal that warrants the
    (exact) table-compute path. Callers use this to skip the LLM 'decide' round
    trip entirely for plain summaries/descriptions — a major latency saving."""
    return bool(question) and bool(_COMPUTE_HINTS.search(question))

# Safe builtins for the eval sandbox — the common functions LLMs use in pandas
# expressions (len, sum, max…), but NOT open/import/exec/eval/etc.
_SAFE_BUILTINS = {
    n: getattr(builtins, n)
    for n in (
        "len", "sum", "min", "max", "sorted", "round", "abs", "int", "float",
        "str", "list", "dict", "set", "tuple", "range", "enumerate", "zip",
        "map", "filter", "any", "all", "bool",
    )
}


# ---------------- loading ----------------

_HEADER_TOKENS = {
    "date", "debit", "credit", "particular", "particulars", "amount", "vch",
    "voucher", "balance", "name", "qty", "quantity", "price", "rate", "total",
    "description", "sr", "sno", "invoice", "gst", "hsn", "account", "ledger",
    "type", "ref", "narration", "value", "id", "email", "question", "answer",
}


def _detect_header_row(raw: pd.DataFrame) -> int:
    """Real-world spreadsheets often have title/address rows before the table,
    and ledger data rows can have MORE filled cells than the header — so detect
    the header by how many cells look like column names, with a non-null-count
    fallback."""
    best_i, best_score = None, 0
    for i in range(min(25, len(raw))):
        score = 0
        for c in raw.iloc[i]:
            if pd.isna(c):
                continue
            words = str(c).strip().lower().split()
            if any(tok == w for tok in _HEADER_TOKENS for w in words):
                score += 1
        if score > best_score:
            best_score, best_i = score, i
    if best_i is not None and best_score >= 2:
        return best_i
    # fallback: row with most non-null cells
    best_i, best_n = 0, -1
    for i in range(min(25, len(raw))):
        n = int(raw.iloc[i].notna().sum())
        if n > best_n:
            best_n, best_i = n, i
    return best_i


def _frame_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    h = _detect_header_row(raw)
    seen: Dict[str, int] = {}
    cols = []
    for j, c in enumerate(raw.iloc[h]):
        name = str(c).strip() if pd.notna(c) and str(c).strip() else f"col{j}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        cols.append(name)
    body = raw.iloc[h + 1:].copy()
    body.columns = cols
    for c in body.columns:
        converted = pd.to_numeric(body[c], errors="coerce")
        # keep numeric conversion only if it doesn't wipe out most values
        if converted.notna().sum() >= body[c].notna().sum() * 0.6:
            body[c] = converted
    return body.reset_index(drop=True)


def _read_excel_raw_sheets(src: str, base: str):
    """Yield (table_name, raw_DataFrame) for an Excel file, choosing the reader
    by the file's real signature rather than its extension. Spreadsheet exports
    (esp. Tally/ERP) are routinely mislabeled — an OOXML .xlsx saved as .xls, or
    HTML saved as .xls/.xlsx — which the default pd.read_excel engine rejects.

    Raw frames are returned with header=None so the existing header-detection in
    _frame_from_raw applies uniformly — so a correctly-labeled file parses
    exactly as before (same engine, same result); only mislabeled files change."""
    with open(src, "rb") as f:
        data = f.read()
    head = data[:8]
    if head[:4] == b"PK\x03\x04":            # OOXML container
        engine = "openpyxl"
    elif head[:4] == b"\xd0\xcf\x11\xe0":    # legacy BIFF (OLE2)
        engine = "xlrd"
    else:
        engine = None
    if engine is not None:
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, engine=engine)
        items = list(sheets.items())
        for sheet, raw in items:
            yield (base if len(items) == 1 else f"{base}::{sheet}"), raw
        return
    # HTML masquerading as Excel — let pandas parse the table(s).
    try:
        raws = pd.read_html(io.BytesIO(data), header=None)
    except Exception:
        raws = pd.read_html(io.BytesIO(data))
    for i, raw in enumerate(raws, start=1):
        yield (base if len(raws) == 1 else f"{base}::Table{i}"), raw


def load_tables(sources) -> Dict[str, pd.DataFrame]:
    """Load every tabular source into named DataFrames. Sheets/tables become
    separate entries keyed `filename` or `filename::sheet`."""
    tables: Dict[str, pd.DataFrame] = {}
    for src in sources:
        ext = os.path.splitext(src)[1].lower()
        base = os.path.basename(src)
        try:
            if ext == ".csv":
                tables[base] = pd.read_csv(src)
            elif ext in {".xlsx", ".xls"}:
                for name, raw in _read_excel_raw_sheets(src, base):
                    tables[name] = _frame_from_raw(raw)
            elif ext in {".db", ".sqlite", ".sqlite3"}:
                import sqlite3
                conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
                try:
                    names = pd.read_sql(
                        "SELECT name FROM sqlite_master WHERE type='table' "
                        "AND name NOT LIKE 'sqlite_%'", conn
                    )["name"].tolist()
                    for t in names:
                        try:
                            tables[f"{base}::{t}"] = pd.read_sql(f'SELECT * FROM "{t}"', conn)
                        except Exception:
                            pass
                finally:
                    conn.close()
        except Exception as e:
            log.warning("table_qa: failed to load table", source=src, error=str(e))
    return tables


# ---------------- query ----------------

def _schema_text(tables: Dict[str, pd.DataFrame]) -> str:
    parts = []
    for name, df in tables.items():
        cols = ", ".join(f"{c} ({df[c].dtype})" for c in list(df.columns)[:30])
        # Sample rows are crucial: messy spreadsheets (merged cells) scatter
        # values into generically-named columns (col2), so the LLM must SEE the
        # data to pick the right column.
        try:
            head = df.head(4).to_string(max_colwidth=28)
            tail = df.tail(3).to_string(max_colwidth=28)
            sample = f"{head}\n...last rows...\n{tail}"
        except Exception:
            sample = "(sample unavailable)"
        parts.append(f'Table "{name}" — {len(df)} rows.\nColumns: {cols}\nSample (first & last rows):\n{sample}')
    return "\n\n".join(parts)


def _parse_json(raw: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _safe_eval(expr: str, df: "pd.DataFrame"):
    """Evaluate a pandas snippet in a restricted sandbox and return its value.

    The decide step is asked for a single expression, but LLMs intermittently
    emit multi-statement code ("df2 = df[...]; df2['x'].sum()") or trailing
    assignments. Plain eval() rejects those with SyntaxError, which used to
    silently drop the computed answer and fall back to (often wrong) RAG — the
    main cause of "sometimes it answers, sometimes it says I don't know". Run
    single expressions via eval; for multi-statement code, exec it and return
    the value of the final expression (or a `result`/`_result`/`ans` variable).
    Scope stays restricted to safe builtins, pd and df."""
    safe_globals = {"__builtins__": _SAFE_BUILTINS, "pd": pd}
    local = {"df": df}
    try:
        return eval(expr, safe_globals, local)  # fast path: one expression
    except SyntaxError:
        pass
    tree = ast.parse(expr, mode="exec")
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        # Capture the trailing expression's value.
        last = tree.body[-1]
        assign = ast.Assign(targets=[ast.Name(id="_result", ctx=ast.Store())], value=last.value)
        ast.copy_location(assign, last)
        tree.body[-1] = assign
        ast.fix_missing_locations(tree)
        exec(compile(tree, "<expr>", "exec"), safe_globals, local)
        return local.get("_result")
    # No trailing expression — accept a conventional result variable name.
    exec(compile(tree, "<expr>", "exec"), safe_globals, local)
    for name in ("_result", "result", "ans", "answer"):
        if name in local:
            return local[name]
    return None


def _format_result(result) -> str:
    if isinstance(result, pd.DataFrame):
        return result.head(30).to_string()
    if isinstance(result, pd.Series):
        return result.head(30).to_string()
    return str(result)


def _fmt_num(v) -> str:
    """Render a value with thousands separators when it's a finite number.
    Non-numeric, NaN, and infinite values fall back to their string form so
    formatting can never raise (e.g. int(NaN) would crash)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    import math
    if math.isnan(f) or math.isinf(f):
        return str(v)
    return f"{int(f):,}" if f == int(f) else f"{f:,.2f}"


def _is_blank(v) -> bool:
    """True for NaN/None — empty ledger cells we should omit from the phrase."""
    try:
        return v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v)
    except (TypeError, ValueError):
        return False


def _pairs(items) -> str:
    """Render label:value pairs, skipping blank (NaN/None) values so empty
    ledger cells don't show as 'nan'."""
    parts = [f"{k}: {_fmt_num(v)}" for k, v in items if not _is_blank(v)]
    if parts:
        return ", ".join(parts)
    # Everything blank — fall back to showing the labels with their raw values.
    return ", ".join(f"{k}: {v}" for k, v in items)


def _humanize_result(result) -> str:
    """Turn a computed scalar/Series/DataFrame/dict/tuple into a clean inline
    phrase, so the answer reads naturally without a second LLM call."""
    if isinstance(result, dict):
        return _pairs(result.items())
    if isinstance(result, pd.Series):
        return _pairs(result.items())
    if isinstance(result, pd.DataFrame):
        if len(result) == 1:
            row = result.iloc[0]
            return _pairs((c, row[c]) for c in result.columns)
        return result.head(30).to_string(index=False)
    if isinstance(result, (list, tuple)):
        return ", ".join(_fmt_num(v) for v in result)
    return _fmt_num(result)


def _closest(name: Optional[str], tables: Dict[str, pd.DataFrame]) -> Optional[str]:
    if not name:
        return None
    if name in tables:
        return name
    low = name.lower()
    for k in tables:
        if low in k.lower() or k.lower() in low:
            return k
    return None


def answer_with_tables(llm, question: str, tables: Dict[str, pd.DataFrame]) -> Optional[str]:
    """Return an exact computed answer for computational questions, or None to
    let the RAG path handle descriptive/semantic questions."""
    if not tables:
        return None
    schema = _schema_text(tables)
    decide = ChatPromptTemplate.from_template(
        "You can compute over these in-memory pandas DataFrames:\n{schema}\n\n"
        'User question: "{q}"\n\n'
        "Decide whether answering needs the TABLE DATA. Answer use=true if the "
        "question involves ANY of: totals, sums, counts, averages, max/min, "
        "sorting, filtering-and-counting; OR reading a specific labeled row such "
        "as a TOTAL / GRAND TOTAL / opening balance / closing balance; OR an "
        "exact single-row or single-cell lookup of a value held in the table. "
        "Only answer use=false for genuinely non-tabular questions — summaries, "
        "descriptions, 'what is this about', or content that is not in the table. "
        "When in doubt for a question about numbers in the ledger, choose true.\n"
        "{force_note}"
        "Reply with ONE JSON object and nothing else:\n"
        '- If yes: {{"use": true, "table": "<exact table name from the list>", '
        '"expr": "<a single pandas expression using variable df and pd that '
        'evaluates to the answer>"}}\n'
        '- If it is a summary, description, "what is this about", or about '
        'non-tabular content: {{"use": false}}\n\n'
        "Rules for expr: it MUST be a SINGLE Python expression — no assignments, "
        "no semicolons, no statements. For an answer with more than one value, "
        "return a LABELED pd.Series so each value is self-describing, e.g. "
        "pd.Series({{'count': ..., 'total': ...}}) — do NOT return a bare tuple. "
        "Use only `df` (the chosen table) and `pd`; "
        "match column names exactly. Use the sample rows to pick the RIGHT column — values may "
        "sit in generically-named columns (e.g. col2). For text matching use "
        "df['col'].astype(str).str.contains('x', case=False, na=False). If unsure "
        "which column holds a value, search every column: "
        "df.astype(str).apply(lambda r: r.str.contains('x', case=False, na=False)"
        ".any(), axis=1).sum(). For date columns, coerce with pd.to_datetime("
        "df['Date'], errors='coerce') and IGNORE implausible dates (keep only "
        "year >= 2000) — messy spreadsheets leak numbers into date cells that "
        "wrongly parse as 1970. Watch for embedded summary rows: if a printed "
        "TOTAL / GRAND TOTAL / CLOSING BALANCE row already exists (visible in the "
        "last rows), RETURN ITS VALUE instead of summing the column (summing would "
        "double-count opening-balance and total rows). IMPORTANT: the LABEL of "
        "such a row (e.g. 'Closing Balance', 'TOTAL') may sit in ANY column — "
        "often a generic one like col2 — NOT necessarily 'Particulars'. Find the "
        "row by searching across ALL columns, then take its non-null numeric "
        "value(s), e.g.: "
        "df[df.astype(str).apply(lambda r: r.str.contains('closing balance', "
        "case=False, na=False).any(), axis=1)].select_dtypes('number').stack()"
        ".dropna(). NEVER use .iloc[0] on a filter that might be empty. "
        "Return a scalar or small DataFrame/Series."
    )
    # Deterministic gate: if the question clearly needs the table, force the
    # compute branch so the model can't intermittently bail to use=false.
    forced = bool(_COMPUTE_HINTS.search(question or ""))
    force_note = (
        "IMPORTANT: this question needs the table data — you MUST set use=true and "
        "provide the best table and a valid expr. Do NOT answer use=false.\n"
        if forced else ""
    )
    try:
        raw = (decide | llm | StrOutputParser()).invoke(
            {"schema": schema, "q": question, "force_note": force_note}
        )
    except Exception as e:
        log.warning("table_qa: decide step failed", error=str(e))
        return None
    data = _parse_json(raw)
    if not data:
        return None
    # When not forced, respect the model's use=false (let RAG handle it). When
    # forced, proceed as long as we got a usable table + expr below.
    if not forced and not data.get("use"):
        return None
    name = _closest(data.get("table"), tables)
    expr = (data.get("expr") or "").strip()
    if not name or not expr:
        return None
    df = tables[name]
    try:
        # Sandboxed evaluation: only safe builtins, pd, and the chosen df in
        # scope. Tolerates both single expressions and multi-statement snippets.
        result = _safe_eval(expr, df)
    except Exception as e:
        log.warning("table_qa: expr failed", table=name, expr=expr, error=str(e))
        return None
    result_str = _format_result(result)
    log.info("table_qa: computed", table=name, expr=expr, result=result_str[:120])
    # Format the exact computed value deterministically instead of paying a
    # second LLM round-trip to phrase it — keeps the value exact and saves
    # ~one slow call per computational query. Never let a formatting edge case
    # (NaN, odd types) crash the request — fall back to the raw value string.
    file = name.split("::")[0]
    try:
        return f"According to {file}, {_humanize_result(result)}."
    except Exception as e:
        log.warning("table_qa: result formatting failed", error=str(e))
        return f"According to {file}: {result_str}"
