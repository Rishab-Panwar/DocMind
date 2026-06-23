"""Structured-data Q&A for tabular files (.csv / .xlsx / .xls / .db / .sqlite).

RAG retrieves text chunks and is great for lookups, descriptions and summaries,
but it cannot reliably compute over many rows (totals, counts, max/min) because
it only sees the chunks that were retrieved. This module loads tabular files into
pandas DataFrames and lets the LLM run a real pandas expression for computational
questions — exact answers that scale. Non-computational questions return None so
the normal RAG path handles them unchanged.
"""
import os
import re
import json
import builtins
from typing import Dict, Optional

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from logger import GLOBAL_LOGGER as log

TABULAR_EXTS = {".csv", ".xlsx", ".xls", ".db", ".sqlite", ".sqlite3"}

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
                sheets = pd.read_excel(src, sheet_name=None, header=None)
                for sheet, raw in sheets.items():
                    name = base if len(sheets) == 1 else f"{base}::{sheet}"
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


def _format_result(result) -> str:
    if isinstance(result, pd.DataFrame):
        return result.head(30).to_string()
    if isinstance(result, pd.Series):
        return result.head(30).to_string()
    return str(result)


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
        "Decide whether answering REQUIRES computing over table rows — totals, "
        "sums, counts, averages, max/min, sorting, or filtering-and-counting — or "
        "an exact single-row lookup. Reply with ONE JSON object and nothing else:\n"
        '- If yes: {{"use": true, "table": "<exact table name from the list>", '
        '"expr": "<a single pandas expression using variable df and pd that '
        'evaluates to the answer>"}}\n'
        '- If it is a summary, description, "what is this about", or about '
        'non-tabular content: {{"use": false}}\n\n'
        "Rules for expr: use only `df` (the chosen table) and `pd`; match column "
        "names exactly. Use the sample rows to pick the RIGHT column — values may "
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
        "double-count opening-balance and total rows). Return a scalar or small "
        "DataFrame/Series."
    )
    try:
        raw = (decide | llm | StrOutputParser()).invoke({"schema": schema, "q": question})
    except Exception as e:
        log.warning("table_qa: decide step failed", error=str(e))
        return None
    data = _parse_json(raw)
    if not data or not data.get("use"):
        return None
    name = _closest(data.get("table"), tables)
    expr = (data.get("expr") or "").strip()
    if not name or not expr:
        return None
    df = tables[name]
    try:
        # Sandboxed eval: only safe builtins, pd, and the chosen df in scope.
        result = eval(expr, {"__builtins__": _SAFE_BUILTINS, "pd": pd}, {"df": df})
    except Exception as e:
        log.warning("table_qa: expr failed", table=name, expr=expr, error=str(e))
        return None
    result_str = _format_result(result)
    log.info("table_qa: computed", table=name, expr=expr, result=result_str[:120])
    phrase = ChatPromptTemplate.from_template(
        'The user asked: "{q}"\n'
        "This value was computed from the file {file}: {result}\n\n"
        "Write a concise, direct answer to the question using this value. Mention "
        "the file name {file}. Do not show code or mention pandas."
    )
    try:
        ans = (phrase | llm | StrOutputParser()).invoke(
            {"q": question, "file": name.split("::")[0], "result": result_str}
        )
        return ans.strip()
    except Exception:
        return f"According to {name.split('::')[0]}: {result_str}"
