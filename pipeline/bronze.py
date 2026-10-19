"""
Bronze layer: load every raw source as-is into Parquet + a local DuckDB database.

Run:   python bronze.py
Needs: pip install pandas pyarrow pyreadstat openpyxl duckdb

Rules of this layer
- No cleaning, filtering or joining: rows and values are kept as they are.
- Two metadata columns are added: _source_file and _loaded_at.
- Mixed-type text columns are stored as strings (Parquet needs one type per column).

Output
- data/bronze/<table>.parquet
- data/warehouse.duckdb, schema "bronze" (one table per Parquet file + a _manifest table)

Keep the whole data/ folder out of git: it contains personal survey data.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pyreadstat

# ---------------------------------------------------------------------------
# Paths: change these to where your files are. Missing sources are skipped.
# ---------------------------------------------------------------------------
# Paths are relative to the project folder (the parent of pipeline/),
# so the script works no matter which folder you run it from.
PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "data" / "raw"
NICKEL_REPO = PROJECT.parent / "STEM_Curriculum_Diversification"   # Laurence's repo, next to the project

HBO_SAV = RAW / "hbomonitor" / "Sis2025hbo tot desan.sav"
CURRICULA_DIR = NICKEL_REPO / "JSON_files"
CROHO_XLSX = NICKEL_REPO / "data" / "CrohoActueel14-10-2024.xlsx"
ISCO_XLSX = NICKEL_REPO / "data" / "codelijstenisco08.xlsx"
ESCO_DIR = RAW / "esco"              # one subfolder per version, e.g. esco/v1.1.0/*.csv
BENCHMARK_DIR = RAW / "benchmarks"   # csv / json / jsonl / parquet files

BRONZE = PROJECT / "data" / "bronze"
DB_PATH = PROJECT / "data" / "warehouse.duckdb"

LOADED_AT = datetime.now(timezone.utc)
manifest = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def slug(text):
    keep = [c.lower() if c.isalnum() else "_" for c in text]
    return "_".join("".join(keep).split("_")).strip("_")


def save(df, table, source_file):
    """Add metadata, write Parquet, and remember it for the manifest."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("string")
    df["_source_file"] = str(source_file)
    df["_loaded_at"] = LOADED_AT

    path = BRONZE / f"{table}.parquet"
    df.to_parquet(path, index=False)
    manifest.append({"table": table, "rows": len(df), "columns": df.shape[1] - 2,
                     "source_file": str(source_file), "loaded_at": LOADED_AT})
    print(f"  {table:40s} {len(df):>8,} rows  {df.shape[1] - 2:>4} cols")


def skip(name, path):
    print(f"  skipped {name}: {path} not found")


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def load_hbo():
    if not HBO_SAV.exists():
        return skip("HBO Monitor", HBO_SAV)
    df, meta = pyreadstat.read_sav(HBO_SAV)
    save(df, "hbo_monitor_2025", HBO_SAV.name)

    # codebook: variable name -> question label
    codebook = pd.DataFrame({"variable": meta.column_names, "label": meta.column_labels})
    save(codebook, "hbo_codebook", HBO_SAV.name)

    # value labels in long form: variable, code, label
    rows = []
    for var, labels in meta.variable_value_labels.items():
        for code, label in labels.items():
            rows.append({"variable": var, "code": str(code), "label": label})
    save(pd.DataFrame(rows), "hbo_value_labels", HBO_SAV.name)


def load_curricula():
    if not CURRICULA_DIR.exists():
        return skip("curricula", CURRICULA_DIR)
    rows = []
    for f in sorted(CURRICULA_DIR.glob("*.txt")):
        for programme in json.loads(f.read_text(encoding="utf-8")):
            programme["_file"] = f.name
            rows.append(programme)
    save(pd.DataFrame(rows), "curricula", CURRICULA_DIR)


def load_croho():
    if not CROHO_XLSX.exists():
        return skip("CROHO", CROHO_XLSX)
    save(pd.read_excel(CROHO_XLSX), "croho", CROHO_XLSX.name)


def load_isco():
    if not ISCO_XLSX.exists():
        return skip("ISCO", ISCO_XLSX)
    sheets = pd.read_excel(ISCO_XLSX, sheet_name=None)
    for sheet, df in sheets.items():
        name = slug(sheet).replace("isco_08_", "")
        save(df, "isco_" + name, f"{ISCO_XLSX.name} [{sheet}]")


def load_files(folder, prefix):
    """Load every csv / json / jsonl / parquet file in one folder as its own table."""
    for f in sorted(folder.iterdir()):
        suffix = f.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(f, dtype=str)
        elif suffix == ".jsonl":
            df = pd.read_json(f, lines=True)
        elif suffix == ".json":
            df = pd.read_json(f)
        elif suffix == ".parquet":
            df = pd.read_parquet(f)
        else:
            continue
        save(df, f"{prefix}_{slug(f.stem)}", f"{folder.name}/{f.name}")


def load_folder(folder, prefix):
    """Files directly in the folder -> <prefix>_<file>.
    Each subfolder (e.g. a version like v1.1.0) -> <prefix>_<subfolder>_<file>."""
    if not folder.exists():
        return skip(prefix, folder)
    load_files(folder, prefix)
    for sub in sorted(folder.iterdir()):
        if sub.is_dir():
            load_files(sub, f"{prefix}_{slug(sub.name)}")


# ---------------------------------------------------------------------------
# DuckDB
# ---------------------------------------------------------------------------
def build_duckdb():
    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE SCHEMA IF NOT EXISTS bronze")
    for item in manifest:
        table = item["table"]
        path = BRONZE / f"{table}.parquet"
        con.execute(f"CREATE OR REPLACE TABLE bronze.{table} AS SELECT * FROM read_parquet('{path}')")

    man = pd.DataFrame(manifest)
    con.register("man", man)
    con.execute("CREATE OR REPLACE TABLE bronze._manifest AS SELECT * FROM man")
    con.close()


def main():
    BRONZE.mkdir(parents=True, exist_ok=True)
    print("Loading sources into", BRONZE)
    load_hbo()
    load_curricula()
    load_croho()
    load_isco()
    load_folder(ESCO_DIR, "esco")
    load_folder(BENCHMARK_DIR, "bench")

    if not manifest:
        print("\nNothing loaded: check the paths at the top of this script.")
        print("Project folder:", PROJECT)
        return
    build_duckdb()
    print(f"\n{len(manifest)} tables written to {DB_PATH} (schema: bronze)")


if __name__ == "__main__":
    main()