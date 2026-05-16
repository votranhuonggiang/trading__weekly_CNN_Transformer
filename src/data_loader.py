from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from src.config import settings


def query_questdb(sql_query: str) -> pd.DataFrame:
    auth = (settings.questdb_username, settings.questdb_password)
    response = requests.get(
        f"{settings.questdb_url}/exec",
        params={"query": sql_query},
        auth=auth,
        timeout=120,
    )
    response.raise_for_status()

    payload = response.json()
    if "dataset" not in payload or "columns" not in payload:
        raise RuntimeError(f"QuestDB query failed: {payload}")

    columns = pd.DataFrame(payload["columns"])["name"].tolist()
    return pd.DataFrame(payload["dataset"], columns=columns)


def fetch_hose_universe() -> pd.DataFrame:
    sql = """
    SELECT
        symbol,
        is_vn30,
        is_vn100,
        is_vnmid,
        is_vnsml,
        is_hnxindex,
        is_upcomindex,
        is_vnindex
    FROM raw_historical_list
    WHERE timestamp >= dateadd('d', -30, now())
      AND type = 'STOCK'
      AND is_vnindex = 1
    LATEST ON timestamp PARTITION BY symbol
    """
    df = query_questdb(sql)
    return df.sort_values("symbol").reset_index(drop=True)


def fetch_raw_ohlcv(start_date: str | None = None) -> pd.DataFrame:
    start_date = start_date or settings.start_date
    sql = f"""
    SELECT
        timestamp,
        symbol,
        open,
        high,
        low,
        close,
        volume
    FROM raw_eod
    WHERE timestamp >= '{start_date}'
      AND length(symbol) = 3
    """
    return query_questdb(sql)


def fetch_vnindex(start_date: str | None = None) -> pd.DataFrame:
    start_date = start_date or settings.start_date
    sql = f"""
    SELECT
        timestamp,
        symbol,
        open,
        high,
        low,
        close,
        volume
    FROM raw_eod
    WHERE timestamp >= '{start_date}'
      AND symbol = 'VNINDEX'
    """
    df = query_questdb(sql)
    if df.empty:
        return df

    df = df.rename(columns={"timestamp": "date", "symbol": "ticker"}).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def clean_ohlcv(df: pd.DataFrame, eligible_symbols: pd.Series) -> pd.DataFrame:
    renamed = df.rename(columns={"timestamp": "date", "symbol": "ticker"}).copy()
    renamed["date"] = pd.to_datetime(renamed["date"], utc=True).dt.tz_localize(None)
    renamed["ticker"] = renamed["ticker"].astype(str).str.upper()

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        renamed[col] = pd.to_numeric(renamed[col], errors="coerce")

    cleaned = renamed[renamed["ticker"].isin(set(eligible_symbols))].copy()
    cleaned = cleaned.drop_duplicates(subset=["date", "ticker"], keep="last")
    cleaned = cleaned.dropna(subset=["date", "ticker", "close", "volume"])
    cleaned = cleaned.sort_values(["ticker", "date"]).reset_index(drop=True)
    return cleaned


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path}")
