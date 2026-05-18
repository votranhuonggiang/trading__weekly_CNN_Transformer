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
    end_date = _get_raw_eod_end_date()
    chunks: list[pd.DataFrame] = []

    for chunk_start, chunk_end in _yearly_ranges(start_date, end_date):
        chunk = _fetch_raw_ohlcv_chunk(chunk_start, chunk_end)
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])

    return pd.concat(chunks, ignore_index=True)


def _get_raw_eod_end_date() -> str:
    df = query_questdb("SELECT max(timestamp) AS max_ts FROM raw_eod WHERE length(symbol) = 3")
    if df.empty or pd.isna(df.loc[0, "max_ts"]):
        raise RuntimeError("raw_eod has no rows for 3-character stock symbols.")
    max_ts = pd.to_datetime(df.loc[0, "max_ts"], utc=True).tz_localize(None)
    return (max_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _yearly_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    ranges: list[tuple[str, str]] = []
    cursor = start
    while cursor < end:
        next_cursor = min(pd.Timestamp(year=cursor.year + 1, month=1, day=1), end)
        ranges.append((cursor.strftime("%Y-%m-%d"), next_cursor.strftime("%Y-%m-%d")))
        cursor = next_cursor
    return ranges


def _fetch_raw_ohlcv_chunk(start_date: str, end_date: str) -> pd.DataFrame:
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
      AND timestamp < '{end_date}'
      AND length(symbol) = 3
    ORDER BY timestamp, symbol
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
