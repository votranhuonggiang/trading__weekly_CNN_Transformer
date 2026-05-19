from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    base_dir: Path = Path(__file__).resolve().parent.parent
    questdb_url: str = os.environ.get("QUEST_DB_URL", "http://localhost:9000")
    questdb_username: str | None = os.environ.get("QUESTDB_USERNAME")
    questdb_password: str | None = os.environ.get("QUESTDB_PASSWORD")
    start_date: str = os.environ.get("PROJECT_START_DATE", "2016-01-01")

    @property
    def data_raw_dir(self) -> Path:
        return self.base_dir / "data" / "raw"

    @property
    def data_processed_dir(self) -> Path:
        return self.base_dir / "data" / "processed"

    @property
    def outputs_figures_dir(self) -> Path:
        return self.base_dir / "outputs" / "figures"

    @property
    def outputs_tables_dir(self) -> Path:
        return self.base_dir / "outputs" / "tables"


settings = Settings()


class V2Configuration:
    """Buy-only v2 strategy configuration."""
    TOP_K = 5  # Buy top 5 stocks only
    REBALANCE_FREQ = 'W'  # Weekly
    POSITION_CARRYOVER = False  # Exit all on Friday, buy fresh Monday
    ENTRY_PRICE_TYPE = 'close'  # Use Friday close as entry
    EXIT_PRICE_TYPE = 'close'   # Use Friday close next week as exit
    TRANSACTION_COST_RATES = [0.0, 0.0015, 0.002, 0.0025, 0.0035]
    EQUAL_WEIGHT = True
    
    @staticmethod
    def describe():
        return (
            f"V2 Buy-Only Strategy: "
            f"Top {V2Configuration.TOP_K} stocks, "
            f"weekly rebalance, "
            f"full position reset each week"
        )
