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
    triple_barrier_take_profit_pct: float = float(os.environ.get("TB_TAKE_PROFIT_PCT", "0.05"))
    triple_barrier_stop_loss_pct: float = float(os.environ.get("TB_STOP_LOSS_PCT", "0.05"))
    triple_barrier_ambiguous_policy: str = os.environ.get("TB_AMBIGUOUS_POLICY", "neutral")

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
