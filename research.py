import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import StrMethodFormatter
from dotenv import load_dotenv
import seaborn as sns
from datetime import datetime
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# --- DATA SOURCES ---

def query_questdb(sql_query):
    """Execute a raw SQL query against QuestDB"""
    host = os.environ.get("QUEST_DB_URL", "http://localhost:9000")
    auth = (os.getenv("QUESTDB_USERNAME"), os.getenv("QUESTDB_PASSWORD"))

    try:
        response = requests.get(
            host + "/exec", params={"query": sql_query}, auth=auth
        ).json()

        if "dataset" not in response or "columns" not in response:
            print(f"Error or no data: {response}")
            return pd.DataFrame()

        df = pd.DataFrame(
            response["dataset"],
            columns=pd.DataFrame(response["columns"])["name"].values,
        )
        return df

    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

def raw_eod():
    sql = """
    SELECT timestamp, symbol, close, volume 
    FROM raw_eod
    WHERE timestamp >='2016-01-01' and length(symbol) = 3
    """
    return query_questdb(sql)

def hose_list():
    sql = """
SELECT symbol, is_vn30, is_vn100, is_vnmid, is_vnsml, is_hnxindex, is_upcomindex, is_vnindex
    FROM raw_historical_list
    WHERE timestamp >= dateadd('d', -30, now())
      AND type = 'STOCK' and is_vnindex = 1
    LATEST ON timestamp PARTITION BY symbol
    """
    return query_questdb(sql)