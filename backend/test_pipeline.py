import json
import asyncio
from app.services.llm_pipeline import run_pipeline, _check_flagged

with open('sample_data/sample_messy_data.csv', 'r') as f:
    pass # we need to actually get the diagnosis first

from app.services.stats_engine import diagnose_dataframe
import pandas as pd
df = pd.read_csv('sample_data/sample_messy_data.csv')
diagnosis = diagnose_dataframe(df)

print(f"Check flagged: {_check_flagged(diagnosis)}")

async def run():
    res = await run_pipeline(diagnosis)
    print("Pipeline result:")
    print(res)

asyncio.run(run())
