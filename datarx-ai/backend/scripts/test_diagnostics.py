import pandas as pd
import json
import asyncio
from pprint import pprint

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.services.dataset_models import detect_multivariate_outliers
from app.services.embedding_retrieval import get_similar_examples
from app.services.stats_engine import full_diagnosis
from app.services.llm_pipeline import run_analysis_pipeline
import sys

async def main():
    print("=" * 60)
    print("STEP 2: Isolation Forest on the actual demo dataset")
    print("=" * 60)
    df = pd.read_csv("sample_data/sample_messy_data.csv")
    
    outlier_indices = detect_multivariate_outliers(df)
    print(f"Total rows: {len(df)}")
    print(f"Detected multivariate outliers: {len(outlier_indices)}")
    
    if outlier_indices:
        numeric_df = df.select_dtypes(include=['number'])
        means = numeric_df.mean()
        medians = numeric_df.median()
        
        for idx in outlier_indices:
            print(f"\n--- Outlier Row {idx} ---")
            row = df.iloc[idx]
            print(row.to_dict())
            
            print("Comparison to Median/Mean for numeric columns:")
            for col in numeric_df.columns:
                val = row[col]
                if pd.notna(val):
                    print(f"  {col}: {val} (Median: {medians[col]:.2f}, Mean: {means[col]:.2f})")
    else:
        print("No multivariate outliers detected.")


    print("\n" + "=" * 60)
    print("STEP 3: Feature Importance trigger check")
    print("=" * 60)
    # Print the code path explanation as requested
    print("Currently, the frontend/API flow DOES have a TargetSelectionForm.tsx (created in a previous task).")
    print("However, TargetSelectionForm is rendered in Step 1.5 of page.tsx IF the user clicks 'Select Target Column'.")
    print("The target column is sent to the backend via POST /api/analyze with 'target_column' and 'target_purpose'.")
    print("In backend/routers/analyze.py, 'target_column' is extracted from the AnalyzeRequest and passed to `run_analysis_pipeline`.")
    print("Inside `target_aware_checks.py`, `compute_feature_importance()` is called dynamically only if `target_column` is present.")

    
    print("\n" + "=" * 60)
    print("STEP 4: Few-shot example bank size and retrieval behavior")
    print("=" * 60)
    with open("app/models/few_shot_examples.json") as f:
        examples = json.load(f)
    print(f"Total entries in few_shot_examples.json: {len(examples)}")
    
    test_columns = ["price", "user_id", "email_address"]
    for col in test_columns:
        print(f"\n--- Retrieving examples for '{col}' ---")
        similars = get_similar_examples(col)
        for s in similars:
            print(f"  -> '{s['column_name']}' (Sim: {s['similarity']:.3f}) - Type: {s['semantic_type']}")

    
    print("\n" + "=" * 60)
    print("STEP 5: Full pipeline dry run")
    print("=" * 60)
    diagnosis = full_diagnosis(df)
    
    # Run the LLM pipeline end-to-end
    result = await run_analysis_pipeline(df, diagnosis, target_column="price", target_purpose="predicting item cost")
    
    print("\n--- Final Semantic Classification Results ---")
    for st in result["semantic_types"]:
        print(st)
        
    print("\n--- Final Target Analysis ---")
    pprint(result.get("target_analysis", {}))
    
    print("\n--- Final Recommendations ---")
    for rec in result["recommendations"]:
        print(f"Col: {rec['column']} | Action: {rec['recommended_action']} | Conf: {rec['confidence']} | Review: {rec['needs_review']}")
        print(f"  Reason: {rec['justification']}")

if __name__ == "__main__":
    asyncio.run(main())
