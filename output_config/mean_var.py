import pandas as pd
import numpy as np
from pathlib import Path

def average_excel_solutions(filepaths: list[str], output_path: str = None) -> dict[str, pd.DataFrame]:
    sheets = ["X_sol", "W_sol", "Y_sol", "I_sol", "D_sol"]
    
    all_data = {sheet: [] for sheet in sheets}
    
    for filepath in filepaths:
        for sheet in sheets:
            df = pd.read_excel(filepath, sheet_name=sheet, index_col=[0, 1])
            df_mean = df.groupby(level=1).mean()
            all_data[sheet].append(df_mean)
    
    averaged = {}
    for sheet in sheets:
        stacked = pd.concat(all_data[sheet])         
        averaged[sheet] = stacked.groupby(level=0).mean() 
    
    if output_path:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for sheet, df in averaged.items():
                df.to_excel(writer, sheet_name=sheet)
        print(f">> Promedio exportado a: {output_path}")
    
    return averaged