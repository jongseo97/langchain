from pydantic import BaseModel, Field
from typing import List, Dict, Annotated, TypedDict, Literal, Union, Optional
import pandas as pd

class mixtox_input(BaseModel):
    smiles_list: List[str] = Field(..., description="List of SMILES strings representing each compound in the mixture.")
    ratio_list: List[float] = Field(..., description="List of mixture ratios for the compounds (same order as SMILES).")
    
def dataframe_to_markdown_table(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    max_rows: int = 10,
    float_format: str = ".3f"
) -> str:
    """Convert DataFrame to markdown-formatted table string."""
    if columns:
        df = df[columns]
    df = df.head(max_rows)

    # format floats
    def format_cell(cell):
        if isinstance(cell, float):
            return format(cell, float_format)
        return str(cell)

    header = "| " + " | ".join(df.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(format_cell(cell) for cell in row) + " |" for row in df.values]

    return "\n".join([header, divider] + rows)

def generate_llm_table_response(
    df: pd.DataFrame,
    note: Optional[str] = "※ 이 결과는 모델 예측값이며 실제 독성과 다를 수 있습니다."
) -> str:
    """Generate full markdown-based response for LLM using selected columns."""
    # columns = extra_cols + endpoint_cols
    table = dataframe_to_markdown_table(df, max_rows=10)
    
    synergistic_endpoints = []
    
    for col in df.columns:
        if 'synergistic' in list(df[col]):
            synergistic_endpoints.append(col.split('_')[0])
    
    if len(synergistic_endpoints) == 0:
        synergist_message = f"이 혼합물은 ER, AR, THR, NPC, EB, SG, GnRH endpoint에서 상승작용이 예측되지 않았습니다."
    else:
        synergist_endpoint = ', '.join(synergistic_endpoints)
        synergist_message = f"이 혼합물은 {synergist_endpoint}에서 **상승작용이 예측되었습니다."
    
    return f"다음은 혼합물의 독성 예측 결과입니다 (상위 10개 조합 기준):\n\n{table}\n\n{synergist_message}\n\n{note}"
