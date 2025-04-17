import requests
from fake_useragent import UserAgent

from langchain.tools import tool
from typing import List, Dict, Annotated, TypedDict, Literal, Union, Optional
from langchain_teddynote.tools import GoogleNews
from langchain_experimental.utilities import PythonREPL

#my functions
from function_utils import mixtox_input, dataframe_to_markdown_table, generate_llm_table_response
from tools.F_predict.F_main import F_main
from tools.skin_predict.skin_main import skin_main
from tools.mixtox_predict.mix_main import mix_main
from rdkit import Chem

# 도구 생성
@tool
def search_news(query: str) -> List[Dict[str, str]]:
    """Search Google News by input keyword"""
    news_tool = GoogleNews()
    return news_tool.search_by_keyword(query, k=5)


# 도구 생성
@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    result = ""
    try:
        result = PythonREPL().run(code)
    except BaseException as e:
        print(f"Failed to execute. Error: {repr(e)}")
    finally:
        return result

@tool
def QUERY_to_CAS(query: str) -> str:
    """
    Input molecule(English name or SMILES), returns CAS number.

    Args:
        query (str): molecule(English name or SMILES)

    Returns:
        str: CAS number
    """
    mode = 'name'
    try:
        mol = Chem.MolFromSmiles(query, sanitize=False)
        if mol:
            mode = 'smiles'
    except:
        pass
    if mode == 'name':
        query = query.replace('_','')
    
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{mode}/{query}/cids/txt'
    ua = UserAgent()
    requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
    header = {"User-Agent":ua.random}
    r = requests.get(url, verify=False, headers = header, timeout=10)
    if r.status_code != 200:
        raise "CID number not found"
    r.encoding = 'utf-8'
    cid = r.text.strip().split('\n')[0]
    
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON'
    r = requests.get(url, verify=False, headers = header, timeout=10)
    if r.status_code != 200:
        raise "CAS number not found"
    r.encoding = 'utf-8'
    data = r.json()
    while 'Information' not in data:
        for d in data:
            if type(data) == dict:
                d = data[d]
            if 'CAS' in str(d):
                break
        data = d
        if type(data) == str:
            raise 'CAS number not found'
    
    cas = data['Information'][0]['Value']['StringWithMarkup'][0]['String']
    return cas   

@tool
def QUERY_to_SMILES(query: str) -> str:
    """
    Input molecule(English name or CAS number), returns SMILES.

    Args:
        query (str): molecule(English name or CAS number)

    Returns:
        str: SMILES
    """
    query = query.replace('_', '')
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}/property/isomericSMILES/txt'
    ua = UserAgent()
    requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
    header = {"User-Agent":ua.random}
    r = requests.get(url, verify=False, headers = header, timeout=100)
    if r.status_code != 200:
        raise "SMILES code not found"
    r.encoding = 'utf-8'
    smiles = r.text.strip().split('\n')[0]
    return smiles       



@tool
def SMILES_to_USE(SMILES:str) -> str:
    """
    Input SMILES, returns predicted functional use in the chemical product of molecule.
    Use functional use as English, not Korean.
    
    Args:
        SMILES (str): SMILES

    Returns:
        str: 'predicted' functional uses in chemical product separated by a semicolon(';')
    """
    USE = F_main([SMILES])
    return USE


@tool
def SMILES_to_skin_sensitization(SMILES:str) -> Dict[str, int]:
    """
    Input SMILES, returns predicted skin-sensitization of molecule.
    The results are provided for four key events (KEs) that are critical in assessing skin-sensitization activity.
    KE1: DPRA
    KE2: Keratinosens
    KE3: h-CLAT
    KE4: LLNA

    Args:
        SMILES (str): SMILES

    Returns:
        str: 'predicted' skin-sensitization of molecule. 
    """
    skin_sens = skin_main([SMILES])
    return skin_sens


@tool("predict_mixture_toxicity", args_schema=mixtox_input, return_direct=True)
def mixture_to_mixtox(
    smiles_list: List[str],
    ratio_list: List[float]
) -> str:
    """
    Input List of SMILES, returns predicted mixture toxicity outcomes for all pairwise mixtures from given SMILES and ratios.

    Args:
        smiles_list (List[str]): list of SMILES
        ratio_list (List[float]): each ratio in mixture

    Returns:
        str: 'predicted' mixture toxicity of binary mixtures as 'Markdown Table'.
    """
    
    result_df = mix_main(smiles_list, ratio_list)
    # endpoint_cols = [col for col in result_df.columns if col.endswith("_toxicity_prediction")]
    return generate_llm_table_response(result_df)#, endpoint_cols)

def get_openai_tools() -> List[dict]:
    functions = [
        QUERY_to_CAS,
        QUERY_to_SMILES,
        SMILES_to_USE,
        search_news, 
        python_repl_tool,
        SMILES_to_skin_sensitization,
        mixture_to_mixtox
    ]

    return functions

if __name__ == '__main__':
    smiles_list = ['CN(C)C(=S)SSC(=S)N(C)C', 'C1=CC=C(C=C1)C2=CC(=O)C3=CC=CC=C3O2', 'C1C(CC2=CC=CC=C2C1C3=C(C4=CC=CC=C4OC3=O)O)C5=CC=C(C=C5)C6=CC=CC=C6', 'CC1=CC2=C(C(=C(C(=C2C(C)C)O)O)C=O)C(=C1C3=C(C4=C(C=C3C)C(=C(C(=C4C=O)O)O)C(C)C)O)O']
    ratio_list = [0.3,0.1,0.15,0.45]
    mix_main(smiles_list, ratio_list)