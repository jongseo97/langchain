import requests
from fake_useragent import UserAgent

from langchain.tools import tool
from typing import List, Dict, Annotated
from langchain_teddynote.tools import GoogleNews
from langchain_experimental.utilities import PythonREPL

#my functions
from tools.F_predict.F_main import F_main
from tools.skin_predict.skin_main import skin_main
from rdkit import Chem

# 도구 생성
@tool
def search_news(query: str) -> List[Dict[str, str]]:
    """입력된 키워드 기반 구글 뉴스 검색"""
    news_tool = GoogleNews()
    return news_tool.search_by_keyword(query, k=5)


# 도구 생성
@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """python 코드를 실행할 때 사용하세요. output의 값을 보고 싶을 때는,
    `print(...) 명령어를 통해 출력하면 사용자가 볼 수 있습니다."""
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
    분자(영문이름 or SMILES)를 입력받아 CAS 번호를 return합니다.

    Args:
        query (str): 분자(영문명 or SMILES)

    Returns:
        str: CAS 번호
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
    분자(영문이름 or CAS 번호)를 입력받아 SMILES를 return합니다.

    Args:
        query (str): 분자(영문명 or CAS 번호)

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
    SMILES를 입력받아 해당 분자가 제품에서 사용되는 기능 및 용도를 예측합니다.
    
    Args:
        SMILES (str): SMILES

    Returns:
        str: '예측된' 화학 제품에서 사용되는 기능 및 용도 (여러 개인 경우 ';' 구분)
    """
    USE = F_main([SMILES])
    return USE


@tool
def SMILES_to_skin_sensitization(SMILES:str) -> Dict[str, int]:
    """
    SMILES를 입력받아 해당 분자의 피부과민성 여부를 예측합니다.
    피부과민성 활성에 중요한 Key Event 4개에 대한 예측 결과로 나타납니다.
    KE1: DPRA
    KE2: Keratinosens
    KE3: h-CLAT
    KE4: LLNA

    Args:
        SMILES (str): SMILES

    Returns:
        str: '예측된' 피부과민성 KEs 활성 여부
    """
    skin_sens = skin_main([SMILES])
    return skin_sens



def get_openai_tools() -> List[dict]:
    functions = [
        QUERY_to_CAS,
        QUERY_to_SMILES,
        SMILES_to_USE,
        search_news, 
        python_repl_tool,
        SMILES_to_skin_sensitization,
    ]

    return functions
