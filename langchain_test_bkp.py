from dotenv import load_dotenv
# API 키 정보 로드
load_dotenv()

# LangSmith 추적을 설정합니다. https://smith.langchain.com
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("LLM-Qwen")


from functions import *

tools = get_openai_tools()


from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate

# 프롬프트 생성
# 프롬프트는 에이전트에게 모델이 수행할 작업을 설명하는 텍스트를 제공합니다. (도구의 이름과 역할을 입력)
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are an expert chemist. Your task is to respond to the question or solve the problem to the best of your ability using the provided tools. "
#             "Your task is to **break down a user's question into a step-by-step plan**, using available tools to solve it.\n"
#             "** Rules **\n"
#             # "- Only use a SMILES string if it is:\n"
#             # "  1. It is explicitly provided by the user, OR\n"
#             # "  2. It has been retrieved using a tool (e.g. QUERY_to_SMILES)\n"
#             '- When using a tool, always refer to the exact output shown after "Observation:".\n'
#             '- Do not try to rewrite or summarize the output. Just use it directly.\n'
#             "- NEVER make a response with a Chinese.\n"
#         ),
#         ("placeholder", "{chat_history}"),
#         ("human", "{input}"),
#         ("placeholder", "{agent_scratchpad}"),
#     ]
# )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 전문 화학자입니다. 주어진 도구들을 활용하여 질문에 답하거나 문제를 최선의 방법으로 해결하는 것이 임무입니다. "
            "당신의 임무는 사용자의 질문을 단계별 계획으로 나누고, 사용 가능한 도구들을 활용해 이를 해결하는 것입니다.\n"
            "** 규칙 **\n"
            '- SMILES는 사용자가 입력하거나, QUERY_to_SMILES의 결과만을 사용하세요.\n'
            '- 도구 출력 값을 절대로 수정하지 말고 그대로 사용하세요(특히 SMILES) .\n'
            '- 한국어로만 대답하세요.\n' 
            '- 만약 도구 출력 값이 영어라면 영어 그대로 사용하세요.\n'
        ),
        (
            "ai",
            "Input: bisphenol-A\n"
            "Observation: CC(C)(C1=CC=C(C=C1)O)C2=CC=C(C=C2)O\n"
            "Tool: SMILES_to_USE\n"
            "Input: CC(C)(C1=CC=C(C=C1)O)C2=CC=C(C=C2)O"
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

from langchain.agents import create_tool_calling_agent
from langchain_ollama import ChatOllama

ollama = ChatOllama(model='qwen2.5:7b', temperature=0.01)
# ollama = ChatOllama(model='llama3.:2', temperature=0.2)
# ollama = ChatOllama(model='qwq', temperature=0.05)
agent = create_tool_calling_agent(ollama, tools, prompt)

from langchain.agents import AgentExecutor

# gpt_agent 실행
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)

# result = agent_executor.invoke({"input": "AI 투자와 관련된 뉴스를 검색해 주세요."})
# result = agent_executor.invoke({"input": "formaldehyde의 용도를 알려주세요"})
# result = agent_executor.invoke({"input": "bisphenol-A의 SMILES를 알려주세요"})

# print("Agent 실행 결과:")
# print(result["output"])


from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_teddynote.messages import AgentStreamParser

store = {}

def get_session_history(session_ids):
    if session_ids not in store:
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

agent_stream_parser = AgentStreamParser()




response = agent_with_chat_history.stream(
    {'input': '안녕 난 박종서야'},
    config={"configurable":{"session_id":"jspark"}},
)

for step in response:
    agent_stream_parser.process_agent_steps(step)


response = agent_with_chat_history.stream(
    {'input': '내 이름이 뭐게?'},
    config={"configurable":{"session_id":"jspark"}},
)
for step in response:
    agent_stream_parser.process_agent_steps(step)

response = agent_with_chat_history.stream(
    {'input': '비스페놀 A의 기능을 알려줘'},
    config={"configurable":{"session_id":"jspark"}},
)
for step in response:
    agent_stream_parser.process_agent_steps(step)


# result = agent_executor.invoke({"input": "안녕 난 박종서야"})
# result = agent_executor.invoke({"input": "내 이름이 뭐게", "chat_hisotry": result})

