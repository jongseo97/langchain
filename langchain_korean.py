# from dotenv import load_dotenv
# # API 키 정보 로드
# load_dotenv()

# # LangSmith 추적을 설정합니다. https://smith.langchain.com
# from langchain_teddynote import logging

# # 프로젝트 이름을 입력합니다.
# logging.langsmith("LLM-Qwen")

from langchain.agents import create_tool_calling_agent, AgentExecutor

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_teddynote.messages import AgentStreamParser
from langchain_ollama import ChatOllama
from functions_korean import *


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 expert chemist입니다. 주어진 도구들을 활용하여 질문에 답하거나 문제를 최선의 방법으로 해결하는 것이 임무입니다. "
            "당신의 임무는 사용자의 질문을 단계별 계획으로 나누고, 사용 가능한 도구들을 활용해 이를 해결하는 것입니다.\n"
            "답변은 한국어를 기반으로 진행하세요.\n"
            "** 중요 규칙 **\n"
            '- 필요한 경우 도구를 순차적으로 호출해서 문제를 해결하세요.\n'
            '- SMILES가 필요한 경우 QUERY_to_SMILES를 이용해서 SMILES를 먼저 저장하세요. SMILES를 이미 알고 있어도 QUERY_to_SMILES를 사용하세요.\n'
            '- 화학물질명을 다룰 때에는 영어로 번역해서 사용하세요.\n'
            '- SMILES를 다룰 때에는 QUERY_to_SMILES를 적극적으로 사용하세요.\n'
            "- tool output이 예측된 결과인 경우 실제와는 다를 수 있다고 안내하세요."
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


ollama = ChatOllama(model='qwen2.5:14b', temperature=0.05) # good
# ollama = ChatOllama(model='llama3.2', temperature=0.0) # bad
# ollama = ChatOllama(model='llama3.1:8b', temperature=0.0) # bad
# ollama = ChatOllama(model='qwq', temperature=0.0) # slow
# ollama = ChatOllama(model='mistral', temperature=0.0) # 한국말 불가
# ollama = ChatOllama(model='kitsonk/watt-tool-8B', temperature=0.0) # 한국말 불가
# ollama = ChatOllama(model='MFDoom/deepseek-r1-tool-calling:8b', temperature=0.0) # 한국말 불가
# ollama = ChatOllama(model='hermes3', temperature=0.0) # bad

tools = get_openai_tools()
agent = create_tool_calling_agent(ollama, tools, prompt)

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
    {'input': 'bisphenol-A의 SMILES를 알려줘'},
    config={"configurable":{"session_id":"jspark"}},
)
for step in response:
    agent_stream_parser.process_agent_steps(step)

response = agent_with_chat_history.stream(
    {'input': 'morphine의 SMILES를 알려줘'},
    config={"configurable":{"session_id":"jspark"}},
)
for step in response:
    agent_stream_parser.process_agent_steps(step)


response = agent_with_chat_history.stream(
    {'input': '기능도 알려줘'},
    config={"configurable":{"session_id":"jspark"}},
)
for step in response:
    agent_stream_parser.process_agent_steps(step)

response = agent_with_chat_history.stream(
    {'input': 'morphine의 피부과민성 여부도 알려줘'},
    config={"configurable":{"session_id":"jspark"}},
)
for step in response:
    agent_stream_parser.process_agent_steps(step)



# result = agent_executor.invoke({"input": "안녕 난 박종서야"})
# result = agent_executor.invoke({"input": "내 이름이 뭐게", "chat_hisotry": result})

