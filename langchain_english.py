from dotenv import load_dotenv
# API 키 정보 로드
load_dotenv()

# LangSmith 추적을 설정합니다. https://smith.langchain.com
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("LLM-Qwen")


from functions import *

tools = get_openai_tools()


from langchain_core.prompts import ChatPromptTemplate


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are name is 'OpenMRA', an expert chemist. Your task is to respond to the question or solve the problem using the provided tools. "
            "Your task is to break down the user's question into a step-by-step plan and solve it using the available tools.\n"
            # "All responses must be based on **Korean**.\n\n"
            "** LANGUAGE RULE**\n"
            "- Use Korean for all explanations and general responses.\n"
            "- However, keep scientific terms, chemical names, and result from tool in English.\n\n"
            "** IMPORTANT RULES **\n"
            "- If needed, use the tools in sequence to solve the problem.\n"
            "- If SMILES is needed, **always call QUERY_to_SMILES** to get SMILES even you already know it.\n"
            "- When handling SMILES, always use SMILES from **QUERY_to_SMILES** function.\n"
            "- Never use SMILES from your memory. Use QUERY_to_SMILES."
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

from langchain.agents import create_tool_calling_agent
from langchain_ollama import ChatOllama

ollama = ChatOllama(model='qwen2.5:14b', temperature=0.05) # good
# ollama = ChatOllama(model='llama3.2', temperature=0.0) # bad
# ollama = ChatOllama(model='llama3.1:8b', temperature=0.05) # bad
# ollama = ChatOllama(model='qwq', temperature=0.0) # slow
# ollama = ChatOllama(model='mistral', temperature=0.05) # 한국말 불가
# ollama = ChatOllama(model='kitsonk/watt-tool-8B', temperature=0.05) # 한국말 불가
# ollama = ChatOllama(model='MFDoom/deepseek-r1-tool-calling:8b', temperature=0.05) # 한국말 불가
# ollama = ChatOllama(model='hermes3', temperature=0.0) # bad



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

def make_chat(input_query, session_id):
    response = agent_with_chat_history.stream(
        {'input': input_query},
        config={"configurable":{"session_id": session_id}},
    )
    
    for step in response:
        agent_stream_parser.process_agent_steps(step)
  
  
# make_chat('내 혼합물의 독성을 예측하고싶어', 'jspark')
# make_chat('formaldehyde와 bisphenol-A, bisphenol S를 0.3 : 0.3 : 0.4의 비율로 넣은 혼합물의 독성을 예측해줘', 'jspark')

# make_chat('안녕 넌 뭐하는애야?', 'jspark')

# make_chat('내 이름은 종서야', 'jspark')

# make_chat('오늘 점심 뭐먹었어?', 'jspark')

# make_chat('나랑 놀자', 'jspark')
    
make_chat('cinnamaldehyde의 SMILES를 알려줘', 'jspark')

make_chat('혹시 기능도 알려줄래?', 'jspark')

make_chat('morphine의 피부과민성은?', 'jspark')

# make_chat('14371-10-9의 SMILES를 알려줘', 'jspark')

# print('\n\n\n')
# response = agent_with_chat_history.stream(
#     {'input': '내가 뭐 물어봤었고 대답이 뭐였지? 기억나?'},
#     config={"configurable":{"session_id":"jspark2"}},
# )

# for step in response:
#     agent_stream_parser.process_agent_steps(step)


# response = agent_with_chat_history.stream(
#     {'input': '내 이름이 뭐게?'},
#     config={"configurable":{"session_id":"jspark"}},
# )
# for step in response:
#     agent_stream_parser.process_agent_steps(step)

# response = agent_with_chat_history.stream(
#     {'input': 'morphine의 SMILES를 알려줘'},
#     config={"configurable":{"session_id":"jspark"}},
# )
# for step in response:
#     agent_stream_parser.process_agent_steps(step)

# response = agent_with_chat_history.stream(
#     {'input': 'i want to know SMILES of bisphenol A'},
#     config={"configurable":{"session_id":"jspark"}},
# )
# for step in response:
#     agent_stream_parser.process_agent_steps(step)


# response = agent_with_chat_history.stream(
#     {'input': 'i want to know about functional use'},
#     config={"configurable":{"session_id":"jspark"}},
# )
# for step in response:
#     agent_stream_parser.process_agent_steps(step)


# response = agent_with_chat_history.stream(
#     {'input': 'I want to know skin-sensitization of it.'},
#     config={"configurable":{"session_id":"jspark"}},
# )
# for step in response:
#     agent_stream_parser.process_agent_steps(step)



# result = agent_executor.invoke({"input": "안녕 난 박종서야"})
# result = agent_executor.invoke({"input": "내 이름이 뭐게", "chat_hisotry": result})


