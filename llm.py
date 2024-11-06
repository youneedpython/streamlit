from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.prompts import FewShotChatMessagePromptTemplate
from config import answer_examples

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


## [함수 정의] ++++++++++++++++++++++++++++++
def get_retriever():
  ## embedding
  model_embedding = OpenAIEmbeddings(model='text-embedding-3-large')

  ## vector database에 저장된 인덱스 정보 가져오기
  index_name = 'tax-markdown-index'

  database = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=model_embedding,
  )

  ## 크로마에 있는 함수로 similarity search 수행
  retriever = database.as_retriever()

  return retriever

def get_history_retriever():
  retriever = get_retriever()
  llm = get_llm()
  ## LangSmith API KEY 설정
  #LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
  # prompt = hub.pull('rlm/rag-prompt', api_key=LANGCHAIN_API_KEY)

  ## QA chain 생성
  # qa_chain = RetrievalQA.from_chain_type(
  #   llm, ## 모델 지정
  #   retriever=retriever, 
  #   chain_type_kwargs={'prompt': prompt}, ## prompt : prompt 작성 후, 별도의 invoke 할 필요 없음
  # )

  ## 시스템 프롬프트
  contextualize_q_system_prompt = (
      "Given a chat history and the latest user question "
      "which might reference context in the chat history, "
      "formulate a standalone question which can be understood "
      "without the chat history. Do NOT answer the question, "
      "just reformulate it if needed and otherwise return it as is."
  )

  ## 채팅 프롬프트
  contextualize_q_prompt = ChatPromptTemplate.from_messages(
      [
          ("system", contextualize_q_system_prompt),
          MessagesPlaceholder("chat_history"),
          ("human", "{input}"),
      ]
  )

  ## 리트리버
  history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
  )

  return history_aware_retriever

def get_llm(model='gpt-3.5-turbo'):
  ## ChatGPT의 LLM 모델 gpt-3.5-turbo 지정
  llm = ChatOpenAI(model=model)

  return llm


def get_dictionary_chain():
  ## query 내 "직장인"을 "거주자"로 변경하는 chain 추가
  dictionary = ['사람을 나타내는 표현 -> 거주자']

  prompt = ChatPromptTemplate.from_template(f'''
          우리 사전을 참고하여 사용자의 질문을 변경해주세요.
          만약 변경할 필요가 없으면, 사용자 질문을 그대로 사용하세요.
          그런 경우에는 질문만 리턴하세요.                                  
          사전: {dictionary}

          질문: {{question}}
          ''')

  llm = get_llm()
  dictionary_chain = prompt | llm | StrOutputParser()

  return dictionary_chain


def get_qa_chain():
  llm = get_llm()

  ## few-shot prompt template
  example_prompt = ChatPromptTemplate.from_messages(
      [
          ("human", "{input}"),
          ("ai", "{answer}"),
      ]
  )
  few_shot_prompt = FewShotChatMessagePromptTemplate(
      example_prompt=example_prompt,
      examples=answer_examples,
  )

  ## history
  system_prompt = (
    "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요. "
    "아래에 제공된 문서를 활용해서 답변하고 "
    "답변을 모르면 모른다고 답변해주세요. "
    "2-3 문장정도의 짧은 내용의 답변을 원합니다. "
    "\n\n"
    "{context}"
  )
  qa_prompt = ChatPromptTemplate.from_messages(
      [
          ("system", system_prompt),
          few_shot_prompt,
          MessagesPlaceholder("chat_history"),
          ("human", "{input}"),
      ]
  )

  history_aware_retriever = get_history_retriever()
  question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
  rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

  ## history  
  conversational_rag_chain = RunnableWithMessageHistory(
      rag_chain,
      get_session_history,
      input_messages_key="input",
      history_messages_key="chat_history",
      output_messages_key="answer",
  ).pick('answer')
  #############################

  # return qa_chain
  return conversational_rag_chain

## [AI Message 함수 정의] ++++++++++++++++++++++++++++++
def get_ai_message(user_message):
  dictionary_chain = get_dictionary_chain()
  qa_chain = get_qa_chain()

  tax_chain = {'input': dictionary_chain} | qa_chain
  ai_message = tax_chain.stream({'question': user_message},
                                config={
        "configurable": {"session_id": "abc123"}
    }, 
  )

  #return ai_message['answer']
  return ai_message
## +++++++++++++++++++++++++++++++++++++++++++++++++++++
