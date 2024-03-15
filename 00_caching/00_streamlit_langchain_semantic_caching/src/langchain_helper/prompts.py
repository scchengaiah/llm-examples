from langchain_core.prompts import ChatPromptTemplate

CONDENSE_QUESTION_PROMPT_STR = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in English language. Avoid presenting empty standalone questions. If ambiguity arises, retain the follow up question as is. Do not include any other content other than the rephrased question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template(CONDENSE_QUESTION_PROMPT_STR)

QA_PROMPT_STR = """You are a friendly chatbot assistant that responds in a conversational manner to users' question on company's policies. 
Respond in 1-2 complete sentences, unless specifically asked by the user to elaborate on something. Use "Context" to inform your answers.
Do not make up answers if the question is out of "Context".
Respond to greetings or compliments in a positive manner and let the user know your capability.

---
Context:
{context}
---
Question:
{question}
---
Response:
"""
QA_PROMPT = ChatPromptTemplate.from_template(QA_PROMPT_STR)
