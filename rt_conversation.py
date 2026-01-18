from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile")#loading the llm model
sys_message = SystemMessage(content="You are a helpful assistant ")#system message
chat_history=[]#to store teh conversation
while (True):
    #alsways run indefinitely
    query = input("User: ")#take user input   is the huiman message
    if(query.lower()=="exit"):

        break
    chat_history.append(HumanMessage(content=query))#append the human message to chat history
#to give this mesages to llm
    result=llm.invoke(chat_history)
    print("ai response:"+result.content)#result .content is only my ai message
    chat_history.append(AIMessage(content=result.content))
#outside the loop it should show thye history
print("message history")
print(chat_history)