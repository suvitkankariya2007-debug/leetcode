import os
import datetime
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chat_models import ChatOpenAI

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI model
llm = ChatOpenAI(model="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))

# Define tool
@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    return datetime.datetime.now().strftime(format)

# Define ReAct prompt directly
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is your current time in the format %Y-%m-%d %H:%M:%S?")
])

system_prompt_text = prompt_template.template

# List of tools
tools = [get_system_time]

# Create agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_template
)

# Create executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Query
query = "What is your current time in the format %Y-%m-%d %H:%M:%S?"

# Run
result = agent_executor.invoke({"input": query})
print(result["output"])
