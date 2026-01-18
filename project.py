from dotenv import load_dotenv
load_dotenv()
from tavily_client import TavilyClient
import datetime
import os
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage
from langchain_classic import hub
from langchain.agents import create_agent
import requests
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent

TAVILY_API_KEY=os.getenv("sss")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
OPENWAEATHER_API_KEY=os.getenv("OPENWEATHER_API_KEY")

# Initialize OpenAI model
llm = ChatOpenAI(model="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))

@tool
def get_weather(city: str, api_key: str = OPENWAEATHER_API_KEY) -> dict:
    """
    Returns current weather info for a given city using WeatherAPI.
    """

    if api_key is None:
        return {"error": "Missing API key in environment"}

    url = (
        f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
    )

    response = requests.get(url)

    if response.status_code != 200:
        return {
            "error": f"API error: {response.status_code}",
            "details": response.text
        }

    data = response.json()

    return {
        "city": data["location"]["name"],
        "temperature": data["current"]["temp_c"],
        "feels_like": data["current"]["feelslike_c"],
        "humidity": data["current"]["humidity"],
        "weather": data["current"]["condition"]["text"],
        "wind_speed": data["current"]["wind_kph"]
    }
# # Example Usage:
# # print(get_weather("Manipal", "YOUR_API_KEY"))




@tool
def tavily_search_tool(query: str, max_results: int = 5) -> dict:
    """This tool performs web search using Tavily web search API and this gives output in json format."""

    response = tavily_client.search(query=query, max_results=max_results)
    return response





@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Get the current system time in the specified format."""
    
    curr_time=datetime.datetime.now()
    formatted_time=curr_time.strftime(format)
    return formatted_time


# Pull ReAct prompt from the hub
react_prompt = hub.pull("hwchase17/react")


system_prompt_text = react_prompt.template


agent = create_agent(
    model=llm,
    tools=[get_system_time, tavily_search_tool],
    system_prompt=system_prompt_text,
)


query = "Do a web search and tell me who won IPL 2024 final?Also, give me today's date"

result = agent.invoke({"messages": [HumanMessage(content=query)]})
# print(result)
ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]

for i,msg in enumerate(ai_messages,1):
    print(f"AI Message {i}:\n{msg.content}\n")


if ai_messages:
    print("Final AI Message:\n", ai_messages[-1].content)

# Test the get_weather tool
if __name__ == "__main__":
    city = "Pune"  # Replace with your desired city
    weather_info = get_weather.run({"city": city})
    print("Weather Information:", weather_info)

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
