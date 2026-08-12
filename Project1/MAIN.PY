import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

@tool
def calculator(a: float, b: float) -> str:
    """Useful for performing basic arithmetic operations."""
    print("Tool has been called.")
    return f"The sum of {a} and {b} is {a + b}."

def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in .env file.")
        return

    model = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",  # free-tier eligible, no billing required
        temperature=0,
        google_api_key=api_key,
    )

    tools = [DuckDuckGoSearchRun(), calculator]
    agent_executor = create_react_agent(model=model, tools=tools)

    print("Welcome! I'm your AI assistant. Type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if not user_input:
            continue

        print("\nAssistant: ", end="")

        try:
            for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content=user_input)]}
            ):
                if "agent" in chunk and "messages" in chunk["agent"]:
                    for message in chunk["agent"]["messages"]:
                        content = message.content
                        if isinstance(content, list):
                            # Gemini can return structured content parts
                            content = "".join(
                                part.get("text", "") if isinstance(part, dict) else str(part)
                                for part in content
                            )
                        print(content, end="")
            print()
        except Exception as e:
            print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()