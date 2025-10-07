from langmem import create_prompt_optimizer
from langchain.chat_models.base import init_chat_model
from langgraph.prebuilt import create_react_agent



# # Multiple conversations showing what to improve
# trajectories = [
#     # Conversation with no annotations (just the conversation)
#     (
#         [
#             "Tell me about Mars",
#             "Mars is the fourth planet...",
#             "I wanted more about its moons",
#         ],
#         None,
#     ),
#     (
#         [
#            "What are Mars' moons?",
#             "Mars has two moons: Phobos and Deimos...",
#         ],
#         {
#             "score": 0.9,
#             "comment": "Should include more details and recommended follow-up questions",
#         },
#     ),
#     # Annotations can be of different types, such as edits/revisions!
#     (
#         [
#             "Compare Mars and Earth",
#             "Mars and Earth have many differences...",
#         ],
#         {"revised": "Earth and Mars have many similarities and differences..."},
#     ),
# ]

# optimizer = create_prompt_optimizer(
#     llm,
#     kind="metaprompt",
#     config={"max_reflection_steps": 1, "min_reflection_steps": 0},
# )
# updated = optimizer.invoke(
#     {"trajectories": trajectories, "prompt": "You are a planetary science expert"}
# )
# print(updated)

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from typing import Optional

class Dog(BaseModel):
    breed: str = Field(description="The breed of the dog")
    traits: list[str] = Field(description="List of traits of the dog")

class State(BaseModel):
    question: str = Field(description="The question to be answered")
    dog: Optional[Dog] = Field(description="Information about the dog", default=None)

def node_1(state: State):
    # llm = llm.with_structured_output(Dog)
    # prompt = f"What is the trait of {state.question}?"
    a = {"breed": "Golden Retriever", "traits": ["Friendly", "Intelligent", "Devoted"]}
    d = Dog(**a)
    return {"dog": d}

builder = StateGraph(state_schema=State)
builder.add_node("node_1", node_1)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", END)

graph = builder.compile()
result = graph.invoke({"question": "Golden Retriever"})

print(result)


async def f_(state, output_types, memories, model_schema, runtime):
    if runtime.context:
        return {"question":f"{runtime.context.breed_dog} is a great dog!"}
    return {"question": "Please provide context with breed_dog."}