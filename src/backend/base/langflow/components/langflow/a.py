from langmem import create_prompt_optimizer

# Multiple conversations showing what to improve
trajectories = [
    # Conversation with no annotations (just the conversation)
    (
        [
            {"role": "user", "content": "Tell me about Mars"},
            {"role": "assistant", "content": "Mars is the fourth planet..."},
            {"role": "user", "content": "I wanted more about its moons"},
        ],
        None,
    ),
    (
        [
            {"role": "user", "content": "What are Mars' moons?"},
            {
                "role": "assistant",
                "content": "Mars has two moons: Phobos and Deimos...",
            },
        ],
        {
            "score": 0.9,
            "comment": "Should include more details and recommended follow-up questions",
        },
    ),
    # Annotations can be of different types, such as edits/revisions!
    (
        [
            {"role": "user", "content": "Compare Mars and Earth"},
            {"role": "assistant", "content": "Mars and Earth have many differences..."},
        ],
        {"revised": "Earth and Mars have many similarities and differences..."},
    ),
]



optimizer = create_prompt_optimizer(
    "anthropic:claude-3-5-sonnet-latest",
    kind="metaprompt",
    config={"max_reflection_steps": 1, "min_reflection_steps": 0},
)
updated = optimizer.invoke(
    {"trajectories": trajectories, "prompt": "You are a planetary science expert"}
)
print(updated)