from langchain_core.runnables import RunnableConfig  # noqa: INP001
from langgraph.store.base import BaseStore


def config_namespace(name_space, config: RunnableConfig | None) -> str | None:
        ns_ = []
        for ns in name_space:
            if ns.startswith("{") and ns.endswith("}"):
                key = ns.replace("{", "").replace("}", "").strip()
                if config and "configurable" in config and key in config["configurable"]:
                    ns_.append(str(config["configurable"].get(key)))
                else:
                    msg = f"Not found {{{key}}} in config. Please check your config in GraphRunner again."
                    raise ValueError(msg)
            else:
                ns_.append(ns)
        return tuple(ns_)


async def extract_memory(
    get_from_mem_addons_input,
    memories_input,
    store: BaseStore | None = None,
    config: RunnableConfig | None = None
):
    # Format system prompt and input value with memory
    if store and get_from_mem_addons_input:
        for get_from_mem in get_from_mem_addons_input:
            mem = await get_from_mem.get_mem_func(store, config=config)
            mem_format = get_from_mem.mem_format
            if mem:
                if isinstance(mem, list):
                    for m in mem:
                        if m not in memories_input:
                            memories_input.append({m: mem_format})
                elif mem not in memories_input:
                    memories_input.append({mem: mem_format})
        # print(f"Retrieved memories: {self.memories}")
    return memories_input


async def store_memory(
    put_to_mem_addons_input,
    result: dict,
    store: BaseStore | None = None,
    config: RunnableConfig | None = None
):
    if store and put_to_mem_addons_input:
        for put_mem_addon in put_to_mem_addons_input:
            await put_mem_addon.store_mem_func(store, result, config)
