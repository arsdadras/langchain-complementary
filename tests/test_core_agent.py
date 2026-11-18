"""
Unit tests for langcomp.core.agent and SmartState.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START

from langcomp.core.agent import AgentBuilder, agent
from langcomp.core.state import SmartState


def test_smart_state_normalization():
    messages = [
        "Hello from string",
        {"role": "user", "content": "Hello from user dict"},
        {"role": "assistant", "content": "Hello from ai dict"},
        HumanMessage(content="Hello from object")
    ]
    normalized = SmartState.normalize_messages(messages)
    assert len(normalized) == 4
    assert isinstance(normalized[0], HumanMessage)
    assert normalized[0].content == "Hello from string"
    assert isinstance(normalized[2], AIMessage)
    assert normalized[2].content == "Hello from ai dict"


def test_agent_builder_custom_nodes():
    def dummy_node(state):
        msgs = state.get("messages", [])
        return {"messages": msgs + [AIMessage(content="Processed by dummy_node")]}

    builder = AgentBuilder("TestAgent")
    builder.add_node("step1", dummy_node)
    builder.add_edge(START, "step1")
    builder.add_edge("step1", END)

    agent_inst = builder.compile()
    assert agent_inst.name == "TestAgent"

    result = agent_inst.invoke("User query")
    assert "messages" in result
    assert len(result["messages"]) >= 2
    assert result["messages"][-1].content == "Processed by dummy_node"


def test_agent_decorator():
    @agent(name="DecoratedNode")
    def node_fn(state):
        return {"messages": [AIMessage(content="Decorated response")]}

    result = node_fn.invoke("Hi")
    assert result["messages"][-1].content == "Decorated response"
