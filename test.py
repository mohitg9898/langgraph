from langgraph.graph import StateGraph

def test_langgraph():
    # Create a simple state graph
    graph = StateGraph(dict)
    
    # StateGraph uses add_node and add_edge methods
    graph.add_node("A", lambda x: x)
    graph.add_node("B", lambda x: x)
    graph.add_edge("A", "B")
    graph.set_entry_point("A")
    graph.set_finish_point("B")
    
    # Compile the graph
    compiled = graph.compile()
    
    assert compiled is not None
    
    print("LangGraph test passed!")
if __name__ == "__main__":
    test_langgraph()
