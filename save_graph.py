"""
Run this script once to generate the LangGraph agent flow diagram as a PNG.
"""
from src.graph import app as agent_graph

output_path = "agent_graph.png"

png_bytes = agent_graph.get_graph().draw_mermaid_png()

with open(output_path, "wb") as f:
    f.write(png_bytes)

print(f"Graph saved to: {output_path}")
