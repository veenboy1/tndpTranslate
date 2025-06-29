import opt_problems as op
import networkx as nx
import parameters
from network_functions import create_sioux_falls, parse_demand, read_node_pos

# Testing Sioux Falls
graph = create_sioux_falls(draw_net=False)
demand = parse_demand(parameters.sf_demand_file)
pos = read_node_pos(parameters.sf_node_file)

# Implementing custom classes
sit = op.Situation(graph, demand)
budget = 600
options = op.MasterOptions(freqwts=[0.5, 1.0, 1.5], gamma=[.10, .25, .7],
                           headings=True, max_len=36)
lines = [
    op.TransitLine(2, 21, [2, 6, 8, 16, 17, 19, 20, 21]),
    op.TransitLine(1, 21, [1, 3, 4, 11, 14, 23, 24, 21]),
    op.TransitLine(18, 24, [18, 16, 10, 11, 12, 13, 24])
]

for i in range(len(lines)):
    lines[i].compute_length(graph, 'Length ')
    print(f'Cost of line {i}: {lines[i].length}')

master = op.MasterProblem(sit, budget, options, lines)
master.setup()
master.solve(False, False)
print(master.dual_values)
for v in master.model.getVars():
    if v.varName[0] == 'x':
        print(f"{v.VarName}: {v.X}")

s_opts = op.SubProblemDuals(master.dual_values)

subproblem = op.SubProblem(sit, s_opts, options)

subproblem.setup()
print(subproblem.model.getConstrs())

first_transit_line = subproblem.optimize()

import matplotlib.pyplot as plt

# Draw base network
plt.figure(figsize=(10, 8))
nx.draw_networkx_edges(graph, pos, edge_color='lightgray', width=1)
nx.draw_networkx_nodes(graph, pos, node_size=50, node_color='gray')

# Highlight the new transit line if it exists
if first_transit_line:
    line_edges = list(zip(first_transit_line.stops[:-1], first_transit_line.stops[1:]))
    nx.draw_networkx_edges(graph, pos, edgelist=line_edges, edge_color='red', width=2.5)
    nx.draw_networkx_nodes(graph, pos, nodelist=first_transit_line.stops, node_color='red', node_size=70)
    nx.draw_networkx_labels(graph, pos, labels={n: n for n in first_transit_line.stops}, font_size=8, font_color='black')
    plt.title(f"Generated Transit Line from {first_transit_line.od[0]} to {first_transit_line.od[1]}")
else:
    plt.title("No valid transit line was found.")

plt.axis('off')
plt.tight_layout()
plt.show()