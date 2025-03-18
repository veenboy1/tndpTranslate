import opt_problems as op
import networkx as nx
import parameters
from network_functions import create_sioux_falls, parse_demand

# Testing Sioux Falls
graph = create_sioux_falls(draw_net=False)
demand = parse_demand(parameters.sf_demand_file)

# Implementing custom classes
sit = op.Situation(graph, demand)
budget = 60
options = op.MasterOptions(freqwts=[0.5, 1.0, 1.5], gamma=[.10, .25, .7])
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

# subproblem.setup()