import opt_problems as op
import parameters
from network_functions import create_sioux_falls, parse_demand, read_node_pos

# Sioux Falls implementation
graph = create_sioux_falls(draw_net=False)
demand = parse_demand(parameters.sf_demand_file)
pos = read_node_pos(parameters.sf_node_file)

# Implementing custom classes
sit = op.Situation(graph, demand)
budget = 300  # Still need to figure out a reasonable budget for this
epsilon = 1e-4  # Reasonable gap parameter
options = op.MasterOptions(freqwts=[0.5, 1.0, 1.5], gamma=[.10, .25, .7],
                           headings=True, max_len=36)

lines = [
    op.TransitLine(2, 21, [2, 6, 8, 16, 17, 19, 20, 21]),
    op.TransitLine(1, 21, [1, 3, 4, 11, 14, 23, 24, 21]),
    op.TransitLine(18, 24, [18, 16, 10, 11, 12, 13, 24])
]

# Compute lengths
for line in lines:
    line.compute_length(graph, 'Length ')

# Initial setup (same as before)
master = op.MasterProblem(sit, budget, options, lines)
master.setup()
master.solve()

# Create subproblem once
duals = master.dual_values
sub_opts = op.SubProblemDuals(duals)
subproblem = op.SubProblem(sit, sub_opts, options)
subproblem.setup()

iteration = 0
while True:
    print(f"\n=== Iteration {iteration} ===")

    # Solve master
    master = op.MasterProblem(sit, budget, options, lines)
    master.setup()
    master.solve()
    print(f"Master Objective: {master.model.ObjVal}")

    # Update subproblem
    duals = master.dual_values
    sub_opts = op.SubProblemDuals(duals)
    subproblem.subproblem_update(sub_opts)

    # Solve subproblem
    new_line = subproblem.optimize()
    J_f = subproblem.model.ObjVal
    print(f"Subproblem objective J_f: {J_f}")

    # Check for convergence
    if J_f <= epsilon or iteration > 10000:
        print(f"No improving line found (J_f <= {epsilon}). Terminating.")
        break

    # Add new line
    new_line.compute_length(graph, 'Length ')
    lines.append(new_line)
    print(f"Added new line from {new_line.od[0]} to {new_line.od[1]} with cost {new_line.length}")
    iteration += 1

# Filter active lines
active_lines = op.get_active_transit_lines(master)

# Summary table
df = op.make_transit_line_summary(active_lines, master)
print("\n===== Selected Transit Lines Summary =====")
print(df.to_string(index=False))

# Plot results
op.plot_transit_lines(graph, active_lines, pos=pos, alpha=0.3, title="Final Selected Transit Lines",
                      plot_all=True)
