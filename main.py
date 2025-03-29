import networkx
from opt_problems import MasterProblem, SubProblem
import opt_problems as op
from gurobipy import tupledict


def column_generation(network, budget, options, initial_lines, max_iterations=100):
    # Initialize problems
    master = MasterProblem(network, budget, options, initial_lines)

    # Step 1: Set up master problem
    master.setup()

    # Step 2: Run master with initial lines:
    master.solve()
    dual_values = op.SubProblemDuals(master.dual_values)

    # Step 3: Set up the SubProblem
    sub = SubProblem(network, dual_values, options)

    for iteration in range(max_iterations):
        print(f"Iteration {iteration}: Solving Master Problem...")

        # Step 2: Solve the master problem
        master.solve()

        # Step 3: Pass dual values to subproblem
        print("Solving Subproblem...")
        sub.setup()
        sub.solve()

        # Check if subproblem generated a new line
        if sub.new_line is None:
            print("No profitable lines generated. Terminating.")
            break

        # Step 4: Add new line to the master problem
        print(f"Adding line to Master Problem: {sub.new_line}")
        master.add_line(sub.new_line)

    return master


if __name__ == '__main__':
    # Situation creation
    network = networkx.DiGraph()
    demand = tupledict()
    sit = op.Situation(network, demand)

    # Lines to start
    lines = []

    # Other parameters
    budget = 1e7
    options = op.MasterOptions() # <- if you don't want the default values, change them here
    max_iterations = 100

    column_generation(sit, budget, options, lines, max_iterations)