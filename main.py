from opt_problems import MasterProblem, SubProblem, NetworkDemand
import parameters as p


def column_generation(network, budget, max_iterations=100):
    # Initialize problems
    master = MasterProblem(network, budget, options={})
    sub = SubProblem(network, options={})

    # Step 1: Set up master problem
    master.setup()

    for iteration in range(max_iterations):
        print(f"Iteration {iteration}: Solving Master Problem...")

        # Step 2: Solve the master problem
        master.solve()

        # Step 3: Pass dual values to subproblem
        print("Solving Subproblem...")
        sub.setup(master.dual_values)
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
    print('hello world')
    # TODO: I left off working on this class
    # nd = NetworkDemand(p.G, )