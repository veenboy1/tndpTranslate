from gurobipy import Model, GRB, quicksum
import gurobipy as gp
import networkx as nx

import parameters as params


# File for storing the classes used to implement the column generation


class MasterOptions:
    def __init__(self, nfreqs=3, freqwts=None, costwts=None, gamma=None, weight=None):
        """
        Initialize MasterOptions with default or user-provided values.

        Parameters:
            nfreqs (int): Number of frequency levels. Default is 3.
            freqwts (list): Relative ridership coefficients for each frequency. Default is [0.5, 1.0, 1.5].
            costwts (list): Relative cost coefficients for each frequency. Default is [1.0, 1.5, 2.0].
            gamma (list): Relative coverage coefficients for each frequency. Default is [.10, .25, .60].
        """
        self.nfreqs = nfreqs
        self.freqwts = freqwts if freqwts is not None else [0.5, 1.0, 1.5]
        self.costwts = costwts if costwts is not None else [1.0, 1.5, 2.0]  # note: cost per length of frequency (rho)
        self.gamma = gamma if gamma is not None else [.10, .25, .60]
        self.weight = weight if weight is not None else 'Length '  # Default value is the one in the SF network

        # Validate lengths
        if len(self.freqwts) != self.nfreqs or len(self.costwts) != self.nfreqs:
            raise ValueError("freqwts and costwts must have the same length as nfreqs.")


class Situation:
    """A class to represent the situation - the roads, nodes, and demand"""
    def __init__(self, graph, demand=None):
        """Initialize the NetworkDemand class"""
        # NetworkX graph
        self.network1 = graph

        # Gurobi TupleDict for demand
        self.demand = demand if demand is not None else gp.tupledict()

    def set_demand(self, origin, destination, value):
        """
        Set demand between two nodes

        Parameters:
        - origin: Origin node
        - destination: Destination node
        - value: Demand value
        """
        self.demand[origin, destination] = value

    def get_demand(self, origin, destination):
        """
        Get demand between two nodes

        Parameters:
        - origin: Origin node
        - destination: Destination node

        Returns:
        - Demand value (or None if not set)
        """
        return self.demand.get((origin, destination))

    def print_network_info(self):
        """
        Print basic information about the network
        """
        print(f"Nodes: {self.network1.nodes()}")
        print(f"Edges: {self.network1.edges()}")
        print("Demands:")
        for (origin, dest), value in self.demand.items():
            print(f"{origin} -> {dest}: {value}")


class TransitLine:
    def __init__(self, origin, destination, stops):
        """
        Initialize a TransitLine.

        Parameters:
            origin (int): The starting stop of the line.
            destination (int): The ending stop of the line.
            stops (list): List of stops along the line, including origin and destination.
        """
        self.od = (origin, destination)
        self.stops = stops
        self.length = 0  # Will be computed based on a given network

    def compute_length(self, network, weight):
        """
        Compute the total length of the transit line based on a given NetworkX graph.

        Parameters:
            network (nx.Graph): The NetworkX graph representing the transit network.
            weight: Weight used to calculate the length of the transit line.

        Returns:
            float: The total length of the line.
        """
        self.length = sum(
            nx.shortest_path_length(network, self.stops[i], self.stops[i + 1], weight=weight)
            for i in range(len(self.stops) - 1)
        )

        return self.length


class MasterProblem:
    def __init__(self, network, budget, options, linelist=None):
        self.network = network  # Transit network (nodes, edges, demand), type Situation
        self.budget = budget  # Total budget available, integer
        self.options = options  # MasterOptions containing nfreqs, freqwts, costwts
        self.model = None  # Gurobi model instance
        self.linelist = linelist if linelist is not None else []  # List of selected lines, type TransitLine
        self.dual_values = {}  # Store dual values after solving

    def add_line(self, origin, destination, stops, first_time=False):
        """Add a new transit line to the master problem."""
        new_line = TransitLine(origin, destination, stops)
        new_line.compute_length(self.network.network1)  # Compute its length using the graph
        self.linelist.append(new_line)
        # not sure if I want this line below - might be inefficient ?
        if not first_time:
            self.setup()  # Rebuild model with the new line

    def setup(self):
        """Set up the Gurobi model for the master problem."""
        self.model = Model("MasterProblem")

        n_lines = len(self.linelist)
        n_freqs = self.options.nfreqs

        # Decision Variables
        self.x = self.model.addVars(n_lines, n_freqs, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
        self.theta = self.model.addVars(self.network.demand.keys(), lb=0, ub=1, name="theta")

        # Objective: Maximize ridership
        self.model.setObjective(
            quicksum(
                self.network.demand[u, v] * self.theta[u, v]
                for u, v in self.network.demand.keys()
            ),
            GRB.MAXIMIZE,
        )

        # Constraints

        # 1. Budget Constraint: Sum of (cost per length * frequency * length) must be ≤ budget
        self.model.addConstr(
            quicksum(
                self.options.costwts[f] * line.length * self.x[l1, f]
                for l1, line in enumerate(self.linelist)
                for f in range(self.options.nfreqs)
            ) <= self.budget,
            "BudgetConstraint",
        )

        # 2. Commute Coverage Constraints

        # Iterate over all demand pairs (u, v)
        for u, v in self.network.demand.keys():
            # Find lines that cover both u and v
            lines_covering_uv = [
                l_idx for l_idx, line in enumerate(self.linelist)
                if u in line.stops and v in line.stops
            ]

            # Add the coverage constraint for this demand pair
            self.model.addConstr(
                self.theta[u, v] <= quicksum(
                    self.options.gamma[f] * self.x[l_idx, f]
                    for l_idx in lines_covering_uv
                    for f in range(self.options.nfreqs)
                ),
                f"Coverage_{u}_{v}"
            )

        # 3. Frequency Constraints (Each line can have only one frequency)
        if n_freqs > 1:
            self.model.addConstrs(
                (self.x.sum(l3, "*") <= 1 for l3 in range(n_lines)), "FrequencyChoice"
            )
        self.model.update()

    def solve(self, print_vars=False, print_duals=False):
        """
        Solve the optimization problem and optionally print all variable values.

        Args:
            print_vars (bool): If True, print the names and values of all variables.
            print_duals (bool): if True, print the dual values of the program
        """
        self.model.optimize()

        if self.model.status == 2:  # Optimal solution found
            print("\nOptimal solution found!")
            print(f"Objective value: {self.model.objVal:.2f}")

            self.dual_values = {constr.constrName: constr.Pi for constr in self.model.getConstrs()}

            if print_vars:
                print("\nVariable Values:")
                for var in self.model.getVars():
                    print(f"{var.varName}: {var.X:.2f}")

                print('\nOptimal Value:', self.model.objVal)

            if print_duals:
                print("\nDual Variables (Shadow Prices):")
                for constr in self.model.getConstrs():
                    print(f"{constr.ConstrName}: {constr.pi}")

        elif self.model.status == 3:  # Infeasible
            print("\nThe problem is infeasible.")
        elif self.model.status == 4:  # Unbounded
            print("\nThe problem is unbounded.")
        else:
            print(f"\nSolver status: {self.model.status}")


class SubProblemDuals:
    def prep_p(self):
        coverage_duals = {(int(key.split('_')[1]), int(key.split('_')[2])): value
                          for key, value in self.duals_dict.items() if key.startswith('Coverage')}
        return gp.tupledict(coverage_duals)

    def __init__(self, duals_dict):
        self.duals_dict = duals_dict
        self.p_dict = self.prep_p()  # duals associated with increase in service of OD (u, v)
        self.q = duals_dict['BudgetConstraint']  # dual variable associated with 1 unit increase in budget


class SubProblem:
    def __init__(self, situation, dual_values, options, f_index=0):
        """
        Initialize the subproblem.

        Parameters:
        - situation (type Situation): class containing network and demand information
        - dual_values (type SubProblemDuals): class containing subproblem dual values
        """
        self.situation = situation  # Class Situation
        self.p = dual_values.p_dict  # p_dict from SubProblemDuals
        self.q = max(dual_values.q, .001)  # budget dual value from SubProblemDuals
        self.model = gp.Model("SubProblem")  # just the gurobi model idk bruv
        self.options = options  # MasterOptions class

        # Frequency related parameters - chosen depending on the frequency selected
        self.gamma_f = self.options.gamma[f_index]
        self.rho_f = self.options.costwts[f_index]

        # Decision variables (to be initialized in build_model)
        self.h = None  # Transit links chosen
        self.g = None  # OD pairs covered
        self.source = None  # Choosing the source node
        self.sink = None  # Choosing the sink node

        # Delta dictionary bc I don't want to deal with a DiGraph right now
        # could be adjusted to have unequal costs in each direction
        self.delta = gp.tupledict({(u, v): self.situation.network1[u][v][self.options.weight]
                                   for u, v in self.situation.network1.edges})

    def setup(self):
        # Debug
        print('Begin Setup')

        # ----- Decision variables ----- #
        # vars for which demand will be served and which edges will be included
        self.h = self.model.addVars(self.situation.network1.edges(), vtype=GRB.BINARY, name="h")
        self.g = self.model.addVars(self.situation.demand.keys(), vtype=GRB.BINARY, name="g")

        # vars for selecting a source and sink
        self.source = self.model.addVars(self.situation.network1.nodes(), vtype=GRB.BINARY, name="source")
        self.sink = self.model.addVars(self.situation.network1.nodes(), vtype=GRB.BINARY, name="source")

        self.model.update()

        # ----- Objective function ----- #
        # first sum:
        first_term = self.gamma_f * gp.quicksum(self.p[u, v] * self.g[u, v] for u, v in self.g.keys())
        # second sum:
        second_term = self.rho_f * gp.quicksum(self.q * self.delta[u, v] * self.h[u, v] for u, v in self.h.keys())
        # set objective function
        self.model.setObjective(first_term - second_term, GRB.MAXIMIZE)

        self.model.update()

        # ----- Model Constraints ----- #

        # Constraint 6e:
        self.model.addConstrs(
            (self.g[u, v] <= gp.quicksum(self.h[v_prime, u] for v_prime in self.situation.network1.predecessors(u))
             for u, v in self.g.keys()),
            name="demand_inflow"
        )

        # Constraint 6f: 
        self.model.addConstrs(
            (self.g[u, v] <= gp.quicksum(self.h[u_prime, v] for u_prime in self.situation.network1.predecessors(v))
             for u, v in self.g.keys()),
            name="demand_inflow_v"
        )

        # Source and Sink Constraints: -> not in paper but in their code
        self.model.addConstr(
            (gp.quicksum(self.source) == 1), name="one_source"
        )

        self.model.addConstr(
            (gp.quicksum(self.sink) == 1), name="one_sink"
        )

        # The ones that need to be lazy otherwise they take too long
        # TODO: Code the lazy constraints

        # Update model
        self.model.update()

    def subproblem_update(self, dual_values, f_index=None):
        """
        Update the subproblem without rebuilding the whole model.

        :param dual_values: class containing subproblem dual values
        :param f_index: optional parameter to specify the frequency of the line
        """
        # ----- Input Updates ----- #
        # Input values adjusted
        self.p = dual_values.p_dict  # p_dict from SubProblemDuals
        self.q = max(dual_values.q, .001)  # budget dual value from SubProblemDuals

        # Frequency related parameters changed if specified
        if f_index:
            self.gamma_f = self.options.gamma[f_index]
            self.rho_f = self.options.costwts[f_index]
        else:
            # TODO: figure out if you need something here
            pass

        # ----- Objective function Updates ----- #
        # first sum:
        first_term = self.gamma_f * gp.quicksum(self.p[u, v] * self.g[u, v] for u, v in self.g.keys())
        # second sum:
        second_term = self.rho_f * gp.quicksum(self.q * self.delta[u, v] * self.h[u, v] for u, v in self.h.keys())
        # set objective function
        self.model.setObjective(first_term - second_term, GRB.MAXIMIZE)

        self.model.update()
