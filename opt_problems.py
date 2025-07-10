from gurobipy import Model, GRB, quicksum
import gurobipy as gp
import networkx as nx
import pandas as pd

import parameters
import parameters as params
from random import random
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from network_functions import read_node_pos


# File for storing the classes used to implement the column generation


class MasterOptions:
    def __init__(self, nfreqs=3, freqwts=None, costwts=None, gamma=None, weight=None, headings=None,
                 max_len=None, length_weight=None, limit_dir=None):
        """
        Initialize MasterOptions with default or user-provided values.

        Parameters:
            nfreqs (int): Number of frequency levels. Default is 3.
            freqwts (list): Relative ridership coefficients for each frequency. Default is [0.5, 1.0, 1.5].
            costwts (list): Relative cost coefficients for each frequency. Default is [1.0, 1.5, 2.0].
            gamma (list): Relative coverage coefficients for each frequency. Default is [.10, .25, .60].
            weight (any): The key for the weight used for each link in the network
            headings (bool): Toggle the heading separators in the Subproblem
            max_len (int): Maximum length for a transit line
        """
        self.nfreqs = nfreqs
        self.freqwts = freqwts if freqwts is not None else [0.5, 1.0, 1.5]
        self.costwts = costwts if costwts is not None else [1.0, 1.5, 2.0]  # note: cost per length of frequency (rho)
        self.gamma = gamma if gamma is not None else [.10, .25, .60]
        self.weight = weight if weight is not None else 'Length '  # Default value is the one in the SF network
        self.headings = headings if headings is not None else True  # Toggle divider headings in Subproblem
        self.max_len = max_len if max_len is not None else 50
        self.length_weight = length_weight if length_weight is not None else 'Length '
        self.limit_dir = limit_dir if limit_dir is not None else False

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
        new_line.compute_length(self.network.network1, self.options.length_weight)  # Compute its length with the graph
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
        self.angle_hash = None  # If limit_dir is true, stores the angle of all the edges in the graph

        # Delta dictionary bc I don't want to deal with a DiGraph right now
        # could be adjusted to have unequal costs in each direction
        self.delta = gp.tupledict({(u, v): self.situation.network1[u][v][self.options.weight]
                                   for u, v in self.situation.network1.edges})

    def setup(self):
        # Debug
        if self.options.headings: print('\n|---------- Begin Setup ----------| \n')

        # ----- Decision variables ----- #
        # vars for which demand will be served and which edges will be included
        self.h = self.model.addVars(self.situation.network1.edges(), vtype=GRB.BINARY, name="h")
        self.model._h = self.h  # h duplicate for the lazy constraints
        self.g = self.model.addVars(self.situation.demand.keys(), vtype=GRB.BINARY, name="g")

        # vars for selecting a source and sink
        self.source = self.model.addVars(self.situation.network1.nodes(), vtype=GRB.BINARY, name="source")
        self.sink = self.model.addVars(self.situation.network1.nodes(), vtype=GRB.BINARY, name="sink")

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

        # One Source and One Sink Constraints: -> not in paper but in their code
        self.model.addConstr(gp.quicksum(self.source[n] for n in self.source) == 1, name="one_source")
        self.model.addConstr(gp.quicksum(self.sink[n] for n in self.sink) == 1, name="one_sink")

        # One flow conservation constraint to rule them all
        self.model.addConstrs(
            (
                gp.quicksum(self.h[i, j] for (i, j) in self.situation.network1.out_edges(n)) -
                gp.quicksum(self.h[i, j] for (i, j) in self.situation.network1.in_edges(n))
                ==
                self.source[n] - self.sink[n]
                for n in self.situation.network1.nodes()
            ),
            name="flow_conservation"
        )

        # Maximum line length constraint to save computational time
        # ----- Maximum line length constraint ----- #
        self.model.addConstr(
            gp.quicksum(self.delta[u, v] * self.h[u, v] for (u, v) in self.h.keys()) <= self.options.max_len,
            name="max_line_length"
        )

        if self.options.limit_dir:
            from numpy import arccos, dot
            from numpy.linalg import norm

            def determine_angle_difference(u, v, limit_angle=None, return_angle=False):
                """
                Determines the angle between two vectors u and v.
                Returns the angle if return_angle is True, otherwise compares to limit_angle.
                """
                theta = arccos(dot(u, v) / (norm(u) * norm(v)))
                return theta if return_angle else theta < limit_angle

            # Read node positions from file
            pos = read_node_pos(parameters.sf_node_file)

            # Find the OD pair with the highest dual variable
            max_od = self.p.idxmax()
            o, d = max_od
            main_dir = (pos[d][0] - pos[o][0], pos[d][1] - pos[o][1])

            # Compute angle between each edge's direction and the main direction
            self.angle_hash = {
                edge: determine_angle_difference(
                    (pos[edge[1]][0] - pos[edge[0]][0], pos[edge[1]][1] - pos[edge[0]][1]),
                    main_dir,
                    return_angle=True
                )
                for edge in self.situation.network1.edges()
            }

            # Add constraints to restrict edges that deviate too far from main direction
            limit_angle = self.options.limit_dir  # in radians
            for edge, angle in self.angle_hash.items():
                if angle > limit_angle:
                    self.model.addConstr(self.x[edge] == 0, name=f"dir_limit_{edge}")


        # Update model
        self.model.update()

    def optimize(self):
        if self.options.headings: print('\n|---------- Begin Optimization ----------| \n')

        self.model.Params.LazyConstraints = 1
        self.model.optimize(subtour_elimination_callback)

        if self.options.headings: print('\n|---------- Results ----------| \n')

        status = self.model.Status

        if status in [GRB.INFEASIBLE, GRB.UNBOUNDED, GRB.INF_OR_UNBD]:
            print("Subproblem infeasible or unbounded.")
            return None

        if status != GRB.OPTIMAL:
            print(f"Optimization ended with status {status}")
            return None

        obj_val = self.model.ObjVal
        print(f"Subproblem optimal objective: {obj_val:.4f}")

        if obj_val < 0:
            print("Warning: Optimal solution has negative objective value.")

        # Extract source and sink nodes
        source_node = None
        sink_node = None

        for n in self.situation.network1.nodes():
            if self.source[n].X > 0.5:
                source_node = n
            if self.sink[n].X > 0.5:
                sink_node = n

        if source_node is None or sink_node is None:
            print("Error: No valid source/sink found.")
            return None

        # Build the subgraph of selected h edges
        selected_edges = [(u, v) for (u, v) in self.h.keys() if self.h[u, v].X > 0.5]
        G_selected = nx.DiGraph()
        G_selected.add_edges_from(selected_edges)

        # Find a path from source to sink
        try:
            G_undirected = G_selected.to_undirected()
            if source_node in G_undirected and sink_node in G_undirected:
                stops = nx.shortest_path(G_undirected, source=source_node, target=sink_node)
            else:
                print(f"Warning: Source {source_node} or sink {sink_node} not in selected subgraph.")
                return None
        except nx.NetworkXNoPath:
            print("No path from source to sink in selected edges.")
            return None

        # Create the TransitLine object
        line = TransitLine(origin=source_node, destination=sink_node, stops=stops)
        line.compute_length(self.situation.network1, weight=self.options.weight)

        return line

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
        if f_index is not None:
            self.gamma_f = self.options.gamma[f_index]
            self.rho_f = self.options.costwts[f_index]

        # ----- Objective function Updates ----- #
        # first sum:
        first_term = self.gamma_f * gp.quicksum(self.p[u, v] * self.g[u, v] for u, v in self.g.keys())
        # second sum:
        second_term = self.rho_f * gp.quicksum(self.q * self.delta[u, v] * self.h[u, v] for u, v in self.h.keys())
        # set objective function
        self.model.setObjective(first_term - second_term, GRB.MAXIMIZE)

        self.model.update()


# Functions defined outside the scope of one class
def subtour_elimination_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        # Get variable values at the current solution
        h_values = model.cbGetSolution(model._h)

        # Build subgraph of selected edges
        selected_edges = [(u, v) for (u, v), val in h_values.items() if val > 0.5]
        G_selected = nx.DiGraph()
        G_selected.add_edges_from(selected_edges)

        # Get all weakly connected components (works for directed graphs)
        for component in nx.weakly_connected_components(G_selected):
            if len(component) == 1:
                continue  # skip singletons

            subgraph_edges = [(u, v) for (u, v) in selected_edges if u in component and v in component]

            # Lazy constraint: sum of h[u,v] inside component ≤ |S| - 1
            lhs = gp.quicksum(model._h[u, v] for u, v in subgraph_edges)
            model.cbLazy(lhs <= len(component) - 1)


def get_active_transit_lines(master):
    """
    Returns a list of (line_index, TransitLine object, freq_index) tuples for lines selected in the master solution.
    """
    active_lines = []

    for i, line in enumerate(master.linelist):
        # Look for variables like x[i,j] — i = line index, j = frequency index
        for v in master.model.getVars():
            if v.varName.startswith(f'x[{i},') and v.X > 0.5:
                freq_index = int(v.varName.split(',')[1][:-1])  # strip the closing bracket
                active_lines.append((i, line, freq_index))
                break  # one freq per line

    return active_lines


def make_transit_line_summary(active_lines, master):
    """
    Returns a pandas DataFrame summarizing each active transit line.

    Parameters:
    - active_lines: output from get_active_transit_lines()
    - master: the MasterProblem object (used for options)

    Returns:
    - DataFrame with line #, OD, length, cost, and freq weight
    """
    rows = []
    for line_index, line, freq_index in active_lines:
        freq_weight = master.options.costwts[freq_index]
        cost = line.length * freq_weight
        rows.append({
            "Line #": line_index,
            "OD": f"{line.od[0]}–{line.od[1]}",
            "Length": round(line.length, 2),
            "Cost": round(cost, 2),
            "Freq Weight": freq_weight
        })

    df = pd.DataFrame(rows)
    return df


def plot_transit_lines(graph, active_lines, pos=None, alpha=0.3, title="Transit Lines", plot_all=False, save_it=None):
    """
    Plots a list of transit lines on the graph using networkx and matplotlib.

    Parameters:
    - graph: NetworkX DiGraph
    - active_lines: output from get_active_transit_lines()
    - pos: node positions (dict)
    - alpha: transparency of the lines
    - title: plot title
    - plot_all: boolean to turn on plotting all transit lines on their own network
    - save_it: string to give a file path to save the images to
    """
    if pos is None:
        pos = nx.spring_layout(graph)

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_edges(graph, pos, edge_color='lightgray', width=1)
    nx.draw_networkx_nodes(graph, pos, node_size=40, node_color='gray')

    if plot_all:
        colors = {}

    for i, line, _ in active_lines:
        path_edges = list(zip(line.stops[:-1], line.stops[1:]))
        color = [random() for _ in range(3)]  # random RGB

        if plot_all:
            colors[str(i)] = color

        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, width=2.5,
                               edge_color=[color], alpha=alpha, arrows=False)
        nx.draw_networkx_nodes(graph, pos, nodelist=line.stops, node_color=[color], node_size=60, alpha=alpha)

    legend_handles = []
    for i, line, _ in active_lines:
        # after you generate the color:
        legend_handles.append(Patch(facecolor=colors[str(i)], edgecolor='black',
                                    label=f'Line {i + 1}: {line.od[0]}→{line.od[1]}'))

    plt.legend(handles=legend_handles, loc='lower left', fontsize='small', frameon=True)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

    if save_it is not None:
        plt.savefig(save_it + 'all lines.png')

    plt.show()

    if plot_all:
        for i, line, _ in active_lines:
            plt.figure(figsize=(10, 8))
            # Base network
            nx.draw_networkx_edges(graph, pos, edge_color='lightgray', width=1, arrows=False)
            nx.draw_networkx_nodes(graph, pos, node_size=40, node_color='gray')

            # Plot current transit line
            edges = list(zip(line.stops[:-1], line.stops[1:]))
            nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color=colors[str(i)], width=2.5)
            nx.draw_networkx_nodes(graph, pos, nodelist=line.stops, node_color=colors[str(i)], node_size=70)
            nx.draw_networkx_labels(graph, pos, labels={n: n for n in line.stops}, font_size=8)

            plt.title(f"Transit Line {i + 1}: {line.od[0]} → {line.od[1]}")
            plt.axis('off')
            plt.tight_layout()

            if save_it is not None:
                plt.savefig(save_it + f'Transit Line {i + 1}.png')

            plt.show()
