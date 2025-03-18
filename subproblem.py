# ---------- Import Statements ---------- #
import networkx as nx
import numpy as np
import parameters as p
from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt


# ---------- Graph Construction ---------- #
def construct_graph(nstns, edgecost, direction, maxdist, delta, rmp_dists=None):
    """
    Constructs an acyclic directed graph with filtered edges based on distance and direction.

    Parameters:
        nstns (int): Number of stations (nodes).
        edgecost (function): Function to compute edge cost (distance) between two nodes.
        direction (list): Desired direction vector for filtering edges.
        maxdist (float): Maximum allowable distance for an edge.
        delta (float): Tolerance for directional similarity (1 - delta in the paper).
        rmp_dists (dict, optional): Pre-computed edge distances.

    Returns:
        graph (nx.DiGraph): The constructed directed acyclic graph.
        dists (dict): Dictionary of edge distances.
        outneighbors (dict): Outgoing neighbors for each node.
        inneighbors (dict): Incoming neighbors for each node.
    """
    # Initialize graph, distance dictionary, and neighbors
    graph = nx.DiGraph()
    dists = rmp_dists if rmp_dists else {}
    outneighbors = {u: [] for u in range(1, nstns + 1)}
    inneighbors = {u: [] for u in range(1, nstns + 1)}

    # Add nodes
    graph.add_nodes_from(range(1, nstns + 1))

    # Iterate over all node pairs
    for u in range(1, nstns + 1):
        for v in range(u + 1, nstns + 1):
            d = edgecost(u, v)  # Edge cost
            b = np.array([v - u, 1])  # Replace with actual direction vector logic
            sim = np.dot(direction, b) / (np.linalg.norm(direction) * np.linalg.norm(b))

            if d < maxdist and (1 - abs(sim)) <= delta:
                if sim >= 0:  # Positive direction
                    dists[(u, v)] = d
                    outneighbors[u].append(v)
                    inneighbors[v].append(u)
                    graph.add_edge(u, v)
                else:  # Reverse direction
                    dists[(v, u)] = d
                    outneighbors[v].append(u)
                    inneighbors[u].append(v)
                    graph.add_edge(v, u)

    # Ensure graph is acyclic
    assert nx.is_directed_acyclic_graph(graph), "Graph contains cycles!"

    return graph, dists, outneighbors, inneighbors


def filter_graph(input_graph, edgecost, direction, maxdist=p.max_edge_count, delta=p.delta):
    """
    Filters an existing NetworkX graph to create an acyclic directed graph.

    Parameters:
        input_graph (nx.DiGraph): Pre-existing NetworkX graph to filter.
        edgecost (function): Function to compute edge cost (distance) for an edge.
        direction (list): Desired direction vector for filtering edges.
        maxdist (float): Maximum allowable distance for an edge.
        delta (float): Tolerance for directional similarity (1 - delta in the paper).

    Returns:
        filtered_graph (nx.DiGraph): The filtered directed acyclic graph.
        dists (dict): Dictionary of edge distances.
        outneighbors (dict): Outgoing neighbors for each node.
        inneighbors (dict): Incoming neighbors for each node.
    """
    # Initialize the filtered graph and auxiliary data structures
    filtered_graph = nx.DiGraph()
    dists = {}
    outneighbors = {u: [] for u in input_graph.nodes()}
    inneighbors = {u: [] for u in input_graph.nodes()}

    # Iterate over edges in the input graph
    for u, v in input_graph.edges():
        d = edgecost(u, v, input_graph)  # Compute edge cost (e.g., distance)
        b = input_graph[u][v]['b']
        sim = np.dot(direction, b) / (np.linalg.norm(direction) * np.linalg.norm(b))

        # Apply distance and direction filters
        if d < maxdist and (1 - abs(sim)) <= delta:
            if sim >= 0:  # Positive direction
                dists[(u, v)] = d
                outneighbors[u].append(v)
                inneighbors[v].append(u)
                filtered_graph.add_edge(u, v)
            else:  # Reverse direction
                dists[(v, u)] = d
                outneighbors[v].append(u)
                inneighbors[u].append(v)
                filtered_graph.add_edge(v, u)

    # Ensure the filtered graph is acyclic
    assert nx.is_directed_acyclic_graph(filtered_graph), "Filtered graph contains cycles!"

    return filtered_graph, dists, outneighbors, inneighbors


# My edge cost function:
# TODO: Dynamically program this so that we look up costs, not calculate them
def edgecost(u, v, inputgraph=None, costparam=p.costparam):
    if inputgraph is None:
        return abs(u - v)
    else:
        return inputgraph[u][v][costparam]


# --------- Gurobi Setup ---------- #
def setup_gurobi_model(graph, nlegs=1, max_edges=p.max_edge_count):
    """
    Sets up a Gurobi optimization model for the subproblem.

    Parameters:
        graph (nx.DiGraph): The transit network as a NetworkX graph.
        nlegs (int): Number of legs for commute (1 for direct-route, 2 for single-transfer).
        max_edges (int): Maximum number of edges allowed in a path.

    Returns:
        model (gurobipy.Model): The Gurobi optimization model.
        variables (dict): Dictionary of decision variables.
    """
    # Initialize Gurobi model
    model = Model("Subproblem")

    # Create variables
    edg = model.addVars(graph.edges(), vtype=GRB.BINARY, name="edg")  # h_ij in the paper
    src = model.addVars(graph.nodes(), vtype=GRB.BINARY, name="src")  # only one sink
    snk = model.addVars(graph.nodes(), vtype=GRB.BINARY, name="snk")  # only one sink
    ingraph = model.addVars(graph.nodes(), vtype=GRB.BINARY, name="ingraph")  # g_ij
    srv = model.addVars(graph.nodes(), graph.nodes(), vtype=GRB.BINARY, name="srv")

    # Update model
    model.update()

    # Editor's note: only used if transfers are included
    # -> this is to say, I'm not using it, at least for now
    srv2 = (
        model.addVars(graph.nodes(), graph.nodes(), vtype=GRB.BINARY, name="srv2")
        if nlegs == 2
        else None
    )

    # Update model
    model.update()

    # TODO: Add the actual objective function
    model.setObjective(
        quicksum(srv[u, v] for u, v in graph.edges()),
        GRB.MAXIMIZE
    )

    # Update model
    model.update()

    # Constraints
    # 1. Single source and sink
    model.addConstr(quicksum(src[u] for u in graph.nodes()) == 1, "single_source")
    model.addConstr(quicksum(snk[u] for u in graph.nodes()) == 1, "single_sink")

    # Update model
    model.update()

    # 2. Flow conservation
    for node in graph.nodes():
        model.addConstr(
            quicksum(edg[u, node] for u in graph.predecessors(node))
            - quicksum(edg[node, v] for v in graph.successors(node))
            == src[node] - snk[node],
            f"flow_{node}"
        )

    # Update model
    model.update()

    # 3. Path length restriction
    model.addConstr(edg.sum() <= max_edges, "max_edges")

    # Update model
    model.update()

    # 4. Connectivity: ingraph variables for used nodes
    for node in graph.nodes():
        model.addConstr(
            ingraph[node] <= quicksum(edg[node, v] for v in graph.successors(node)),
            f"connectivity_{node}_out"
        )
        model.addConstr(
            ingraph[node] <= quicksum(edg[u, node] for u in graph.predecessors(node)),
            f"connectivity_{node}_in"
        )

    # Update model
    model.update()

    # 5. Commute service constraints
    for u, v in graph.edges():
        model.addConstr(srv[u, v] <= edg[u, v], f"srv_direct_{u}_{v}")

    # Update model
    model.update()

    if nlegs == 2:
        # Constraints for srv2 (single-transfer model)
        for u, v in graph.nodes():
            model.addConstr(
                srv2[u, v]
                <= quicksum(edg[u, w] for w in graph.successors(u))
                * quicksum(edg[w, v] for w in graph.predecessors(v)),
                f"srv_transfer_{u}_{v}"
            )

        # Update model
        model.update()

    return model, {"edg": edg, "src": src, "snk": snk, "ingraph": ingraph, "srv": srv, "srv2": srv2}


# testruns
fg, dists, out_n, in_n = filter_graph(p.G, edgecost=edgecost, direction=[0, 1])
# nx.draw_networkx(fg, pos=p.sf_positions)
# plt.show()
model, dict1 = setup_gurobi_model(fg)
print(model)
for key in dict1:
    print(key, dict1[key])

