import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import parameters as p
from gurobipy import tupledict
from random import random


# TODO: figure out which of these funcitons are important for outsiders to use
# \_> I don't really want to clutter it up if I don't need all these functions.
# ---------- Load network data ---------- #
def read_network_info(file_name, skip_rows=7, returndf=False, tilde='~ ',
                      init_node='Init node ', term_node='Term node ',
                      is_anaheim=False):
    df = pd.read_csv(file_name, sep='\t', skiprows=skip_rows)
    nodes = list(set(df[init_node].to_list() + df[term_node].to_list()))

    if is_anaheim:
        df['length'] = df['length'] / 5280

    edge_params = df.drop(columns=[tilde, init_node, term_node, ';']).to_dict(orient='records')
    edges = list(zip(df[init_node].values, df[term_node].values, edge_params))

    if returndf:
        return nodes, edges, df
    else:
        return nodes, edges


def create_test_net(dg=False):
    """
    Creates a 5x5 directed grid graph with bidirectional edges and a diagonal path.

    This function constructs a directed graph with nodes numbered from 1 to 25 arranged in a 5x5 grid.
    The graph includes bidirectional edges between adjacent nodes (right and down), as well as a
    one-directional diagonal path from node 1 to 25.

    Parameters:
    draw (bool): If True, the function will use matplotlib to draw and display the graph with nodes
                 positioned in a grid layout. Default is False.

    Returns:
    G (networkx.DiGraph): The created directed graph with the specified node and edge configurations.
    """

    G = nx.DiGraph()

    # Just trying to make a 5x5 grid
    grid_size = 5
    nodes = range(1, grid_size ** 2 + 1)
    G.add_nodes_from(nodes)

    # Right and Left edges
    G.add_edges_from([(i, i + 1, {'length': 1}) for i in range(1, grid_size ** 2) if i % grid_size != 0])  # Right
    G.add_edges_from([(i + 1, i, {'length': 1}) for i in range(1, grid_size ** 2) if i % grid_size != 0])  # Left

    # Down and Up edges
    G.add_edges_from([(i, i + grid_size, {'length': 1}) for i in range(1, grid_size ** 2 - grid_size + 1)])  # Down
    G.add_edges_from([(i + grid_size, i, {'length': 1}) for i in range(1, grid_size ** 2 - grid_size + 1)])  # Up
    # Add one-directional diagonal from node 1 -> 7 -> 13 -> 19 -> 25
    diagonal_nodes = [1, 7, 13, 19, 25]
    G.add_edges_from((diagonal_nodes[i], diagonal_nodes[i + 1], {'length': 1}) for i in range(len(diagonal_nodes) - 1))

    if dg:
        # Generate positions for a grid layout
        pos = {i: ((i - 1) % grid_size, (i - 1) // grid_size) for i in range(1, grid_size ** 2 + 1)}

        # Draw the graph with the calculated grid positions
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='#aaaaff', arrowsize=20)
        edge_labels = nx.get_edge_attributes(G, 'length')
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        plt.title('5x5 Grid Graph in Grid Layout')
        plt.show()

    return G


def create_test_net2(dg=False):
    """
    creates a test graph that hopefully proves our point.
    :param dg: draw the graph. Default is False.
    :return:
    """
    # Create the network
    G = nx.DiGraph()
    G.add_nodes_from(['1', '2', '3', '4', '5',
                      'A', 'B', 'C', 'D', 'E',
                      '21', '31', '41', '32',
                      '29', '38', '39', '49'])
    edges = [
        ('1', 'A', {'length': 10}),
        ('5', 'E', {'length': 10}),

        ('2', '21', {'length': 1}),
        ('21', '29', {'length': 8}),
        ('29', 'B', {'length': 1}),

        ('4', '41', {'length': 1}),
        ('41', '49', {'length': 8}),
        ('49', 'D', {'length': 1}),

        ('3', '31', {'length': 1}),
        ('31', '32', {'length': 1}),
        ('32', '38', {'length': 6}),
        ('38', '39', {'length': 1}),
        ('39', 'C', {'length': 1}),

        # diagonal ones
        ('1', '21', {'length': 2}),
        ('21', '32', {'length': 2}),
        ('2', '31', {'length': 2}),
        ('4', '31', {'length': 2}),
        ('5', '41', {'length': 2}),
        ('41', '32', {'length': 2}),

        ('49', 'E', {'length': 2}),
        ('38', '49', {'length': 2}),
        ('38', '29', {'length': 2}),
        ('29', 'A', {'length': 2}),
        ('39', 'B', {'length': 2}),
        ('39', 'D', {'length': 2}),

        ('5', '4', {'length': 1}),
        ('4', '3', {'length': 1}),
        ('3', '2', {'length': 1}),
        ('2', '1', {'length': 1}),

        ('E', 'D', {'length': 1}),
        ('D', 'C', {'length': 1}),
        ('C', 'B', {'length': 1}),
        ('B', 'A', {'length': 1}),

    ]

    edges_bidirectional = []
    for u, v, attr in edges:
        edges_bidirectional.append((u, v, attr))
        edges_bidirectional.append((v, u, attr.copy()))

    G.add_edges_from(edges_bidirectional)

    # add position data
    pos = {
        '1': (0, 0),
        '2': (0, 1),
        '3': (0, 2),
        '4': (0, 3),
        '5': (0, 4),

        'A': (10, 0),
        'B': (10, 1),
        'C': (10, 2),
        'D': (10, 3),
        'E': (10, 4),

        '21': (1, 1),
        '31': (1, 2),
        '32': (2, 2),
        '41': (1, 3),

        '29': (9, 1),
        '38': (8, 2),
        '39': (9, 2),
        '49': (9, 3),

    }

    if dg:
        plt.figure(figsize=(24, 12))
        nx.draw_networkx(G, pos, node_size=500, node_color='#aaaaff', arrowsize=20)
        plt.show()

    return G, pos


def create_test_net2_demand():
    demand_data = {
        ('1', 'A'): 500,
        ('2', 'B'): 500,
        ('3', 'C'): 500,
        ('4', 'D'): 500,
        ('5', 'E'): 500
    }

    return tupledict(demand_data)


def create_line_graph():
    """
    Creates a simple directed graph with 5 nodes (1 to 5) in a line.

    Returns:
        nx.DiGraph: A directed graph with edges forming a line (1 → 2 → 3 → 4 → 5).
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes (1 through 5)
    G.add_nodes_from(range(1, 6))

    # Add edges to form a line: 1 → 2 → 3 → 4 → 5
    G.add_edges_from([(1, 2, {'Length ':1}), (2, 3, {'Length ':1}), (3, 4, {'Length ':1}), (4, 5, {'Length ':1})])

    # Add a cost attribute to each edge for testing
    for u, v in G.edges():
        G[u][v]["cost"] = 1  # Assigning a constant cost of 1 for simplicity

    return G


def create_sioux_falls(draw_net=False):
    # initialize graph
    G = nx.DiGraph()

    # add nodes and edges
    nodes, edges = read_network_info(p.sf_net_file)

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Creating the direction vector of each edge
    npd = read_node_pos(p.sf_node_file)

    for u, v in G.edges():
        # Extract node coordinates
        xu, yu = npd[u]
        xv, yv = npd[v]

        # Assign the direction vector as an edge attribute
        G[u][v]["b"] = (xu - xv, yu - yv)

        # Optional: Uncomment the line below to print the direction vector for debugging
        # print(f'Direction of {u} → {v}: {(xu - xv, yu - yv)}')

    if draw_net:
        positions = read_node_pos(p.sf_node_file)
        nx.draw(G, with_labels=True, node_size=200,
                node_color='lightblue', pos=positions)
        plt.show()

    return G


def read_node_pos(file_name, verbose=False):
    df = pd.read_csv(file_name, sep='\t')
    if verbose:
        print(df)

    # might need to change 'Node' to something else if not Sioux falls
    nodes_position_dict = df.set_index('Node')[['X', 'Y']].apply(tuple, axis=1).to_dict()
    if verbose:
        print(nodes_position_dict)

    return nodes_position_dict


def parse_demand(file_path):
    demand_data = tupledict()

    with open(file_path, "r") as f:
        lines = f.readlines()

    origin = None
    for line in lines:
        line = line.strip()
        if line.startswith("Origin"):
            origin = int(line.split()[1])  # Extract the origin node
        elif ":" in line and origin is not None:
            parts = line.split(";")
            for part in parts:
                if ":" in part:
                    dest, value = part.split(":")
                    dest = int(dest.strip())
                    value = float(value.strip())
                    demand_data[(origin), (dest)] = value

    return demand_data


def plot_transit_lines(graph, lines, pos=None, title="Transit Lines", alpha=0.3, show_labels=True):
    """
    Plots a list of TransitLine objects on top of a base graph.

    Parameters:
    - graph: NetworkX DiGraph or Graph
    - lines: list of TransitLine objects (each has .stops and .od)
    - pos: dictionary of node positions; if None, spring_layout will be used
    - title: string for plot title
    - alpha: transparency for line edges (default 0.3)
    - show_labels: whether to draw node labels
    """
    if pos is None:
        pos = nx.spring_layout(graph)

    plt.figure(figsize=(12, 10))

    # Draw base graph
    nx.draw_networkx_edges(graph, pos, edge_color='lightgray', width=1, arrows=False)
    nx.draw_networkx_nodes(graph, pos, node_size=50, node_color='gray')

    # Draw each transit line
    for i, line in enumerate(lines):
        color = (random(), random(), random())  # random RGB
        line_edges = list(zip(line.stops[:-1], line.stops[1:]))

        nx.draw_networkx_edges(graph, pos, edgelist=line_edges, edge_color=[color], width=3, alpha=alpha)
        nx.draw_networkx_nodes(graph, pos, nodelist=line.stops, node_color=[color], node_size=70, alpha=alpha)

        if show_labels:
            nx.draw_networkx_labels(graph, pos, labels={n: n for n in line.stops}, font_size=8, font_color='black')

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
