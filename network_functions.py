import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import parameters as p
from gurobipy import tupledict


'''
Status of the networks: 

* The only usable network right now is Souix Falls 
    - it is the one we have been testing for much of this project so that should be okay
    - I (Geoff) think that it should be okay for the completion of this project 
* If you want to use other networks, you will need to do some direction vector logic. 
    - usually you can just use the positions of the nodes for this 
    - store your direction as a vector called 'b' in the graph to avoid errors  
'''


# ---------- Load network data ---------- #
def read_node_pos_geojson(file_name, verbose=False):
    gdf = gpd.GeoDataFrame.from_file(file_name)

    if verbose:
        print(gdf)

    positions = {row['id']: (row.geometry.y, row.geometry.x) for _, row in gdf.iterrows()}

    if verbose:
        print(positions)

    return positions


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


def create_disneytown(dg=False, save_file=None):
    G = nx.DiGraph()

    nodes2, edges = read_network_info(p.ana_net_file, 8, tilde='~',
                                      init_node='init_node', term_node='term_node',
                                      is_anaheim=True)

    nodes = list(read_node_pos_geojson(p.ana_node_file).keys())

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    if dg:
        positions = read_node_pos_geojson(p.ana_node_file)
        plt.figure(figsize=p.fig_size)
        nx.draw(G, pos=positions, with_labels=True, node_size=200, node_color="lightblue")
        if save_file:
            plt.savefig(save_file, dpi=p.dpi)
            plt.close()
        else:
            plt.show()

    return G


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
                    demand_data[origin, dest] = value

    return demand_data
