# ---------- Sub-problem Parameters ---------- #

# Gurobi setup parameters
max_edge_count = 30  # probably network dependent
delta = .3           # pretty high, should work for Sioux falls

# ---------- Network Creation Variables --------- #
# Network File Paths
ana_node_file = './resources/Anaheim/anaheim_nodes.geojson'
ana_net_file = './resources/Anaheim/Anaheim_net.txt'
sf_net_file = './resources/SiouxFalls/SiouxFalls_net.txt'
sf_node_file = './resources/SiouxFalls/SiouxFalls_node.txt'
sf_demand_file = './resources/SiouxFalls/SiouxFalls_trips.txt'

# figure size variable
fig_size = 24
dpi = 250

# ---------- Options for the master problem ---------- #
options_m = {
    "nfreqs": 3,
    "freqwts": [0.5, 1.0, 1.5],
    "costwts": [1.0, 1.5, 2.0]
}