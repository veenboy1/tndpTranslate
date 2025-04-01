# TNDP Translate
### *Revisitng "Data-Driven Transit Network Design at Scale"*

Hello all! This is code used by Geoffrey Vander Veen and the team at the University of California, Irvine, Institute of Transportation studies to run the math program outlined by Dimitris Bertsimas, Yee Sian Ng, and Julia Yan in their paper "Data-Driven Transit Network Design at Scale" (2020). 

We plan to use this as a complement to other research conducted by ITS Irvine to speedily and efficiently create bus networks as necessary in our exploration of transit network design. It is a direct translation of the work done by Bertisimas, et al., but using modern Python implementation with Gurobi.  

## Installation 
### Versions
This code was tested most recently on Python 3.10.9 with Gurobi version 10.0.1.

### Dependencies

This project requires the following Python libraries:

- [`gurobipy`](https://www.gurobi.com/documentation/) – For optimization modeling with Gurobi
- [`networkx`](https://networkx.org/) – For creating and analyzing network graphs
- [`matplotlib`](https://matplotlib.org/) – For visualizing graphs and data
- [`pandas`](https://pandas.pydata.org/) – For handling and processing tabular data
- [`geopandas`](https://geopandas.org/) – *(Optional)* Used only in `read_node_pos_geojson()` for working with geospatial data

To install all required dependencies, run:  
```sh
pip install gurobipy networkx matplotlib pandas geopandas
```
Or, if you do not plan to use `read_node_pos_geojson`, you may also run: 
```sh
pip install gurobipy networkx matplotlib pandas 
```

### Cloning 

You can clone the respository from [our GitHub](https://github.com/veenboy1/tndpTranslate), or by running the following: 

```shell
git clone https://github.com/veenboy1/tndpTranslate
```

## Usage 

### Networks

This software wil require you to first define a transportation network with demand specified between every nodal pair. This will be in the form of a `Situation` object. To successfully create one, you must have a NetworkX `DiGraph` object that represents nodes and links in the network along with a gurobipy `tupledict` of demand, formatted `tupledict(origin, destination) = demand`. You can also add the demand one-by-one or edit existing demand with the `set_demand()` method. All of these will be used as inputs to the main script. 

*Note:* If you would like to display your network, you can give each node a location when creating the `DiGraph`. In a future version I will create a function to plot the map and the selected lines within it. 

### Run

Once the network is established, running it only requires changing the variables inside the `if` statement in the `main.py` file. It should be compatible with any combination of demand and network, as long as they are compatible. 

## Project Structure 

The code is straightforward. The file `main.py` contains everything necessary for a successful run of the problem. All classes associated with the optimization model are stored in the file `opt_problems`. `network_functions.py` contains functions that help with the creation of transportation networks, primarily from the `.tntp` file type; see [this GitHub](https://github.com/bstabler/TransportationNetworks) for details on the networks used in testing.

## Final notes!

### License  

This project is licensed under the MIT License, which means you are free to use, modify, and distribute this code. There are no restrictions on commercial or private use.

### Contact & Contributions 

If you have questions or comments, or especially possible improvements to the code, feel free to reach out to Geoffrey at `vandervg@uci.edu` or `gvanderveen04@gmail.com`; he checks both, but he might have graduated by the time you need to get in contact. 

Also, if you end up using it at all, please reach out! It would be a pleasure to hear that the hard work on this project is going farther than its home at the University of California, Irvine. _~_ GVV