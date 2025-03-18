# TNDP Translate
### *Revisitng "Data-Driven Transit Network Design at Scale"*

Hello all! This is code used by Geoffrey Vander Veen and the team at the University of California, Irvine, Institute of Transportation studies to run the math program outlined by Dimitris Bertsimas, Yee Sian Ng, and Julia Yan in their paper "Data-Driven Transit Network Design at Scale" (2020). 

We plan to use this as a complement to other research conducted by ITS Irvine to speedily and efficiently create bus networks as necessary in our exploration of transit network design. It is a direct translation of the work done by Bertisimas, et. al., but using modern Python implementation with Gurobi.  

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

You can clone the respository from the [github](https://github.com/veenboy1/tndpTranslate), or by running the following: 

```shell
git clone https://github.com/veenboy1/tndpTranslate
```

