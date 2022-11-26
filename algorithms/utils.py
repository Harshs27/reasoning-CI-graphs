"""
Utility functions
"""
# Import libraries
import networkx as nx
from pyvis import network as net

# helper functions
def retrieve_graph(graph_edges):
    """ Read the graph edgelist from RC
    and convert it to a networkx graph.
    """
    graph_edges = graph_edges[2:-1].split("', '")
    edge_list = []
    for e in graph_edges:
        e = e.split(',')        
        edge_list.append(
            (e[0][1:], ''.join(e[1:-2]).lstrip(), 
            {"weight":float(e[-2]), 'color':e[-1][1:-1]})
        )
    G = nx.Graph()
    G.add_edges_from(edge_list)
    for n in G.nodes():
        G.nodes[n].update({'category':'unknown'})
    return G


def get_interactive_graph(G, title=''):
    Gv = net.Network(
        notebook=True, 
         height='750px', width='100%', 
    #     bgcolor='#222222', font_color='white',
        heading=title
    )
    Gv.from_nx(G.copy(), show_edge_weights=True, edge_weight_transf=(lambda x:x) )
    for e in Gv.edges:
        e['title'] = str(e['weight'])
        e['value'] = abs(e['weight'])
    for n in Gv.nodes:
#         print(n)
        n['title'] = 'Stage:'+n['category']
    Gv.show_buttons()
    return Gv