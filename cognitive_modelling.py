import json
import numpy as np
import networkx as nx
from pyvis.network import Network

class CognitiveModelling(object):

    params_dict = {
        "edges": {
            "arrows": {
                    "to": {
                        "scaleFactor": 0.2}
                     },
                    "smooth": {
                        "enabled": True,
                        "type": "dynamic"
                    },
        },
        "nodes": {
            "scaling": {
                "label": {
                    "enabled": True,
                    "min": 20,
                    "max": 20,
                }
            },
            "value": 1
        },
        "physics": {
            "solver": "repulsion",
                "repulsion": {
                    "nodeDistance": 100,
                }
        }
    }

    def __init__(self, cognitive_matrix: np.array = None):
        self.cognitive_matrix: np.array = cognitive_matrix
        self.graph: nx.DiGraph = nx.DiGraph(self.cognitive_matrix, name="Cognitive Matrix")

    def show_graph_static(self):
        pos = nx.circular_layout(self.graph)
        nx.draw_networkx_nodes(self.graph, pos)
        edge_labels = nx.get_edge_attributes(self.graph, "weight")
        nx.draw_networkx_edges(self.graph,
                               pos=pos,
                               width=[abs(4 * elem[2])
                                      for elem in self.graph.edges(data="weight")],
                               edgelist=[(elem[0], elem[1]) for elem in list(self.graph.edges(data="weight"))],
                               arrows=True,
                               connectionstyle='arc3,rad=0.15',
                               alpha=0.75,
                               edge_color=[(0.3, 0.5, 1) if elem[2] > 0 else (1, 0.5, 0.3) for elem in
                                           self.graph.edges(data="weight")])
        nx.draw_networkx_labels(self.graph, pos)

        nx.draw_networkx_edge_labels(self.graph, pos,
                                     edge_labels=edge_labels,
                                     font_size=6,
                                     label_pos=0.15,
                                     verticalalignment="bottom",
                                     horizontalalignment="right",
                                     )

    def show_graph_html(self):
        json_object = json.dumps(CognitiveModelling.params_dict)
        nx.set_edge_attributes(self.graph,
                               dict([(elem[:2],
                                      "rgb(0, 100, 256)" if elem[2] > 0 else "rgb(256, 100, 0)")
                                     for elem in self.graph.edges(data="weight")]), "color")

        nx.set_edge_attributes(self.graph,
                               dict([(elem[:2],
                                      abs(3 * elem[2]))
                                     for elem in self.graph.edges(data="weight")]), "width")
        nx.set_node_attributes(self.graph, "circle", "shape")

        pyvis_network = Network(notebook=True)
        pyvis_network.from_nx(self.graph)
        pyvis_network.set_options(json_object)
        pyvis_network.show("network.html")

    def check_system_stability_structural(self):
        cycles = nx.recursive_simple_cycles(self.graph)
        odd_cycles = []
        for cycle in cycles:
            temp_cycle = cycle.copy()
            first_array = temp_cycle
            second_array = temp_cycle[1:]
            second_array.append(temp_cycle[0])
            curr_weights = self.cognitive_matrix[first_array, second_array]
            if np.prod(curr_weights) > 0:
                temp_cycle.append(temp_cycle[0])
                odd_cycles.append(np.array(temp_cycle)+1)
        return odd_cycles

    def check_system_stability_numerical(self):
        eigenvalues = np.linalg.eigvals(self.cognitive_matrix)
        max_eig = np.max([np.linalg.norm(val) for val in eigenvalues])
        return eigenvalues, max_eig

    def delete_connection(self, x, y):
        self.cognitive_matrix[x-1, y-1] = 0

    def impulse_modelling(self, t: int = 5, q: np.array = None):
        x_0 = np.zeros((self.cognitive_matrix.shape[0], 1))
        init_q = x_0.copy()
        x_list = [x_0, x_0]
        if q is None:
            q = init_q.copy()
            q[1] = 1
        else:
            q = np.array(q).reshape(-1, 1)
        save_q = q.copy()
        for _ in range(t):
            x_next = x_list[-1] + np.dot(self.cognitive_matrix, (x_list[-1] - x_list[-2])) + q
            x_list.append(x_next)
            q = init_q.copy()
        x_plot = np.array(x_list[1:])
        x_plot = x_plot.reshape(x_plot.shape[:2])
        return x_plot
