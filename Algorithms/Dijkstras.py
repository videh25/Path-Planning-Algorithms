"""@package Dijkstras
@brief Contains implementation class of Dijkstras Algorithm along with methods to display it on a map
"""

from Algorithms.PathPlanningAlgorithm import PathPlanningAlgorithm
from Maps.map import Map
from typing import Tuple, List, Any
import heapq as hq
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

class DijkstrasAlgorithm(PathPlanningAlgorithm):
    def __init__(self, map: Map = None, map_path:str = None) -> None:
        """
        @breif Initialises Dijkstras object
        @param map Map(Coarse) object
        @map_path Path to load a map(Coarse) from
        """
        super().__init__(map, map_path)
        if self.map.map_type == "Fine":
            print("[ERROR] Dijkstras algorithm can be applied to a Coarse Map only(with lesser number of nodes)")
            exit()

    def run(self, source_point: Tuple[int, int], target_point: Tuple[int, int], visual: bool = False) -> Tuple[int, List[int]]:
        """
        @brief Runs the Dijkstras Algorithm
        @param source_point The starting node to find shortest path
        @param target_point The target node to find shortest path
        @return shortest_distance, shortest_path
        """
        def list_search(list_: List[List], index: int, search_item: Any):
            """
            @brief Searches for search_item at 'index' of each list in the list_
            """
            for ind, List_ in enumerate(list_):
                if List_[index] == search_item:
                    return ind
            return -1


        cost_matrix, node_array = self.map.graphify(propogation_pattern='*')
        source_node = node_array.index(tuple(source_point))
        target_node = node_array.index(tuple(target_point))
        print(f'Source: {source_node} -> Target: {target_node}')
        
        if visual:
            plt.ion()
            live_fig, live_ax = plt.subplots(figsize=(12, 10))
            self.map.show("Dijkstras Algorithm", live_fig, live_ax, False)
            live_ax.scatter(source_point[0], source_point[1], c="green")
            live_ax.scatter(target_point[0], target_point[1], c="yellow")
            for point in node_array:
                live_ax.scatter(point[0], point[1], c = 'black', alpha=0.2)
                   
        check_queue = []
        # Priority Queue Entries: (distance, node_number, via_node)
        for node in range(len(node_array)):
            if node == source_node:
                hq.heappush(check_queue,[0, node, -1])
                continue
            hq.heappush(check_queue,[np.inf, node, -1])

        just_poped = None
        jp_distance = None
        jp_via = None
        poped = []
        if visual:
            just_poped_marker = live_ax.scatter(source_point[0], source_point[1], c = '#1d7874', zorder = 100, label = "Current Node")
            checked_markers = live_ax.scatter([], [], c = '#679289', label = "Shortest Distance Calculated", zorder = 99)
            in_queue_markers = live_ax.scatter([], [], c = '#f4c095', label = "Non infinite Distance")
            final_path, = live_ax.plot([], [], color  = 'green', label='Path', zorder = 101)
        while just_poped != target_node:
            if just_poped is not None:
                poped.append([jp_distance, just_poped, jp_via])
                if visual:
                    if just_poped != source_node:
                        current_offsets = list(checked_markers.get_offsets())
                        current_offsets.append(list(node_array[just_poped]))
                        checked_markers.set_offsets(current_offsets)
            jp_distance, just_poped, jp_via = hq.heappop(check_queue)
            if jp_distance == np.inf:
                print("[INFO]: No path possible between the source and target.")
                exit()
            if visual:
                just_poped_marker.set_offsets(node_array[just_poped])
                live_fig.canvas.draw()
                sleep(0.)

            if just_poped == target_node:
                poped.append([jp_distance, just_poped, jp_via])
                break
            
            for node, cost in enumerate(cost_matrix[just_poped]):
                if cost == np.inf:
                    continue 
                poped_index = list_search(poped, 1, node)
                if poped_index != -1:
                    continue
                if node == just_poped:
                    continue
                
                index = list_search(check_queue, 1, node)
                if check_queue[index][0] >= jp_distance + cost:
                    check_queue[index][0] = jp_distance + cost
                    check_queue[index][2] = just_poped
                    hq.heapify(check_queue)
                    if visual:
                        current_offsets = list(in_queue_markers.get_offsets())
                        current_offsets.append(list(node_array[node]))
                        in_queue_markers.set_offsets(current_offsets)
                        live_ax.legend()
        
        shortest_path = []
        shortest_distance, this_node, prev_node =  poped[-1]
        shortest_path.append(this_node)
        
        while prev_node != -1:
            index = list_search(poped, 1, prev_node)
            if index == -1:
                break
            _, this_node, prev_node = poped[index]
            shortest_path.append(this_node)

        shortest_path = shortest_path[::-1]
        if visual:
            for node1 in shortest_path:
                x1, y1 = node_array[node1]
                x_data, y_data = list(final_path.get_data())
                x_data = list(x_data)
                y_data = list(y_data)
                x_data.append(x1)
                y_data.append(y1)
                final_path.set_data((x_data, y_data))
                live_fig.suptitle(f'Final Path [Distance: {shortest_distance}]')
                live_ax.legend()
                live_fig.canvas.draw()
                sleep(0.1)
            sleep(2)
            plt.close(live_fig)
            plt.ioff()
        return shortest_distance, shortest_path

    def visualise_graph(self):
        """
        To verify the graph formed from graphiphy method
        """
        cost_matrix, node_array = self.map.graphify(propogation_pattern='*')
        fig, ax = plt.subplots()
        self.map.show("Graph on Map", fig, ax, False)
        for node1 in range(len(node_array)):
            for node2, cost in enumerate(cost_matrix[node1]):
                if node1 == node2:
                    continue 
                if cost == np.inf:
                    continue 
                elif cost == 1:
                    x1, y1 = node_array[node1]
                    x2, y2 = node_array[node2]
                    ax.plot([x1, x2], [y1, y2], 'y-')
                elif cost == 2**0.5:
                    x1, y1 = node_array[node1]
                    x2, y2 = node_array[node2]
                    ax.plot([x1, x2], [y1, y2], 'r-')


        plt.show()

if __name__ == "__main__":
    PPA = DijkstrasAlgorithm(map_path='Maps/demo_maps/30x10_B.png')
    PPA.visualise_graph()
    PPA.operate(True)