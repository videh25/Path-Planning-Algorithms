"""@package RRT
@brief Contains implementation class of RRT Algorithm along with methods to display it on a map

"""

from time import sleep
from Algorithms.PathPlanningAlgorithm import PathPlanningAlgorithm
from Maps.map import Map
from typing import Tuple, List, Any
import heapq as hq
import numpy as np
import matplotlib.pyplot as plt

class RRTAlgorithm(PathPlanningAlgorithm):
    def __init__(self, map: Map = None, map_path:str = None, target_radius: float = 0.1, exploration_bias: float = 0., delta: float = 0.1, max_nodes: int = 0) -> None:
        """
        @brief Initialises RRT object
        @param map Map object
        @param map_path Path to load a map from
        @param target_radius Radius of vicinity around the target point
        @param exploration_bias Probablity that the target point will be sampled as a random point (A float in [0, 1])
        @param delta Distance by which the tree is expanded at each step
        @param max_nodes Max number of nodes the RRT will search till
        """
        super().__init__(map, map_path)
        self.target_radius = target_radius
        self.exploration_bias = exploration_bias
        self.delta = delta
        self.max_nodes = max_nodes

    def run(self, source_point: Tuple[int, int], target_point: Tuple[int, int], visual: bool = False, visualise_metrics: bool = False) -> Tuple[int, List[int]]:
        """
        @brief Runs the RRT Algorithm
        @param source_point The starting node to find shortest path
        @param target_point The target node to find shortest path
        @return discovered_path, path_distance
        """
        def list_search(list_: List[List], index: int, search_item: Any):
            """
            @brief Searches for search_item at 'index' of each list in the list_
            """
            for ind, List_ in enumerate(list_):
                if List_[index] == search_item:
                    return ind
            return -1
        
        def distance(point1: np.ndarray, point2: np.ndarray):
            """
            @brief To get the distances between two points
            """
            return np.linalg.norm(point1-point2)

        if visualise_metrics or visual:
            self.visualise_metrics(source_point, target_point)
        # Data Structure used:
        # Adjency list of heapified lists: The list index represents the node number
        # The entry in the list contain a list of all children
        # The inner list have a format: (distance_to_node_from_this_source, node)
        # And a point_array is maintained which contains a list of all coordinates stored as number

        adjency_lists: List[List] = []
        point_array = []

        source_point = np.array(source_point)
        target_point = np.array(target_point)

        # Adding Source Point to to point_array and adjency_lists
        point_array.append(np.array(source_point))
        adjency_lists.append([])

        if visual:
            plt.ion()
            live_fig, live_ax = plt.subplots(figsize=(12, 10))
            self.map.show("RRT Algorithm", live_fig, live_ax, False)
            plt.scatter(source_point[0], source_point[1], c= 'green', s = 15)
            plt.scatter(target_point[0], target_point[1], c= 'yellow', s = 15)
            target_circle = plt.Circle(target_point, self.target_radius, color='r', ls = '--', fill = False)
            live_ax.add_patch(target_circle)

            rand_point_marker = live_ax.scatter([], [], c = '#3caea3', label = "Random Point", s = 15)
            rand_point_line, = live_ax.plot([], [], color = '#3caea3', ls = '--')
            new_point_marker = live_ax.scatter([], [], c = '#20639b', label = "New Point", s = 15)
            tree_nodes = live_ax.scatter([], [], c = '#ed553b', label = "Tree Nodes", s = 7)
            live_fig.canvas.draw()


        new_node = None
        while (new_node is None) or (distance(point_array[new_node], target_point) > self.target_radius):
            if np.random.random() < self.exploration_bias:
                rand_point = target_point
            else:
                rand_point = np.random.random(2)
                rand_point[0] = rand_point[0]*self.map.size[0]
                rand_point[1] = rand_point[1]*self.map.size[1]
            if visual:
                rand_point_marker.set_offsets(rand_point)
                live_fig.canvas.draw()

            distance_array = []
            for node in range(len(point_array)):
                hq.heappush(distance_array, (distance(rand_point, point_array[node]), node))
            nearest_distance, nearest_node = hq.heappop(distance_array)
            del(distance_array)

            new_point = point_array[nearest_node] + (rand_point - point_array[nearest_node])*self.delta/nearest_distance
            new_pixel_x = int(min(np.round(new_point[0]), self.map.size[0] - 1))
            new_pixel_y = int(min(np.round(new_point[1]), self.map.size[1] - 1))
            if self.map.array[new_pixel_y][new_pixel_x] == 0:
                continue

            if visual:
                rand_point_line.set_data(tuple(np.column_stack([point_array[nearest_node], rand_point])))
                new_point_marker.set_offsets(new_point)
                live_fig.canvas.draw()

            # Append to point_array and to graph
            point_array.append(new_point)
            print(f'Number of Nodes: {len(point_array)}')
            if len(point_array) > self.max_nodes:
                print(f'[INFO]: Max number of nodes reached, could not find the path yet.')
                exit()
            adjency_lists.append([]) # Adjency list of the new_node
            new_node = len(point_array) - 1  # Index of new point in the point_array
            adjency_lists[new_node].append((nearest_distance, nearest_node))
            adjency_lists[nearest_node].append((nearest_distance, new_node))
            if visual:
                current_offsets = list(tree_nodes.get_offsets())
                current_offsets.append(new_point)
                tree_nodes.set_offsets(current_offsets)
                live_ax.plot([point_array[nearest_node][0], point_array[new_node][0]], [point_array[nearest_node][1], point_array[new_node][1]], c = "#ed553b", lw = "1")
                live_fig.canvas.draw()

            if distance(point_array[new_node], target_point) <= self.target_radius:
                nearest_distance = distance(point_array[new_node], target_point)
                point_array.append(target_point)
                adjency_lists.append([]) # Adjency list of the target_node
                target_node = len(point_array) - 1  # Index of target point in the point_array
                adjency_lists[target_node].append((nearest_distance, new_node))
                adjency_lists[new_node].append((nearest_distance, target_node))
                break
        
        current_node = None
        discovered_path = [target_point]
        path_distance = 0
        current_node = len(point_array) - 1 # Target Node
        while current_node != 0:
            parent_distance, parent_node = adjency_lists[current_node][0]
            discovered_path.append(point_array[parent_node])
            path_distance += parent_distance

            current_node = parent_node
        discovered_path = discovered_path[::-1]
        
        for ind, point_tuple in enumerate(discovered_path):
            print(f'Point: {point_tuple} | Point Number: {ind}')
        print(f'Path Distance: {path_distance}')
        if visual:
            for point1, point2 in zip(discovered_path[:-1], discovered_path[1:]):
                live_ax.plot([point1[0], point2[0]], [point1[1], point2[1]], c = "#3caea3", lw = "3", zorder = 102)
                live_fig.canvas.draw()
            sleep(5)
            plt.ioff()

        
        return discovered_path, parent_distance

    def visualise_metrics(self, source_point: Tuple[int, int], target_point: Tuple[int, int]):
        """
        @brief Plots the RRT Algorithm metrics on the map for clarity
        """
        fig, ax = plt.subplots()
        self.map.show("Metric Visualisation", fig, ax, False)
        plt.scatter(source_point[0], source_point[1], c= 'green')
        plt.scatter(target_point[0], target_point[1], c= 'yellow')
        
        target_circle = plt.Circle(target_point, self.target_radius, color='r', ls = '--', fill = False)
        ax.add_patch(target_circle)
        ax.plot([source_point[0], source_point[0]+self.delta], [source_point[1], source_point[1]], color = 'red')

        plt.show()

if __name__ == "__main__":
    # PPA = RRTAlgorithm(map_path='Maps/demo_maps/600x600_A.png', target_radius=10, delta  =10, exploration_bias=0.4, max_nodes=500)
    PPA = RRTAlgorithm(map_path='Maps/demo_maps/30x10_B.png', target_radius=0.5, delta  =0.5, exploration_bias=0.4, max_nodes=500)
    # PPA.visualise_metrics([50, 50], [90, 90])
    PPA.operate(visual = True)
