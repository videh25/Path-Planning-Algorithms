"""@package RRT_star
@brief Contains implementation class of RRT_star Algorithm along with methods to display it on a map

"""

from time import sleep
from Algorithms.PathPlanningAlgorithm import PathPlanningAlgorithm
from Maps.map import Map
from typing import Tuple, List, Any
import heapq as hq
import numpy as np
import matplotlib.pyplot as plt

class RRT_starAlgorithm(PathPlanningAlgorithm):
    def __init__(self, map: Map = None, map_path:str = None, target_radius: float = 0.1, exploration_bias: float = 0., delta: float = 0.1, path_planning_constant: float = 10., max_nodes: int = 0) -> None:
        """
        @brief Initialises RRT_star object
        @param map Map object
        @param map_path Path to load a map from
        @param target_radius Radius of vicinity around the target point
        @param exploration_bias Probablity that the target point will be sampled as a random point (A float in [0, 1])
        @param delta Distance by which the tree is expanded at each step
        @param max_nodes Max number of nodes the RRT_star will search till
        @param path_planning_constant A constant that is proportional to the rewiring radius of RRT_star
        @param rewiring_factor The factor by which the tree will rewire itself after the target is achieved 
        """
        super().__init__(map, map_path)
        self.target_radius = target_radius
        self.exploration_bias = exploration_bias
        self.delta = delta
        self.max_nodes = max_nodes
        self.path_planning_constant = path_planning_constant
    def run(self, source_point: Tuple[int, int], target_point: Tuple[int, int], visual: bool = False, visualise_metrics: bool = False) -> Tuple[int, List[int]]:
        """
        @brief Runs the RRT_star Algorithm
        @param source_point The starting node to find shortest path
        @param target_point The target node to find shortest path
        @return discovered_path, path_distance
        """
        def list_search(list_: List[List], index: int, search_item: Any) -> int:
            """
            @brief Searches for search_item at 'index' of each list in the list_
            """
            for ind, List_ in enumerate(list_):
                if List_[index] == search_item:
                    return ind
            return -1
        
        def distance(point1: np.ndarray, point2: np.ndarray) -> float:
            """
            @brief To get the distances between two points
            """
            return np.linalg.norm(point1-point2)

        def collision_free(point1: np.ndarray, point2: np.ndarray) -> bool:
            new_pixel1 = np.array([int(min(np.round(point1[0]), self.map.size[0] - 1)), int(min(np.round(point1[1]), self.map.size[1] - 1))])
            new_pixel2 = np.array([int(min(np.round(point2[0]), self.map.size[0] - 1)), int(min(np.round(point2[1]), self.map.size[1] - 1))])
            min_dim = min(self.map.size)
            if min_dim < 300:
                mid_pixels_count = int(distance(new_pixel1, new_pixel2)*500/min_dim)
            else:
                mid_pixels_count = int(distance(new_pixel1, new_pixel2))
            sample_pixels = np.array(np.unique(np.round(np.linspace(new_pixel1, new_pixel2, mid_pixels_count)), axis = 0), dtype='int')

            collision_free = True
            for pixel_x, pixel_y in sample_pixels:
                if self.map.array[pixel_y][pixel_x] == 0:
                    collision_free = False
                    break

            return collision_free

        def cost(node: int) -> float:
            if (node >= len(adjency_lists)):
                print("[ERROR]: node refernced does not have an adjency_list yet")
                exit()

            cost_ = 0
            while node != 0:
                distance, node = adjency_lists[node][0]
                cost_ += distance
            return cost_

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
            self.map.show("RRT_star Algorithm", live_fig, live_ax, False)
            plt.scatter(source_point[0], source_point[1], c= 'green', s = 15)
            plt.scatter(target_point[0], target_point[1], c= 'yellow', s = 15)
            target_circle = plt.Circle(target_point, self.target_radius, color='r', ls = '--', fill = False)
            live_ax.add_patch(target_circle)

            rand_point_marker = live_ax.scatter([], [], c = '#3caea3', label = "Random Point", s = 15)
            rand_point_line, = live_ax.plot([], [], color = '#3caea3', ls = '--')
            new_point_marker = live_ax.scatter([], [], c = '#20639b', label = "New Point", s = 15)
            tree_nodes = live_ax.scatter([], [], c = '#ed553b', label = "Tree Nodes", s = 7)
            tree_edge_lines: List[plt.Line2D] = [plt.Line2D([source_point[0], source_point[0]], [source_point[1], source_point[1]], c = "#ed553b", lw = "1")]
            live_ax.add_line(tree_edge_lines[0])
            vicinity_circle = plt.Circle(source_point, 0., color='green', ls = '--', fill = False)
            live_ax.add_patch(vicinity_circle)
            live_fig.canvas.draw()


        new_node = None
        while (new_node is None) or (distance(point_array[new_node], target_point) > self.target_radius):
            if np.random.random() < self.exploration_bias:
                rand_point = target_point
            else:
                rand_point = np.random.random(2)
                rand_point[0] = rand_point[0]*self.map.size[0]
                rand_point[1] = rand_point[1]*self.map.size[1]
                if self.map.array[int(min(np.round(rand_point[1]), self.map.size[1]-1))][int(min(np.round(rand_point[0]), self.map.size[0]-1))] == 0:
                    continue

            if visual:
                rand_point_marker.set_offsets(rand_point)
                live_fig.canvas.draw()

            distance_array = []
            for node in range(len(point_array)):
                hq.heappush(distance_array, (distance(rand_point, point_array[node]), node))
            nearest_distance, nearest_node = hq.heappop(distance_array)

            new_point = point_array[nearest_node] + (rand_point - point_array[nearest_node])*self.delta/nearest_distance
            new_pixel_x = int(min(np.round(new_point[0]), self.map.size[0] - 1))
            new_pixel_y = int(min(np.round(new_point[1]), self.map.size[1] - 1))
            
            if self.map.array[new_pixel_y][new_pixel_x] == 0:
                continue

            if visual:
                rand_point_line.set_data(tuple(np.column_stack([point_array[nearest_node], rand_point])))
                new_point_marker.set_offsets(new_point)
                live_fig.canvas.draw()
            
            # Add the point and its adjency list
            point_array.append(new_point) # Append new_point to the point_array
            adjency_lists.append([]) # Adjency list of the new_node
            new_node = len(point_array) - 1  # Index of new point in the point_array
            print(f'Number of Nodes: {len(point_array)}')
            if len(point_array) > self.max_nodes:
                print(f'[INFO]: Max number of nodes reached, could not find the path yet.')
                exit()

            ## RE-WIRING
            # List out the collision free points in vicinity separately
            rewiring_radius = self.path_planning_constant*(np.log(len(adjency_lists))/len(adjency_lists))**(1/len(np.shape(self.map.array)))
            rewiring_nodes = []
            if visual:
                vicinity_circle.set_center(new_point)
                vicinity_circle.set_radius(rewiring_radius)
                live_fig.canvas.draw()
            for node, point in enumerate(point_array):
                vicinity_distance = distance(point, point_array[new_node])
                if node == new_node:
                    break
                if vicinity_distance <= rewiring_radius and collision_free(point, new_point):
                    rewiring_nodes.append((node, vicinity_distance, cost(node)))

            # Connect to node that gives the lowest cost
            connecting_node: int = None
            new_node_cost = np.inf
            connecting_node_vicinity_distance: int = None
            for node, vicinity_distance, node_cost in rewiring_nodes:
                if node_cost + vicinity_distance < new_node_cost:
                    connecting_node = node
                    connecting_node_vicinity_distance = vicinity_distance
                    new_node_cost = node_cost + vicinity_distance
            if connecting_node is not None:
                adjency_lists[new_node].append((connecting_node_vicinity_distance, connecting_node))
                adjency_lists[connecting_node].append((connecting_node_vicinity_distance, new_node))
                if visual:
                    tree_edge_lines.append(plt.Line2D([point_array[connecting_node][0], point_array[new_node][0]], [point_array[connecting_node][1], point_array[new_node][1]], c = "#ed553b", lw = "1"))
                    live_ax.add_line(tree_edge_lines[new_node])
                    live_fig.canvas.draw()
            else:
                print("Vicinity list is empty")

            # Rewire the existing nodes if their cost is reduced
            for node, vicinity_distance, node_cost in rewiring_nodes:
                if new_node_cost + vicinity_distance < node_cost:
                    print('Rewiring!')
                    adjency_lists[new_node].append((vicinity_distance, node))
                    adjency_lists[node][0] = (vicinity_distance, new_node)
                    if visual:
                        tree_edge_lines[node].set_data([point_array[new_node][0], point_array[node][0]], [point_array[new_node][1], point_array[node][1]])
                        live_fig.canvas.draw()
            
            if visual:
                current_offsets = list(tree_nodes.get_offsets())
                current_offsets.append(new_point)
                tree_nodes.set_offsets(current_offsets)
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
        current_node = len(point_array) - 1 # Target Node
        path_distance = cost(current_node)
        while current_node != 0:
            parent_distance, parent_node = adjency_lists[current_node][0]
            discovered_path.append(point_array[parent_node])

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

        
        return discovered_path, path_distance

    def visualise_metrics(self, source_point: Tuple[int, int], target_point: Tuple[int, int]):
        """
        @brief Plots the RRT_star Algorithm metrics on the map for clarity
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
    # PPA = RRT_starAlgorithm(map_path='Maps/demo_maps/600x600_A.png', target_radius=10, delta  =10, exploration_bias=0.5, max_nodes=500, path_planning_constant=500)
    PPA = RRT_starAlgorithm(map_path='Maps/demo_maps/30x10_B.png', target_radius=0.5, delta  =0.5, exploration_bias=0.4, max_nodes=500, path_planning_constant=25)
    # PPA.visualise_metrics([50, 50], [90, 90])
    PPA.operate(True)
    # tup1, tup2 = PPA.get_the_source_and_target()
    # PPA.collision_free(tup1, tup2)
