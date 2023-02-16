"""@package maps
@brief Contains Map class for path planning project

@detail Map provides a class to maintain a map with methods to save it as an image, load it, generate a random map and also methods to provide a graph for Djikstras and A* to act upon
"""

import numpy as np
from typing import Tuple, List
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.ndimage import zoom

class Map:
    """
    @brief The Map class

    @detail Map class to generate test cases for the path planning algorithms.
    """
    def __init__(self, size: Tuple[int, int] = None, generate_random: bool=False,  from_path: bool = False, path:str = None) -> None:
        """
        @brief Normal Initialisation

        @param size Size of the map (x, y)
        @param generate_random Map is ranomly generated if set true
        """
        if size is not None:
            self.array = np.zeros((size[1], size[0]), dtype='uint8')
            self.size = size

        if generate_random:
            if np.min(size) > 20:
                self.generate_random_fine()
                self.map_type = 'Fine'
            else:
                self.map_type = 'Coarse'
                self.generate_random_coarse()
        
        if from_path:
            img = cv.imread(path)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            self.array = np.array(img, dtype='uint8')
            self.size = (np.shape(self.array)[1], np.shape(self.array)[0])
            if np.min(self.size) > 20:
                self.map_type = 'Fine'
            else:
                self.map_type = 'Coarse'
    
    def generate_random_coarse(self) -> None:        
        arr = np.random.uniform(size=(self.size[1]//2, self.size[0]//2))
        arr = zoom(arr, 2)
        arr[arr > 0.3] = 255
        self.array = np.array(arr, dtype='uint8')

    def generate_random_fine(self) -> None:        
        arr = np.random.uniform(size=(self.size[1]//16, self.size[0]//16))
        arr = zoom(arr, 16)
        arr[arr > 0.3] = 255
        self.array = np.array(arr, dtype='uint8')

    def save(self, path: str) -> None:
        """
        @brief Saves the current map as a jpg image

        @param path Path to save the image
        """
        cv.imwrite(path, self.array)

    def show(self, title: str, fig: Figure, ax: Axes, show_here:bool = True) -> None:
        ax.imshow(self.array, cmap = 'gray')
        fig.suptitle(title)
        if show_here:
            plt.show()

    def graphify(self, propogation_pattern: str, straight_cost: float = 1, diagonal_cost: float = 2**0.5) ->  Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        @brief Generates cost matrix for A* and Djikstras to work upon
        @detail Only for coarse maps
        @param propogation_pattern '+' or '*'
        @return cost_matrix, node_array
        """

        if self.map_type == 'Fine':
            print('[ERROR]: Graphying a fine map.')
            exit()
        total_nodes = np.sum(np.array(self.array==255, dtype='uint8'))
        cost_matrix = np.full(shape = (total_nodes, total_nodes), fill_value=np.inf)
        
        # Path cost to self is zero  
        for i in range(total_nodes):
            cost_matrix[i][i] = 0
        
        node_array = []
        """
        @brief Maps integers to individual nodes on the map
        @detail The index of a node is the vertex number and the tuple (i, j) shows the pixel position
        """
        print(self.size)
        for j in range(self.size[1]):
            for i in range(self.size[0]):
                if self.array[j][i] == 0:
                    continue
                else:
                    node_array.append((i,j))

        # plt.imshow(self.array, cmap='gray')
        # for ind, (i,j) in enumerate(node_array):
        #     plt.text(i, j, f'{ind}')
        # plt.show()

        for ind, (i,j) in enumerate(node_array):
            if (i+1, j) in node_array:
                cost_matrix[ind][node_array.index((i+1, j))] = straight_cost
                cost_matrix[node_array.index((i+1, j))][ind] = straight_cost
            
            if (i, j+1) in node_array:
                cost_matrix[ind][node_array.index((i, j+1))] = straight_cost
                cost_matrix[node_array.index((i, j+1))][ind] = straight_cost
        
        if propogation_pattern == '*':
            for ind, (i,j) in enumerate(node_array):
                if (i+1, j+1) in node_array:
                    cost_matrix[ind][node_array.index((i+1, j+1))] = diagonal_cost
                    cost_matrix[node_array.index((i+1, j+1))][ind] = diagonal_cost
                if (i-1, j+1) in node_array:
                    cost_matrix[ind][node_array.index((i-1, j+1))] = diagonal_cost
                    cost_matrix[node_array.index((i-1, j+1))][ind] = diagonal_cost


        return cost_matrix, node_array

    def figure_instance(self, x_width:int = None, y_width:int = None) -> Tuple[plt.figure, plt.axes]:
        """
        @brief Returns a figure with map drawn upon, for algorithms to draw upon
        @param x_width The x_width(in pixels) of the final figure requested
        @param y_width The y_width(in pixels) of the final figure requested

        @detail Any one of x_width or y_width must be provide. Uses x_width if both are given.
        """
        fig, ax = plt.subplots()
        if x_width is not None:
            scale_factor = x_width//self.size[0]
            # print(f'Scale Factor: {scale_factor}')
            dim = np.array(np.array(self.size)*scale_factor, dtype = 'int')
            # print(dim)
            return_image = cv.resize(self.array, dim, interpolation=cv.INTER_NEAREST)
            ax.imshow(return_image, cmap = 'gray')
            return fig, ax

        if y_width is not None:
            # print(self.size)
            scale_factor = y_width//self.size[1]
            # print(f'Scale Factor: {scale_factor}')
            dim = np.array(np.array(self.size)*scale_factor, dtype = 'int')
            # print(dim)
            return_image = cv.resize(self.array, dim, interpolation=cv.INTER_NEAREST)
            ax.imshow(return_image, cmap = 'gray')
            return fig, ax

if __name__ == '__main__':
    # map = Map(size = (600, 600), generate_random=True)
    # map.show("The Map")
    # map.save("600x600_B.png")


    # map2 = Map(from_path=True, path= "Maps/demo_maps/600x600_A.png")
    # map2.show("Map2")
    # # fig, ax = map2.figure_instance(y_width=1000)
    # plt.show()
    # map.graphify('+')

    path = r'Maps/demo_maps/600x600_A.png'
    map_ = Map(from_path=True, path=path)
    fig, ax = plt.subplots()
    map_.show("map_", fig, ax)