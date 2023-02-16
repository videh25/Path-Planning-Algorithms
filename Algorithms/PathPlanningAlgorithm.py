from time import sleep
from Maps.map import Map
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np

class PathPlanningAlgorithm:
    def __init__(self, map: Map = None, map_path:str = None) -> None:
        """
        @brief Initialises the Algorithm
        @param map The map object to be acted upon
        @param map_path Algorithm uses this path to acces map if given; Gets priority over the map object provided.
        """
        self.map = map
        if map_path is not None:
            self.map = Map(from_path=True, path=map_path)

    def operate(self, visual: bool = False) -> List[Tuple[int, int]]:
        """
        @brief Runs the algorithm and returns the shortest path
        @returns A list of tuples(x, y), showing the shortest path
        
        """
        src_tup , tar_tup  = self.get_the_source_and_target()

        if self.map is not None:
            return self.run(src_tup, tar_tup, visual)
        
    def get_the_source_and_target(self) -> Tuple[List[int], List[int]]:
        """
        @brief Gets the source and target from mouse cursor
        @return source_tuple, target_tuple
        """
        source_tup = []
        target_tup = []
        cid_src = None
        cid_tar = []
        fig, ax = plt.subplots()
        self.map.show("This map", fig, ax, show_here=False)
        fig.suptitle("Click for Source")
        fig.canvas.draw()

        def onclick_source(event):
            if (self.map.array[int(np.round(event.ydata))][int(np.round(event.xdata))] != 255):
                return None
            source_tup.append(int(np.round(event.xdata)))
            source_tup.append(int(np.round(event.ydata)))
            print(f'Source: {source_tup}')
            ax.scatter(source_tup[0], source_tup[1], c="green")
            fig.suptitle("Click for Target")
            fig.canvas.draw()
            fig.canvas.mpl_disconnect(cid_src)
            cid_tar.append(fig.canvas.mpl_connect('button_press_event', onclick_target))
            
        def onclick_target(event):
            if (self.map.array[int(np.round(event.ydata))][int(np.round(event.xdata))] != 255):
                return None
            target_tup.append(int(np.round(event.xdata)))
            target_tup.append(int(np.round(event.ydata)))
            print(f'Target: {target_tup}')
            ax.scatter(target_tup[0], target_tup[1], c="yellow")
            fig.canvas.draw()
            fig.canvas.mpl_disconnect(cid_tar[0])
            sleep(1)
            plt.close(fig)

        cid_src = fig.canvas.mpl_connect('button_press_event', onclick_source)
        plt.show()

        return source_tup, target_tup

    def run(self, source_point: Tuple[int, int], target_point: Tuple[int, int], visual: bool = False) -> Tuple[int, List[int]]:
        """
        @brief To be defined in the subclasses
        """
        pass

if __name__ == "__main__":
    pass
    PPA = PathPlanningAlgorithm(map_path='Maps/demo_maps/600x600_A.png')
    PPA.operate()