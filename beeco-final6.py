"""
    Author: Yifei Chen
    Student ID: sc22yc
    
    Project:
        This is a model based on Bee coloney optimization algorithm which was inspired by some of the 
        behaivor of bees, and it is specialized to solve the TSP(Travelling salesman problem) without going
        back to the starting point.
        
        It haven't based on any project that already exist and it was just based on the idea
        of BCO so it might very different from others who want to solve the similar problem.
        
        We will compare the performance of this model with seval other Bio inspired method
        which focuse on the same problem
    
    Important Notice:
        1. It runs the map store in the txt file by default.
            If you want to get a random map just go to line 296 to change use the random coords
            that has been comment out. Looks like following
            #----------------------------------Change the Coords here !!!----------------------
            coords = get_city_map("my_array.txt")
            #coords = np.random.randint(max_distance, size=(number_of_cities, 2))
            #----------------------------------------------------------------------------------
            
        2. Sometimes this algorithm will fall into a local optimal solution,
            please restart the hive (rerun this program directly).
        3. Sometime when rerun the code it can not display the final fitness graph,
            please just clear the output, stop and restart the code.
"""

import numpy as np
import random
import matplotlib.pyplot as plt

class Bee:
    """
    Bee class defines for each bee path and the distance traveled for the bee
    """
    def __init__(self, curr_path):
        """
        calculate the distance of a given path of bee
        
        return the distance
        """
        self.curr_path = curr_path
        self.curr_path_distance = self.compute_path_distance(self.curr_path)
        
    def compute_path_distance(self, path):
        """
        calculate the distance of a given path of bee
        
        return the distance
        """
        distance = 0
        visited_cities = set()
        for i in range(len(path)-1):
            if path[i] == path[i+1] or path[i] in visited_cities:
                distance = float('inf')
                break
            visited_cities.add(path[i])
            distance += dist_matrix[path[i], path[i+1]]
        return distance

class BeeColony:
    """
    Bee Colony class defines for all the bee in the Colony include 
    """
    def __init__(self, num_cities, num_bees, num_epochs, elite_rate,
                 local_search_rate,mutate_rate, neighborhood_size,limitation,scout_bee_rate):
        """
        initialize the beeColoney
        
            num_cities : the number of city in the graph
            num_bees : number of bees total inside the coloney
            num_epochs : number of iteration that the coloney are going to go through
            elite_rate : the percentage of the bees that are going to survive to the next generation
            mutate_rate : the rate to mutate for the outlooker bee
            scout_bee_rate : the percentage of the scout bee
            local_search_rate : the posibility for the scout bee to do local search
            neighborhood_size : the range to rotate city for the scout bee when do the local search
            limitation : the number of chance that the scout bee are going to try
            valid_cities: index of all the citys in the graph
            population : all the bee objects
            best_path : The current best path amoung all the bees inside the population
            best_distance : The current best distance amoung all the bees inside the population
            average_distance : The current average distance amoung all the bees inside the population
        """
        self.num_cities = num_cities
        self.num_bees = num_bees
        self.num_epochs = num_epochs
        self.elite_rate = elite_rate
        self.mutate_rate = mutate_rate
        self.scout_bee_rate=scout_bee_rate
        self.local_search_rate = local_search_rate
        self.neighborhood_size = neighborhood_size
        self.limitation=limitation
        self.valid_cities = [i for i in range(num_cities)]
        curr_population = []
        for i in range(num_bees):
            bee_pos = np.random.permutation(self.valid_cities)
            bee=Bee(bee_pos)
            curr_population.append(bee)
        self.population = curr_population
        self.best_path = self.get_best_path()
        self.best_distance = self.compute_path_distance(self.best_path)
        self.average_distance= self.compute_path_distance(self.best_path)
        
    def get_best_path(self):
        #get current best path among all the bees 
        return min(self.population, key=lambda bee: bee.curr_path_distance).curr_path

    def compute_path_distance(self, path):
        #get the distance of one bee
        distance = 0
        visited_cities = set()
        for i in range(len(path)-1):
            if path[i] == path[i+1] or path[i] in visited_cities:
                distance = float('inf')
                break
            visited_cities.add(path[i])
            distance += dist_matrix[path[i], path[i+1]]
        return distance
    
    
    def get_average_distance(self):
        #get the average distance for all the bees
        fitness = []
        for bee in self.population:
            curr_bee_dis = self.compute_path_distance(bee.curr_path)
            fitness.append(curr_bee_dis)
        average_distance = sum(fitness)/len(fitness)
        return average_distance
    
    
    def send_employee_bees(self):
        #send the employee bees 
        num_elite = int(self.elite_rate * self.num_bees)
        elite_bees = sorted(self.population, key=lambda x: x.curr_path_distance)[:num_elite]
        for i in range(num_elite,self.num_bees):
            new_path = np.random.permutation(self.valid_cities)
            new_bee = Bee(new_path)
            new_distance = self.compute_path_distance(new_path)
            if new_distance < self.best_distance:
                self.best_distance = new_distance
                self.best_path = new_path
            self.population[i] = new_bee
            self.population[i].curr_path_distance = new_distance
        for i in range(num_elite):
            self.population[i] = elite_bees[i]
    
    def greedy_algorithm(self, path):
        #greedily find the best path in a given city list
        unvisited_cities = set(path[1:])
        curr_city = path[0]  # keep the start city
        new_path = [curr_city]
        
        while unvisited_cities:
            nearest_city = min(unvisited_cities, key=lambda next_city: dist_matrix[curr_city, next_city])
            new_path.append(nearest_city)
            unvisited_cities.remove(nearest_city)
            curr_city = nearest_city
        return np.array(new_path)
    
    def send_scout_bee(self):
        #send the scout bee
        max_trial=self.limitation * self.num_bees
        num_trial = 0
        while num_trial < max_trial:
            for i in range(self.num_bees):
                if random.uniform(0, 1) < self.scout_bee_rate:
                    bee_curr_path = self.population[i].curr_path
                    
                    start = np.random.randint(0, number_of_cities)
                    end = np.random.randint(start+1, number_of_cities + 1)
                    subpath  = bee_curr_path[start:end]
                    if random.uniform(0, 1) < self.local_search_rate:
                        roll_num=random.randint(0, self.neighborhood_size)
                        roll_num=int(random.uniform(-roll_num, roll_num))
                        subpath = np.roll(subpath, roll_num)
                    greedy_path = self.greedy_algorithm(subpath)
                    new_path=[]
                    new_path.extend(bee_curr_path[:start])
                    new_path.extend(greedy_path)
                    new_path.extend(bee_curr_path[end:])
                    new_distance = self.compute_path_distance(new_path)
                    if new_distance < self.population[i].curr_path_distance and new_distance!=0:
                        self.population[i].curr_path = new_path
                        self.population[i].curr_path_distance = new_distance
                        num_trial = 0  # Reset the counter for trials without improvement
                    else:
                        num_trial += 1

    def roulette_wheel_selector(self, probs):
        #do a roulette wheel selection on all the bees
        roulette_wheel = random.uniform(0, sum(probs))
        current = 0
        for i, p in enumerate(probs):
            current += p
            if roulette_wheel <= current:
                return i
    
    def crossover(self, path1, path2):
        #do a crossover on two given path to generate two new path
        path_length = len(path1)
        child1 = [-1] * path_length
        child2 = [-1] * path_length
        start = np.random.randint(0, path_length)
        end = path_length
        child1[start:end],child2[start:end] = path1[start:end],path2[start:end]
        unvisit_cities1 = [i for i in path2 if i not in child1]
        unvisit_cities2 = [i for i in path1 if i not in child2]
        for i in range(path_length):
            if child1[i] == -1:
                child1[i] = unvisit_cities1.pop(0)
            if child2[i] == -1:
                child2[i] = unvisit_cities2.pop(0)
        return child1, child2

    def mutate(self, path,mutate_rate):
        # random select one to two position in the path to mutate
        if random.uniform(0, 1) < mutate_rate:
            for i in range(random.randint(1,2)):
                j, k = random.sample(range(len(path)), 2)
                path[j], path[k] = path[k], path[j]
        return path

    def send_onlooker_bees(self):
        # send the onlooker bees to change path
        fitness = []
        for bee in self.population:
            curr_bee_dis = self.compute_path_distance(bee.curr_path)
            fitness.append(curr_bee_dis)
        num_elites = int(len(self.population) * self.elite_rate)
        elites = sorted(self.population, key=lambda x: x.curr_path_distance, reverse=False)[:num_elites]
        new_population = elites[:]
        #move elites bees to the next generation first before look by onlooker bees
        while len(new_population) < len(self.population):
            # select parents by roulette wheel selection
            parent1 = self.roulette_wheel_selector(fitness)
            parent2 = self.roulette_wheel_selector(fitness)
            # crossover
            child1, child2 = self.crossover(self.population[parent1].curr_path, self.population[parent2].curr_path)
            # mutation
            child1 = self.mutate(child1,self.mutate_rate)
            child2 = self.mutate(child2,self.mutate_rate)
            child1_fitness = self.compute_path_distance(child1)
            child2_fitness = self.compute_path_distance(child2)
            if child1_fitness < fitness[parent1]:
                new_population.append(Bee(child1))
            else:
                new_population.append(self.population[parent1])
            if child2_fitness < fitness[parent2]:
                new_population.append(Bee(child2))
            else:
                new_population.append(self.population[parent2])

        self.population = new_population



def get_city_map(filename):
    """
    get the city map from a file
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    print(len(lines))
    
    coords = []
    for y in range(len(lines)):
        curr_line=lines[y].split()
        for x in range(len(curr_line)):
            if curr_line[x] == '1':
               coords.append([y,x])
    return np.array(coords)




def generate_distance_matrix(n, coords):
    """
    get the city map from a file
    """
    distance_matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            distance = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix

#The number of cities in the graph
number_of_cities = 100
max_distance = 300  #you can change this if there is more cities add into the graph
#----------------------------------Change the Coords here !!!----------------------------------------------
#coords = get_city_map("my_array.txt")
coords = np.random.randint(max_distance, size=(number_of_cities, 2))
#----------------------------------------------------------------------------------------------------------
dist_matrix = generate_distance_matrix(number_of_cities, coords)

all_points = list(range(number_of_cities))
print(all_points)
print(coords)
bee_colony = BeeColony(num_cities=number_of_cities,
                       num_bees=200,
                       num_epochs=35,
                       elite_rate=0.05,
                       local_search_rate=0.5,
                       mutate_rate=0.5,
                       neighborhood_size=8,
                       limitation=2,
                       scout_bee_rate=0.1
                       )

plt.ion()
epochs=[]
bestdist=[]
averagedist = []

#start training
for i in range(bee_colony.num_epochs):
    bee_colony.send_employee_bees()
    bee_colony.send_scout_bee()
    bee_colony.send_onlooker_bees()
    bee_colony.best_path = bee_colony.get_best_path()
    bee_colony.best_distance = bee_colony.compute_path_distance(bee_colony.best_path)
    bee_colony.average_distance = bee_colony.get_average_distance()
    print(f"Epoch {i+1}: Best distance = {bee_colony.best_distance}, Average distance = {bee_colony.average_distance}")
    plt.clf()
    x = coords[:, 0]
    y = coords[:, 1]
    best_path=bee_colony.best_path
    for k in range(len(best_path) - 1):
        j = best_path[k]
        k = best_path[k + 1]
        dx, dy = coords[k] - coords[j]
        plt.arrow(coords[j, 0], coords[j, 1], dx, dy, head_width=0.2, length_includes_head=True,
                  color='red', linestyle='dashed', linewidth=2, shape='right')
    plt.scatter(x, y)
    for j, city in enumerate(coords):
        plt.text(city[0], city[1], str(j))
    plt.scatter(coords[best_path[0],0], coords[best_path[0],1], color='green', s=100, zorder=3)
    plt.scatter(coords[best_path[-1],0], coords[best_path[-1],1], color='orange', s=100, zorder=3)
    plt.grid()
    plt.xlabel('X')
    plt.ylabel('Y')
    print()
    #plt.title('BCO TSP Solution distance: '+str(bee_colony.best_distance))
    #plt.title('BCO Algorithm Solution')
    plt.savefig('./results/'+ str(i) +'.jpg')
    epochs.append(i)
    bestdist.append(bee_colony.best_distance)
    averagedist.append(bee_colony.average_distance)
    plt.pause(0.1)


plt.clf()
plt.plot(epochs, bestdist, linestyle = '-', label = 'Best fitness')
plt.plot(epochs,averagedist, linestyle = '-.', label = 'Average fitness')
plt.xlabel('epochs')
plt.ylabel('Distance')
plt.title('BCO TSP')
plt.legend()
plt.grid()
plt.show()
plt.savefig('./results/bee.jpg')




