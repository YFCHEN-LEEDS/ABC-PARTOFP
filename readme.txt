This is a model based on Bee coloney optimization algorithm which was inspired by some of the 
behaivor of bees, and it is specialized to solve the TSP(Travelling salesman problem) without going
back to the starting point.
        
It haven't based on any project that already exist and it was just based on the idea
of BCO so it might very different from others who want to solve the similar problem.
        
We will compare the performance of this model with seval other Bio inspired methodwhich focuse on the same problem


Important Notice:
	1. It runs the map store in the txt file by default.
		If you want to get a random map just go to line 316 to change use the random coords
		that has been comment out. Looks like following
	#----------------------------------Change the Coords here !!!----------------------------------------------
	coords = get_city_map("my_array.txt")
	#coords = np.random.randint(max_distance, size=(number_of_cities, 2))
	#----------------------------------------------------------------------------------------------------------

	2. Sometimes this algorithm will fall into a local optimal solution, please restart the hive (rerun this program directly).

	3. Sometime when rerun the code it can not display the final fitness graph, please just clear the output, stop and restart the code.


Environmental requirements:
python3.0+

Content:
just run the only .py file in a IDE


Uses:
the code used following library, you might want to install matplotlib if it has not been installed yet
import numpy
import random
import matplotlib.pyplot



Example Output:
Texts:
Epoch 1: Best distance = 889.468172088207, Average distance = 997.407801670432

Epoch 2: Best distance = 868.2909807959097, Average distance = 991.7055982560408

Epoch 3: Best distance = 867.576785896169, Average distance = 977.5372893992663

Epoch 4: Best distance = 867.5379572923723, Average distance = 996.8856341614313

Epoch 5: Best distance = 867.5379572923723, Average distance = 1032.8946646212983

Epoch 6: Best distance = 846.7285443542612, Average distance = 1010.1848400466321

Epoch 7: Best distance = 831.3175435886745, Average distance = 1028.8221640149086

Epoch 8: Best distance = 830.872005422321, Average distance = 991.1651945480477

Epoch 9: Best distance = 826.6199179019926, Average distance = 976.9847442817355

Epoch 10: Best distance = 826.6199179019926, Average distance = 987.5303064582198

Epoch 11: Best distance = 822.3790092587052, Average distance = 1001.4575090313554

Pictures:
	/results/0.jpg			This is the travel path of the first iteration
	/results/1.jpg			This is the travel path of the second iteration
	/results/bee.jpg			This is the fitness graph for the model



Statement of Contributions：

	All the code are written by Yifei Chen Student, ID: sc22yc
	The .txt file my_array.txt was provided by Group Member: Yujun Wang, Student ID:sc223yw
	If you got any error when implementing the code please contact: sc22yc@leeds.ac.uk







