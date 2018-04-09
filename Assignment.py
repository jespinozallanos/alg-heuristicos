import numpy as np
import matplotlib.pyplot as plt
import re 
from scipy.spatial import distance
import random



def read_kfile(fname):
    with open(fname, 'r') as kfile: # opening the file in the read mode
        pd = [] # empty list
        for data in kfile: # Iterating over file
            if re.findall(r'\d+.',data): # Looking for numbers starting with a dot 
               v = data.strip().split(' ') # splitting the data with a space
               pd.append(v) # Appending the data in an empty list pd
    pd=pd[1:1001] # It depicts the range of colours i.e.
    return pd


size = 1000
permutation = np.arange(size)
        
        

def plot_colours(colours, permutation):    # Here the colour is pd and pemutation is size 
	assert len(colours) == len(permutation) # Here the length of colour should be equal to length of permutation which we are getting from pd=pd[1:101]
	ratio = 100 # ratio of line height/width, e.g. colour lines will have height 10 and width 1
	img = np.zeros((ratio, len(colours), 3))
	for i in range(0, len(colours)):
		img[:, i, :] = colours[permutation[i]]
	fig, axes = plt.subplots(1, figsize=(8,4)) # figsize=(width,height) handles window dimensions
	axes.imshow(img, interpolation='nearest')
	axes.axis('off')
	plt.show()


def eucl_dist(fname): # Function to find Euclidean distances.
    a = 0 # Counter
    cal_eu_dist = [] # An empty list
    for i in range(0,len(fname)-1): # loop to run for 100 times
        dist = distance.euclidean(fname[a],fname[a+1]) # Calculating the euclidean distance
        cal_eu_dist.append(dist) # appending the calculated euclidean distance in an empty list.
        a += 1 # Incrementing the counter
    return cal_eu_dist # Returning the list containing calculated euclidean distances.
        
        
def min_sum_dist(cal_eu_dist): # function finding the minimum amongst the calculated euclidean distance
    return sum(cal_eu_dist)  # returning the list containing the sum of euclidean distance
    

# Generating a fixed number of solutions at random through Random Search

def random_search(fname):
    Min = len(fname) # Assigning the length of fname as a value to a variable Min
    my_lst = []
    my_sol = []
    sum_best_list = [] # list to append the list of minimum values after every iterations i.e. sum_list
    for i in range(20):
        sum_list = [] # List to append the minimum values after every iterations for line graph.
        for i in range(10000):
            random.shuffle(fname) # Generating Random Solutions
            eu_dist = eucl_dist(fname) # Calculating euclidean distance
            sum_rs_eu = sum(eu_dist) # Calculating sum of euclidean distance
            if sum_rs_eu < Min: #  comparing if sum of euclidean distances is less than Min
                Min = sum_rs_eu # assigning 'Min' the value of a particular value of sum less than Min
                min_sol = fname # Assigning 'min_sol' its corresponding solution i.e. colours        
            sum_list.append(Min) # Appending the minimum values after every iterations for line graph
        my_lst.append(Min) # Appending minimum values i.e. at last it will be 20.
        my_sol.append(min_sol) # Appending solution corresponding to minimum value.
        min_my_lst = min(my_lst) # Calculating minimum out of 20 minimum values.
        my_lst_index = my_lst.index(min_my_lst) # Finding index corresponding to the minimum value.
        sum_best_list.append(sum_list) # Appending the list in a list.
        ran_line = sum_best_list[my_lst_index] # finding list of minimum values after every iterations at same index as of minimum value.
    avg_ran_best_sol = np.mean(my_lst) # finding average of euclidean distance or fitness
    std_ran_best_sol = np.std(my_lst) # Finding standard deviation of euclidean distance or fitness.
    plot_colours(min_sol, permutation) # Ploting the colours for Random search best solution
    return min_my_lst,my_lst, avg_ran_best_sol, std_ran_best_sol,ran_line # Returning minimum value, my_lst for boxplot, avg and std and ran_line for line graph.

    

def perturbation(sol): # Move Operator, Double swap
    j1 = random.randint(0, len(sol)-1)
    j2 = random.randint(0, len(sol)-1)
    j3 = random.randint(0, len(sol)-1)
    j4 = random.randint(0, len(sol)-1)
    if (j1 == j2):
        j2 = random.randint(0, len(sol)-1)
    else:
        pass
    if (j3 == j4 or j3 == j1 or j3 == j2):
        j3 = random.randint(0, len(sol)-1)   
    else:
        pass
    if (j4 == j1 or j4 == j2):
        j4 = random.randint(0, len(sol)-1)   
    else:
        pass
    sol[j1], sol[j2] = sol[j2], sol[j1] # Swapping happenning. 
    sol[j3], sol[j4] = sol[j4], sol[j3]
    return sol 




def inverting(sol): # Move Operator, In this two elements swap and the other elements between them they invert.
    best = sol[:]
    best_size = len(best)
    i1,i2 = 0, 0
    while (i1 == i2): 
        i1, i2 = random.randrange(0, best_size), random.randrange(0, best_size)
    if i2 < i1:
        i1, i2 = i2, i1
    i2 = i2+1
    inverting_section = list(reversed(best[i1:i2]))
    best[i1:i2] = inverting_section
    return best


def two_inverting(sol): # Move Operator, Here it is done with 4 indexes.
    best = sol[:]
    best_size = len(best)
    i1,i2,i3,i4 = 0, 0, 0, 0
    if(i1 == i2): 
        i1, i2 = random.randrange(0,(best_size)/2 ), random.randrange(0,(best_size)/2)
    if(i2 < i1):
       i1, i2 = i2, i1
       i2 = i2+1
    inverting_section = list(reversed(best[i1:i2]))
    if(i3 == i4): 
        i3, i4 = random.randrange(((best_size)/2)+1,best_size), random.randrange(((best_size)/2)+1,best_size)
    if(i4 < i3):
       i3, i4 = i4, i3
       i4 = i4+1
    inverting_section1 = list(reversed(best[i3:i4]))
    best[i1:i2] = inverting_section
    best[i3:i4] = inverting_section1
    return best



def box_plot(random,hill,local,genetic): # Function to plot the box i.e. boxplot
    box_list = [random,hill,local,genetic]
    #new_box_list = ['(1)-Random Search','(2)-Hill Climbing','(3)-Iterated local Search','(4)-Genetic Algorithm']
    plt.boxplot(box_list)
    #.legend(new_box_list, fontsize=7, loc = 'upper right')
    plt.xticks([1,2,3,4],['Random','Hill','Iterated','Genetic'])
    #plt.xticks(rotation=90)
    return plt.show()

def line_plot(random,hill,local,genetic): # Function to plot the line i.e. Line Graph
    lineplot = [random,hill,local,genetic]
    new_lineplot = ['Random Search','Hill Climbing','Iterated local Search','Genetic Algorithm']
    nested_list = [list(zip(*[(ia+1,b) for ia,b in enumerate(a)])) for a in lineplot]
    for j in nested_list:
        plt.plot(*j) 
    plt.ylabel("Best value")
    plt.xlabel("Number of Iterations")
    plt.legend(new_lineplot, fontsize=7, loc = 'upper right')   
    return plt.show()




def hill_climbing(fname):
    best_list = [] # List used for boxplot
    sol_list = []
    best_hill_list = []
    for i in range(20):
        hill_list = []
        random.shuffle(fname) # Generating Random Solution
        ran_eu_dist = eucl_dist(fname) # Calculating Eucledian distance of randomly generated solution
        #print(len(ran_eu_dist))
        sum_ran_eu_dist = sum(ran_eu_dist) # Calculating Sum of Eucledian distance of randomly generated solution
        #print(sum_ran_eu_dist)
        d = fname # Assigning the file to a variable called 'd'
        for v in range(10000):  # Loop for number of iterations       
            d1= inverting(d) # Using inverting function
            hill_eu_dist = eucl_dist(d1) # finding euclidean distance
            
            sum_hill_eu_dist = sum(hill_eu_dist) # Finf=ding sum of euclidean distance
            
            if sum_hill_eu_dist < sum_ran_eu_dist: #Checking if sum of euclidean distance after swapping the index is less than the sum of euclidean distance of randomly generated solution
                best = sum_hill_eu_dist# Which ever is best assigning its value to 'best' variable
                sol = d1
            else:
                best = sum_ran_eu_dist # Which ever is best assigning its value to 'best' variable
                sol = d 
                
            d = sol # Assigning the solution corresponding to best value back to 'd' so that the value improves
            sum_ran_eu_dist = best
            hill_list.append(best) # Appending the best value after every iteration for line graph
        best_hill_list.append(hill_list) # Appending the list in a list for line graph
        best_list.append(best) # Appending 20 best values.
        sol_list.append(sol) # Appending solutions corresponding to 20 best values.
    best_value = min(best_list) # Finding minimum out of 20 best values.
    best_index = best_list.index(best_value) # Finding index of the minimum value.
    best_sol = sol_list[best_index] #solution on a similar index of minimum value.
    best_iter = best_hill_list[best_index] # finding values at the same index for line graph.
    avg_best = np.mean(best_list) # Calculating Avg
    std_best = np.std(best_list) # Calculating Std
    plot_colours(best_sol,permutation) # Ploting the colours for Hill Climbing best solution
    return best_value,best_list,avg_best,std_best,best_iter # returning best value, best_list for box plot, avg, std and best_iter for line graph




def iterated_local_search(fname):
    best_value_list = [] # List used for boxplot
    best_value_sol =[]
    best_local_list = [] # List processed for line graph
    for i in range(20): # loop to run 20 times
        random.shuffle(fname) # Generating random solution
        local_ran_eu_dist = eucl_dist(fname) # Calculating euclidean distance
        sum_local_ran_eu_dist = sum(local_ran_eu_dist) # Calculating sum of euclidean distance
        e = fname
        for i in range(10000):    # loop for iterations
            invert_sol = inverting(e) # Inverting function or move operator
            local_eu_dist = eucl_dist(invert_sol) # Calculating euclidean distance after swapping
            sum_local_eu_dist = sum(local_eu_dist) # Calculating sum of euclidean distance after swapping
            if sum_local_eu_dist < sum_local_ran_eu_dist: # Conditional statement
                best = sum_local_eu_dist # Assigning lowest value to a variable named as 'best'
                sol = invert_sol # corresponding solution of best
            else:
                best = sum_local_ran_eu_dist # Assigning lowest value to a variable named as 'best'
                sol = e # corresponding solution of best
            e = sol # reassigning the best sol after every iteration to 'e' so that it improves
            sum_local_ran_eu_dist = best
            
        two_invert_sol1 = two_inverting(sol) # Another Move operator.
            
        local_list = [] # List to have best values after every iterations
        
        for k in range(10000): # Loop for iterations
            
            invert_sol2 = two_inverting(two_invert_sol1) # Move operator
            
            local1_eu_dist = eucl_dist(invert_sol2) # Calculating euclidean distance after swapping
            sum_local1_eu_dist = sum(local1_eu_dist) # Calculating sum of euclidean distance after swapping
            
            if sum_local1_eu_dist < best: # Conditional statement
                new_best = sum_local1_eu_dist # Assigning lowest value to a variable named as 'best'
                sol2 = invert_sol2 # Corresponding solution
            else:
                new_best = best # Assigning lowest value to a variable named as 'best'
                sol2 = sol # Corresponding solution
            local_list.append(new_best) # Appending best values after every iterations
            two_invert_sol1 = sol2 # Assigning best solution to move operator to improve
            best = new_best # Assigning new best value to best value variable.
        best_local_list.append(local_list) # Appending list in a list for line graph
        best_value_list.append(new_best) # Appending list for box plot. 
        best_value_sol.append(sol2) # appending corresponding solutions
        
    
    min_best_value = min(best_value_list) # Finding minimum out of best_value_list
    min_index = best_value_list.index(min_best_value) # Finding index corresponding to minimum out of best_value_list
    min_best_sol = best_value_sol[min_index] # Fetching the value with the index similar to minimum value index
    best_iter1 = best_local_list[min_index] # Value in this variable used for line graph
    
    # Calculating Average and Standard Deviation
    
    avg_best_value_list = np.mean(best_value_list) # Calculating Average
    std_best_value_list = np.std(best_value_list) # Calculating Std
    plot_colours(min_best_sol,permutation) # Ploting the colours for Iterated Local Search best solution
    return min_best_value,best_value_list,avg_best_value_list,std_best_value_list,best_iter1 # returning minimum value, best_value_list for boxplot, avg, std and best_iter1 for line graph 



def genetic_algorithm(fname):
    new_gal = [] # List used for boxplot
    new_gal_sol = []
    new_gen_list = [] # List used for line graph
    for i in range(20):
        gal = [] # List containing the Generations
        sum_gen_eu_dist_list = []
        
        for i in range(10000):
            g = random.sample(fname,len(fname)) # Generating random solution or generations
            gen_eu_dist = eucl_dist(g) # Calculating euclidean distances
            sum_gen_eu_dist = sum(gen_eu_dist) # Calculating sum of euclidean distances
            gal.append(g) # appending the random solution or generations to an empty list 'gal[]'
            sum_gen_eu_dist_list.append(sum_gen_eu_dist) # appending sum of euclidean distances for random solution to an empty list 'sum_gen_eu_dist_list[]'
        
    # Randomly selecting four list out of those random solution in list 'gal[]'
    # Calculating euclidean distances and sum of euclidean distances for them
    
        gen_list = [] # List used for line graph
        
        for i in range(10000): # Loop for iterations
            i = random.randint(0,len(gal)-1) # Generating index randomly
            first_list = gal[i] # Assigning a list at the index 'i' to a variable
            first_list_eu_dist = eucl_dist(first_list) # Calculating Euclidean distances
            sum_first_list_eu_dist = sum(first_list_eu_dist) # Calculating Sum Of Euclidean distances.
            
            j = random.randint(0,len(gal)-1) # Generating index randomly
            second_list = gal[j]  # Assigning a list at the index 'j' to a variable
            second_list_eu_dist = eucl_dist(second_list) # Calculating Euclidean distances
            sum_second_list_eu_dist = sum(second_list_eu_dist) # Calculating Sum Of Euclidean distances.
            
            
            k = random.randint(0,len(gal)-1) # Generating index randomly
            third_list = gal[k] # Assigning a list at the index 'k' to a variable
            third_list_eu_dist = eucl_dist(third_list) # Calculating Euclidean distances
            sum_third_list_eu_dist = sum(third_list_eu_dist) # Calculating Sum Of Euclidean distances.
            
            
            l = random.randint(0,len(gal)-1) # Generating index randomly
            fourth_list = gal[l] # Assigning a list at the index 'l' to a variable
            fourth_list_eu_dist = eucl_dist(fourth_list) # Calculating Euclidean distances
            sum_fourth_list_eu_dist = sum(fourth_list_eu_dist) # Calculating Sum Of Euclidean distances.
            
            if sum_first_list_eu_dist < sum_second_list_eu_dist: # Finding best value among these two
                best3, cl = sum_first_list_eu_dist, gal[i] # Cl carries the corresponding solutions
            else:
                best3, cl = sum_second_list_eu_dist, gal[j] # Cl carries the corresponding solutions
             
            if sum_third_list_eu_dist < sum_fourth_list_eu_dist: # Finding best value among these two
                best4, cl1 = sum_third_list_eu_dist, gal[k] # Cl1 carries the corresponding solutions
            else:
                best4, cl1 = sum_fourth_list_eu_dist, gal[l] # Assigning the best value out of all the four sum of euclidean distances which was randomly generated
    
    
    # Applying Crossover which generated child1 and child2
    # Calculating euclidean distances and sum of euclidean distances after crossover on child1 and child2
            
            child1 = inverting(cl) # Using move operator on child1
            
            child2 = inverting(cl1) # Using move operator on child2
            
            child1_eu_dist = eucl_dist(child1) # Calculating Euclidean distances
            
            sum_child1_eu_dist = sum(child1_eu_dist) # Calculating Sum of Euclidean distances of child1
            
            child2_eu_dist = eucl_dist(child2) # Calculating Euclidean distances
            
            sum_child2_eu_dist = sum(child2_eu_dist) # Calculating Sum of Euclidean distances of child2
                
                
            max_sum_gen_eu_dist_list = max(sum_gen_eu_dist_list) # Finding worst out of sum of euclidean distances i.e. max
            z = sum_gen_eu_dist_list.index(max_sum_gen_eu_dist_list) # Storing the index of the maximum value in 'z'
            
            if max_sum_gen_eu_dist_list < sum_child1_eu_dist: # Comparing if max sum of euclidean distances is less than the sum of euclidean distances generated from child1
                pass
            else:
                gal[z] = child1 # Replacing the worst value in the randomly generated solution list with the best solution be it child1 
                sum_gen_eu_dist_list[z] = sum_child1_eu_dist # Replacing the better value in the corresponding list.
                
            max1_sum_gen_eu_dist_list = max(sum_gen_eu_dist_list) # Finding worst out of sum of euclidean distances i.e. max
            z1 = sum_gen_eu_dist_list.index(max1_sum_gen_eu_dist_list) # Storing the index of the maximum1 value in 'z'
            
            if max1_sum_gen_eu_dist_list < sum_child2_eu_dist:
                pass
            else:
                gal[z1] = child2 # Replacing the worst value in the randomly generated solution list with the best solution be it child2
                sum_gen_eu_dist_list[z1] = sum_child2_eu_dist
            new_min = min(sum_gen_eu_dist_list) # Finding the minimum value for every iteration
            gen_list.append(new_min) # Appending the minimum value after every iteration in a list for line graph
        min_sum_gen_eu_dist_list = min(sum_gen_eu_dist_list) # Finding the best minimum value.
        new_gal.append(min_sum_gen_eu_dist_list) # Appending the best minimum value in a list for boxplot
        new_gen_list.append(gen_list) # Appending the list in a list for line graph
        p = sum_gen_eu_dist_list.index(min_sum_gen_eu_dist_list) # Fetching index corresponding to minimum value
        new_gal_sol.append(gal[p]) # Appedning the corresponding solution in a list.
    min_new_gal = min(new_gal) # Finding the minimum out of the best minimum values.
    q = new_gal.index(min_new_gal) # Finding the corresponding index for min_new_gal
    gen_line = new_gen_list[q] # Getting solutions at the same index for line graph.
    
        
    # Calculating the Average and Standard deviation for 20 best solutions
        
    avg_gal_sol = np.mean(new_gal) # finding average of euclidean distance or fitness
    std_gal_sol = np.std(new_gal) # Finding standard deviation of euclidean distance or fitness.
    plot_colours(new_gal_sol[q],permutation) # Ploting the colours for Genetic Algorithm's best solution
    return min_new_gal,new_gal,avg_gal_sol,std_gal_sol,gen_line # returning minimum value, new_gal for boxplot, avg, std and gen_line for line graph.


def main():
    fname = read_kfile("colours.txt") # Opening the file through read_kfile function in read mode
    fname = [[float(y) for y in x] for x in fname] # Reading the file through iterating
    eucl = eucl_dist(fname) #Calculating euclidean distances
    sum_dist = min_sum_dist(eucl) # Calculating sum of euclidean distances
    
    ran_search_min,my_lst,avg_ran_best_sol, std_ran_best_sol,ran_line = random_search(fname) # Objects for Random search function
    print('The min value of Random Search is', ran_search_min)
    print('The average of 20 best minimum values is', avg_ran_best_sol)
    print('The standard deviation of 20 best minimum values is', std_ran_best_sol)
    
    
    sum_hill_eu_dist,best_list,avg_best,std_best,best_iter = hill_climbing(fname) # Objects for Hill Climbing function
    print('The min value of Hill Climbing Search is', sum_hill_eu_dist)
    print('The average of 20 best minimum values is', avg_best)
    print('The standard deviation of 20 best minimum values is', std_best)
    
    local_search,best_value_list,avg_best_value_list,std_best_value_list,best_local_list = iterated_local_search(fname) # Objects for Iterated Local search function
    print('The min value of Iterated Local Search is', local_search)
    print('The average of 20 best minimum values is', avg_best_value_list)
    print('The standard deviation of 20 best minimum values is', std_best_value_list)
    
    
    gen_alg,new_gal,avg_gal_sol, std_gal_sol,gen_line = genetic_algorithm(fname) # Objects for Genetic Algorith / Evolutionary Algorithm function
    print('The min value of Genetic Algo is', gen_alg)
    print('The average of 20 best minimum values is', avg_gal_sol)
    print('The standard deviation of 20 best minimum values is', std_gal_sol)
    
    box_plot(my_lst,best_list,best_value_list,new_gal) # passing values to boxplot function
    
    line_plot(ran_line,best_iter,best_local_list,gen_line) # # passing values to line graph function
    
    
main()

