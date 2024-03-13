#VERSION 1.1

import numpy as np
import matplotlib.pyplot as plt

phantom_size = [256,256]

#constants as input for function
maximum_height = 256
n_voids = 100
r_c = 1
r_max = 0.5
rng = np.random.default_rng(seed=35)

#old
'''def euc_dist(current_x, current_y, current_z,other_points):
    full_list = [] #we keep this list to calculate overlap with voids later
    closest_void = float('inf')
    for j in other_points:
        x_j, y_j, z_j,r_j = j[0],j[1],j[2],j[3]
        current_dist = np.linalg.norm([current_x,current_y,current_z]-[x_j,y_j,z_j])
        full_list.append(current_dist)
        #term 2 in equation (1), page 4. Keep min(d(a,b) - r_j) over all j's
        if (current_dist - r_j) < closest_void:
            closest_void = (current_dist - r_j)
    return full_list,closest_void'''
def euc_dist(current_x, current_y, current_z, other_points):
    full_list = []
    closest_void = float('inf')
    #print('other_points',len(other_points))
    if len(other_points) == 0:
        return full_list, closest_void
    for j in other_points:
        x_j, y_j, z_j,r_j = j[0],j[1],j[2],j[3]
        current_dist = np.linalg.norm(np.array([current_x,current_y,current_z])-np.array([x_j,y_j,z_j]))
        full_list.append(current_dist)
        #term 2 in equation (1), page 4. Keep min(d(a,b) - r_j) over all j's
        if (current_dist - r_j) < closest_void:
            closest_void = (current_dist - r_j)
    return full_list,closest_void

def add_trial_points(kill_list, phantoms, trials):
    counter = 0
    while counter < len(kill_list):
        x_i = rng.uniform(-1,1)
        #pick RANDOM y
        y_i = rng.uniform(-1,1)
        #pick RANDOM z
        z_i = rng.uniform(0,1)
        #for i,point in enumerate(kill_list):
        _, closest_void = euc_dist(x_i,y_i,z_i,phantoms) #[3] trial_list
        #pick radius r_i from equation (1) in paper
        if ((x_i**2 + y_i**2)**0.5) > r_c:
            continue
        r_i = min((1-(x_i**2 + y_i**2)**0.5),closest_void,r_max)
        if r_i < 0:
            continue
        if abs(z_i) <= maximum_height:
            #add to trial_list
            trials[kill_list[counter]] = np.array([x_i, y_i, z_i, r_i])
            counter += 1
        else:
            continue
    #print('trials in add_method',len(trials))
    return trials

def update_trial_list(trials):
  #take the largest void (sort by r_i, 4th column)
  trials[trials[:,3].argsort()]
  #when there are multiple, take RANDOM
  largest_index = np.argmax(trials[:,3])
  largest_void = np.asarray(trials[largest_index])
  trials = np.delete(trials,largest_index,axis=0) #remove from trial points
  kills = kill_points(largest_void, trials)
  return trials, largest_void, kills

def kill_points(largest, trials):
  largest_radius = largest[3]
  distances,_ = euc_dist(largest[0],largest[1],largest[2],trials)
  kill_list = []
  for count,j in enumerate(trials):
    distance = distances[count]
    trial_radius = j[3]
    if distance < (largest_radius + trial_radius):
      kill_list.append(count)
    else:
      continue
  return kill_list

#fill all pixels outside cylinder with 1
def sphereDrawer(positions, radii, size, z, resolution):

    radiiSeen = radii ** 2 - (positions[:, 2] - z) ** 2
    radiiSeen[radiiSeen < 0] = 0
    radiiSeen = np.sqrt(radiiSeen)

    image = np.zeros((resolution, resolution))
    for x in range(resolution):
        X = size * (x + 0.5) / resolution - 0.5 * size

        for y in range(resolution):
            Y = size * (y + 0.5) / resolution - 0.5 * size

            if X**2 + Y**2 > 1:
                image[x,y] = 1
                continue

            dr = positions - np.array([X, Y, z])

            if np.sum(np.sum(dr * dr, axis=1) < radiiSeen**2) > 0:
                image[x, y] = 1

    plt.imshow(image, cmap="Greys")

    plt.show()

if __name__ == "__main__":
    phantom_matrix = []
    trial_list = np.zeros([n_voids,4],dtype=float) #x coord, y coord, z coord, radius
    kills = np.arange(0,100,1)
    for n in range(n_voids):
        trial_list = add_trial_points(kills, phantom_matrix, trial_list)
        trial_list, largest_void, kills = update_trial_list(trial_list)
        phantom_matrix.append(largest_void)
    phantom_matrix = np.asarray(phantom_matrix,dtype=float)
    print(phantom_matrix,'voids')
    sphereDrawer(phantom_matrix[:,0:3],phantom_matrix[:,3],size=3,z=1,resolution=maximum_height)