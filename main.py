import numpy as np
import pylab as plt

city = np.array([[-1,  0, -1, -1, -1, -1, -1, -1],
                 [ 0, -1,  0, -1, -1,  0, -1, -1],
                 [-1, -1, -1,  -1,  0, -1, -1, -1],
                 [-1, -1, -1,  0, -1, -1,  0, -1],
                 [-1, -1,  0, -1, -1,  0, -1,  0],
                 [-1, -1, -1, -1, -1, -1,  0, -1],
                 [-1, -1,  0, -1, -1, -1, -1,  100],
                 [-1, -1, -1, -1, -1, -1, 0,  -1]])

Q = np.zeros((8,8))
learningRate = 0.01
reward = 0.1
def availPaths(index):
    return np.where(city[index,]>=0)[0]

def choosePathRandomly(indexs):
    return np.random.choice(indexs,1)[0]

def update(currentIndex, nextIndex):   
    max_indexes = np.where(Q[nextIndex,] == np.max(Q[nextIndex,]))[0]
    
    if max_indexes.shape[0]>1:
        max_indexes = choosePathRandomly(max_indexes)
    
    max_value = Q[nextIndex, max_indexes]
    
    Q[currentIndex, nextIndex] = city[currentIndex, nextIndex] + learningRate * max_value     

    #return max index value to save in score
    
    if (np.max(Q) > 0):
        return(np.sum(Q/np.max(Q)*100))
    else:
        return (0)

scores = []
#train
for _ in range(100):
    currentIndex = np.random.randint(0,7)
    indexes = availPaths(currentIndex)
    nextIndex = choosePathRandomly(indexes)
    score = update(currentIndex, nextIndex)
    scores.append(score)
    
#show score rate
plt.plot(scores)
plt.show()

#test
path = []
current_state = 0 
while current_state != 7:

    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[0]
    
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size = 1))
    else:
        next_step_index = int(next_step_index)
    
    path.append(next_step_index)
    current_state = next_step_index

print(path)

