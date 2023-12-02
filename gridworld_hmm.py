import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


class Gridworld_HMM:
    def __init__(self, size, epsilon: float = 0, walls: bool = False):
        if walls:
            self.grid = np.zeros(size)
            for cell in walls:
                self.grid[cell] = 1 # is this assigning a value of 1 to all non-obstacle cells
        else:
            self.grid = np.random.randint(2, size=size) # generate numbers between 0 and 1

        self.epsilon = epsilon #
        self.trans = self.initT() # generate the transition matrix
        self.obs = self.initO() # generate the observation matrix

    def neighbors(self, cell):
        i, j = cell # 
        M, N = self.grid.shape
        adjacent = [
            (i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j - 1), 
            (i, j),(i, j + 1), (i + 1, j - 1), (i + 1, j), (i + 1, j + 1), # why do we check all 4
        ]
        neighbors = []
        for a in adjacent:
            if a[0] >= 0 and a[0] < M and a[1] >= 0 and a[1] < N and self.grid[a] == 0:
                neighbors.append(a)
        return neighbors

    """
    4.1 Transition and observation probabilities
    """

    def initT(self):
        """
        Create and return NxN transition matrix, where N = size of grid.
        """
        M, N = self.grid.shape
        T = np.zeros((M * N, M * N))
        # TODO:
        # implement the forward algorith 
        for i in range(M):
          for j in range(N):
            k = (i, j)
            arr_n = self.neighbors(k)
            count = 0
            #for cell_n in arr_n:
              #count += self.grid(cell_n) this just show you have to explore the code better 
            for cell_n in arr_n:
              T[cell_n[0]*N + cell_n[1],j + (N)*i] = 1/len(arr_n)

        return T

     
    def initO(self):
        """
        Create and return 16xN matrix of observation probabilities, where N = size of grid.
        """
        
        M, N = self.grid.shape
        O = np.zeros((16, M * N))
        # TODO:
        for i in range(M):
          for j in range(N):
            adjacent_cardinal = [
              (i, j + 1), (i + 1, j), (i, j - 1), (i - 1, j),
            ]

            str_obs = ""
            for a in adjacent_cardinal:
              #print(int(self.grid[a]))
              if a not in self.neighbors((i, j)):
                str_obs += "1"
              else:
                str_obs += "0"
            obs_curr = int(str_obs, 2)
            for val in range(16):
              d = 0
              for k in bin(obs_curr ^ val, )[2:]:
                d += int(k)
                #count()
              
              
              O[val, j + i * N] = (1-self.epsilon)**(4-d)*(self.epsilon)**(d)

        
        return O

    """
    4.2 Inference: Forward, backward, filtering, smoothing
    """
# read about how the algorithm was performed
    def forward(self, alpha: npt.ArrayLike, observation: int):
        """Perform one iteration of the forward algorithm.
        Args:
          alpha (np.ndarray): Current belief state.
          observation (int): Integer representation of bitstring observation.
        Returns:
          np.ndarray: Updated belief state.
        """
        # TODO:
        #res = np.linalg.eig(T)
       # eidx = np.argmax(res[0])
        #pf = res[1][:,eidx]
        #pi = pf/sum(pf) 
        
        alpha_p = self.trans@alpha
        return np.multiply(alpha_p, self.obs[observation])
        

    def backward(self, beta: npt.ArrayLike, observation: int):
        """Perform one iteration of the backward algorithm.
        Args:
          beta (np.ndarray): Current "message" of probabilities.
          observation (int): Integer representation of bitstring observation.
        Returns:
          np.ndarray: Updated message.
        """
        # 
        # TODO:
        beta_t_f = np.multiply(beta, self.obs[observation])
        return self.trans.T@beta_t_f
        

    def filtering(self, init: npt.ArrayLike, observations: list[int]):
        """Perform filtering over all observations.
        Args:
          init (np.ndarray): Initial belief state.
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Estimated belief state at each timestep.
        """
        # TODO:
        Alpha = np.zeros((self.grid.size, len(observations)))
        alpha_t_curr = init
        for i in range(len(observations)):
          alpha_t_plus = self.forward(alpha_t_curr, observations[i])
          Alpha[:, i] = alpha_t_plus/sum(alpha_t_plus)
          alpha_t_curr = alpha_t_plus

        return Alpha

    def smoothing(self, init: npt.ArrayLike, observations: list[int]):
        """Perform smoothing over all observations.
        Args:
          init (np.ndarray): Initial belief state.
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Smoothed belief state at each timestep.
        """
        Alpha = self.filtering(init, observations)
        Beta = np.zeros((np.prod(self.grid.shape) , len(observations)))
        beta_curr = np.ones(len(init))
        
        for i in range(len(observations)-1, -1, -1):
          beta_t_minus = self.backward(beta_curr, observations[i])
          Beta[:, i] = beta_t_minus
          beta_curr = beta_t_minus/sum(beta_t_minus)

        return np.multiply(Alpha, Beta)



        

    """
    4.3 Localization error
    """

    def loc_error(self, beliefs: npt.ArrayLike, trajectory: list[int]):
        """Compute localization error at each timestep.
        Args:
          beliefs (np.ndarray): Belief state at each timestep.
          trajectory (list[int]): List of states visited.
        Returns:
          list[int]: Localization error at each timestep.
        """
        # TODO:
        tract_est = np.zeros(np.shape(beliefs)[1])
        error = np.zeros(np.shape(beliefs)[1])
        for i in range(np.shape(beliefs)[1]):
          tract_est[i] = np.argmax(beliefs[:,i])
          #print(tract_est[i])
          error[i] = abs(tract_est[i]//16 - (trajectory[i]//16)) + abs(tract_est[i]%16 - (trajectory[i]%16))
        return error

