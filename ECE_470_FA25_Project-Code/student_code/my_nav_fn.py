import numpy as np
import math

def compute_gradients(state, targets, obstacles, r):
    """
    Compute a velocity field (interpreted by the envelope as -∇U) for each robot.

    Inputs:
      state:    (n, 2) ndarray. Current xy positions for n robots.
      targets:  (n, 2) ndarray. Target xy positions for the same robots.
      obstacles: list of dicts: [{"center": (x, y), "radius": R}, ...]
      r:        float. REQUIRED minimum pairwise distance between any two robots.
                If any pair gets <= r, the simulation declares a collision & fails.
                Use this to shape your inter-robot repulsion term/barriers.

    Returns:
      grads:    (n, 2) ndarray. Velocity vectors for each robot at the current state.
                The envelope will cap the speed and step the system forward.
                IMPORTANT: Must be finite numbers (no NaN/Inf).
    """

    # ============================================================
    # TODO: Replace this naive baseline with YOUR navigation field.
    # ============================================================

    ## --- Naive placeholder baseline (will often fail): straight to target.
    #V = targets - state
    #norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    #return V / norms  # unit direction toward target

    ## Navigation field ##

    # obs_detect_rad = 0.25
    k = 0.8
    k_bot = 3
    K_weight = 1
    safetyZone = 0.95
    boundaryStrength = 1

    ####Goal's pot bowl (0 potential)
    K = K_weight * np.eye(len(state))
    #Pgoal = 0.5*np.subtract(state, targets).T @ K @ np.subtract(state, targets) #bowl centered @ target
    Fgoal = K @ np.subtract(targets, state)

    ####boundary repulsion
    # Fboundary = np.copy(Fgoal)
    # for i in range(len(state)):
    #   if(state[i][0]**2 + state[i][1]**2 > safetyZone): #x^2+y^2
    #      Fboundary[i][0] = boundaryStrength * -state[i][0]
    #      Fboundary[i][1] = boundaryStrength * -state[i][1]
    #   else:
    #      Fboundary[i][0] = 0
    #      Fboundary[i][1] = 0

    ####obstacles repulsion (1 potential)
    obs_forces = []
    for obstacle in obstacles:
      d_obs = np.copy(state) #get size of state vector
      for i in range(len(d_obs)):
         d_obs[i][0] = state[i][0] - obstacle["center"][0]
         d_obs[i][1] = state[i][1] - obstacle["center"][1]

         angle = np.arctan2(d_obs[i][1], d_obs[i][0])

         d_obs[i][0] -= np.cos(angle)*(obstacle["radius"] + r/2)
         d_obs[i][1] -= np.sin(angle)*(obstacle["radius"] + r/2)

      for i in range(len(d_obs)):
        # Obstacle detected and close enough to move
        if np.linalg.norm(d_obs[i]) < 0.05:
          # Colinearity detection
          angle = np.arctan2(d_obs[i][1], d_obs[i][0])
          if np.dot(Fgoal[i]/np.linalg.norm(Fgoal[i]), d_obs[i]/np.linalg.norm(d_obs[i])) < -0.99:
            d_obs[i][0] = -Fgoal[i][0] - np.sin(angle)
            d_obs[i][1] = -Fgoal[i][1] + np.cos(angle)
          # Default repulsion function
          else:
            magnitude = k/np.linalg.norm(d_obs[i])**3
            d_obs[i][0] = np.cos(angle)*magnitude
            d_obs[i][1] = np.sin(angle)*magnitude
        else:
          d_obs[i][0] = 0 
          d_obs[i][1] = 0
      
      obs_forces.append(d_obs)

    ob_F_total = np.copy(obs_forces[0])
    for i in range(len(obs_forces) - 1):
      #  print(ob_F_total)
      #  print(obs_forces[i + 1])
       ob_F_total = np.add(ob_F_total, obs_forces[i + 1])
       
    # rob_forces = []
    # for j in range(len(state)):
    #   d_rob = np.copy(state) #get size of state vector
    #   for i in range(len(d_rob)):
    #      d_rob[i][0] = state[i][0] - state[j][0]
    #      d_rob[i][1] = state[i][1] - state[j][1]

    #      angle = np.arctan2(d_obs[i][1], d_obs[i][0])

    #      d_rob[i][0] -= np.cos(angle)*r
    #      d_rob[i][1] -= np.sin(angle)*r

    #   for i in range(len(d_rob)):
    #     # Obstacle detected and close enough to move
    #     if (i < j and np.linalg.norm(d_rob[i]) < 2.1*r) or (j < i and np.linalg.norm(d_rob[i]) < 3*r):
    #       # Colinearity detection
    #       angle = np.arctan2(d_rob[i][1], d_rob[i][0])
    #       if np.dot(Fgoal[i]/np.linalg.norm(Fgoal[i]), d_rob[i]/np.linalg.norm(d_rob[i])) < -0.99:
    #         d_rob[i][0] = -Fgoal[i][0] - np.sin(angle)
    #         d_rob[i][1] = -Fgoal[i][1] + np.cos(angle)
    #       # Default repulsion function
    #       else:
    #         magnitude = k_bot/np.linalg.norm(d_rob[i])**3
    #         d_rob[i][0] = np.cos(angle)*magnitude
    #         d_rob[i][1] = np.sin(angle)*magnitude
    #     else:
    #       d_rob[i][0] = 0 
    #       d_rob[i][1] = 0
      
    #   rob_forces.append(d_rob)

    # rob_F_total = np.copy(rob_forces[0])
    # for i in range(len(rob_forces) - 1):
    #   #  print(ob_F_total)
    #   #  print(obs_forces[i + 1])
    #    rob_F_total = np.add(rob_F_total, rob_forces[i + 1])

    # return np.add(Fboundary, np.add(Fgoal, ob_F_total))
    Fsum = np.add(Fgoal, ob_F_total)
    # Fsum = np.add(Fsum, rob_F_total)

    # for i in range(len(Fsum)):
    #    if np.linalg.norm(Fsum[i]) < 0.001:
    #       # print("Fgoal:\n", Fgoal)
    #       # print("Fobs:\n", ob_F_total)
    #       Fsum[i] = np.add(Fsum[i], np.array([0.01, 0.01]))

    return Fsum

    
