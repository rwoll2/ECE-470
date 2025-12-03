import numpy as np

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

    #Goal's pot bowl (0 potential)
    #obstacles repulsion (1 potential)
    #collision repulsion  (1 potential)

    
