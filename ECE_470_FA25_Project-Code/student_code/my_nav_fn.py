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
    
    n = state.shape[0]
    eps = 1e-8  # Small constant to prevent division by zero
    
    # =====================
    # Tunable Parameters
    # =====================
    k_goal = 1.0          # Goal attraction strength
    k_obs = 0.5           # Obstacle repulsion strength
    k_pair = 2.0          # Robot-robot repulsion strength
    k_boundary = 0.5      # Boundary repulsion strength
    
    obs_detect_rad = 0.02      # Distance at which obstacle forces activate
    pair_detect_rad = 0.5     # Distance at which robot-robot forces activate
    med_pair_detect_rad = 0.125 
    close_pair_detect_rad = 0.0625 
    boundary_detect_rad = 0  # Distance from boundary to activate repulsion
    
    # =====================
    # Step 1: Goal Attraction (Potential Bowl)
    # =====================
    F_goal = k_goal * (targets - state)
    
    # =====================
    # Step 2: Obstacle Repulsion
    # =====================
    F_obs = np.zeros_like(state)
    
    for obs in obstacles:
        center = np.array(obs["center"], dtype=float)
        R_obs = float(obs["radius"])
        
        # Vector from obstacle center to each robot
        diff = state - center  # (n, 2)
        dist_to_center = np.linalg.norm(diff, axis=1, keepdims=True)  # (n, 1)
        
        # Distance to obstacle surface (accounting for robot radius)
        d_surface = dist_to_center - (R_obs + r)  # (n, 1)
        
        # Unit direction away from obstacle
        direction = diff / (dist_to_center + eps)  # (n, 2)
        
        # Apply repulsion within detection radius
        in_range = (d_surface < obs_detect_rad) & (d_surface > eps)
        magnitude = np.where(in_range, k_obs / (d_surface ** 2 + eps), 0.0)
        
        # Local minima escape: add tangential component when obstacle is between robot and goal
        for i in range(n):
            # Vector from robot to goal
            to_goal = targets[i] - state[i]
            to_goal_norm = np.linalg.norm(to_goal)
            if to_goal_norm < eps:
                continue
            to_goal_unit = to_goal / to_goal_norm
            
            # Vector from robot to obstacle center
            to_obs = center - state[i]
            to_obs_norm = np.linalg.norm(to_obs)
            if to_obs_norm < eps:
                continue
            to_obs_unit = to_obs / to_obs_norm
            
            # Distance from robot to obstacle surface
            d_to_obs_surface = to_obs_norm - (R_obs + r)
            
            # Check if obstacle is in the way (goal is roughly behind obstacle)
            alignment = np.dot(to_goal_unit, to_obs_unit)
            
            # Also check if goal is farther than obstacle
            if alignment > 0.5 and to_goal_norm > to_obs_norm and d_to_obs_surface < obs_detect_rad * 2:
                # Add perpendicular tangent force to escape
                tangent = np.array([-to_obs_unit[1], to_obs_unit[0]])  # 90 degree rotation
                
                # Use robot index to pick consistent direction
                if i % 2 == 0:
                    tangent = -tangent
                
                # Tangent force based on alignment and proximity
                # Stronger when more aligned and closer
                prox_factor = max(0, 1 - d_to_obs_surface / (obs_detect_rad * 2))
                tangent_strength = alignment * prox_factor * k_obs * 0.8
                F_obs[i] += tangent_strength * tangent
        
        F_obs += magnitude * direction
    
    # =====================
    # Step 3: Robot-Robot Repulsion with Symmetry Breaking
    # =====================
    F_pair = np.zeros_like(state)
    collision_dist = 2 * r  # Distance at which collision is declared
    
    for i in range(n):
        for j in range(i+1, n):  # Only process each pair once
            diff_ij = state[i] - state[j]  # Vector from j to i
            dist_ij = np.linalg.norm(diff_ij)
            
            # Distance beyond collision threshold
            d_separation = dist_ij - collision_dist
            
            if d_separation < pair_detect_rad and d_separation > eps:
                # Unit direction
                direction = diff_ij / (dist_ij + eps)
                
                my_goal_dir = targets[i] - state[i]
                their_goal_dir = targets[j] - state[j]
                my_goal_norm = np.linalg.norm(my_goal_dir)
                their_goal_norm = np.linalg.norm(their_goal_dir)
                
                # Force magnitude
                magnitude = 0
                if d_separation < close_pair_detect_rad:
                  magnitude = k_pair / (d_separation ** 3 + eps)
                elif d_separation < med_pair_detect_rad:
                  magnitude = k_pair / (d_separation ** 2 + eps)
                else:
                  magnitude = k_pair / (20*(d_separation + eps))
                
                
                # Apply repulsive force to both robots (Newton's 3rd law)
                if np.linalg.norm(state[i]) < 1 - r - eps and (their_goal_norm > eps or d_separation < close_pair_detect_rad):
                  F_pair[i] += magnitude * direction
                if np.linalg.norm(state[j]) < 1 - r - eps and (my_goal_norm > eps or d_separation < close_pair_detect_rad):
                  F_pair[j] -= magnitude * direction
                
                # Symmetry breaking: detect if robots would collide head-on
                if my_goal_norm > eps and their_goal_norm > eps:
                    my_goal_unit = my_goal_dir / my_goal_norm
                    their_goal_unit = their_goal_dir / their_goal_norm
                    
                    # Check if trajectories would cross
                    # 1. Goals are roughly opposite (heading toward each other)
                    goal_alignment = np.dot(my_goal_unit, their_goal_unit)
                    
                    # 2. Each robot is heading toward the other's current position
                    heading_toward_j = np.dot(my_goal_unit, -direction)  # i heading toward j?
                    heading_toward_i = np.dot(their_goal_unit, direction)  # j heading toward i?
                    
                    # Apply symmetry break if either condition is met
                    if goal_alignment < -0.3 or (heading_toward_j > 0.3 and heading_toward_i > 0.3):
                        # Add perpendicular nudge
                        tangent = np.array([-direction[1], direction[0]])
                        
                        # Use index to consistently split: robot with lower index goes one way
                        # This creates consistent "traffic rules"
                        nudge_strength = magnitude * 0.2
                        if np.linalg.norm(state[i]) < 1 - r - eps:
                          F_pair[i] += nudge_strength * tangent
                        if np.linalg.norm(state[j]) < 1 - r - eps:
                          F_pair[j] -= nudge_strength * tangent  # Opposite direction
    
    # =====================
    # Step 4: Boundary Repulsion (Stay inside unit disk)
    # =====================
    F_boundary = np.zeros_like(state)
    
    dist_from_origin = np.linalg.norm(state, axis=1, keepdims=True)  # (n, 1)
    d_to_boundary = 1.0 - dist_from_origin  # (n, 1)
    
    # Unit direction toward center (negative of position direction)
    direction_to_center = -state / (dist_from_origin + eps)  # (n, 2)

    to_goal = np.linalg.norm(targets - state)
    
    # Apply repulsion when close to boundary
    in_range = (d_to_boundary < boundary_detect_rad) & (d_to_boundary > eps) & (to_goal > r + eps)
    magnitude = np.where(in_range, k_boundary / (d_to_boundary ** 2 + eps), 0.0)
    
    F_boundary = magnitude * direction_to_center
    
    # =====================
    # Combine Forces
    # =====================
    F_total = F_goal + F_obs + F_pair + F_boundary
    
    # Ensure no NaN/Inf values
    F_total = np.nan_to_num(F_total, nan=0.0, posinf=0.0, neginf=0.0)
    
    return F_total
