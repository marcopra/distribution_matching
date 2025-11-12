# Creating a tabular grid environment (two 5x5 rooms connected by a 1x1 corridor),
# building transition matrices for actions, creating a policy operator, computing
# the discounted occupancy measure nu_star = (I - gamma M_pi)^{-1} nu0, and
# simulating rollouts to validate the distribution empirically.
#
# This cell will construct:
# - state space and mapping to (x,y)
# - deterministic transitions for actions (up/down/left/right) with walls
# - policy: uniform random policy (π(a|s) = 1/4)
# - policy operator (maps state distribution nu -> state-action distribution mu)
# - M_pi (state -> next-state matrix under policy)
# - compute nu_star (discounted future-state occupancy) for gamma
# - simulate trajectories to estimate empirical state visitation distribution
#
# Finally it will show small printed diagnostics and two heatmaps:
# - analytical discounted occupancy (reshaped on the global map)
# - empirical visit frequencies from simulation
#
# No external network calls.


import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# Geometry: two 5x5 rooms with a 1x1 corridor at (5,2)
left_xs = range(0,5)
right_xs = range(6,11)
ys = range(0,5)
corridor = (5,2)

# Build valid cells (states)
cells = []
# left room
for x in left_xs:
    for y in ys:
        cells.append((x,y))
# corridor
cells.append(corridor)
# right room
for x in right_xs:
    for y in ys:
        cells.append((x,y))

# state indexing
state_to_idx = {s:i for i,s in enumerate(cells)}
idx_to_state = {i:s for s,i in state_to_idx.items()}
n_states = len(cells)

# actions: up, down, left, right
actions = ['up','down','left','right']
n_actions = len(actions)

def step_from(cell, action):
    x,y = cell
    if action == 'up':
        nx, ny = x, y-1
    elif action == 'down':
        nx, ny = x, y+1
    elif action == 'left':
        nx, ny = x-1, y
    elif action == 'right':
        nx, ny = x+1, y
    else:
        raise ValueError(action)
    if (nx,ny) in state_to_idx:
        return (nx,ny)
    else:
        return (x,y)

# Build per-action transition matrices P_a of shape (n_states, n_states)
# P_a[s, s'] = P(s' | s, a) (we use deterministic transitions => entries are 0/1)
# Build T with rows = next-state, cols = linear index of (state, action)
T_operator = np.zeros((n_states, n_states * n_actions))
for s in range(n_states):
    x, y = idx_to_state[s]
    for a_idx, a in enumerate(actions):
        s_next = step_from((x, y), a)
        s_next_idx = state_to_idx[s_next]
        col = s * n_actions + a_idx
        T_operator[s_next_idx, col] = 1.0

# Build T with rows = next-state, cols = linear index of (state, action)
T_mat = np.zeros((n_states, n_states * n_actions))
for s in range(n_states):
    x, y = idx_to_state[s]
    for a_idx, a in enumerate(actions):
        s_next = step_from((x, y), a)
        s_next_idx = state_to_idx[s_next]
        col = s * n_actions + a_idx
        T_mat[s_next_idx, col] = 1.0
T_operator = T_mat  # shape (n_states, n_states*n_actions)


# Policy operator: rows = (state, action), cols = state
P_mat = np.zeros((n_states * n_actions, n_states))
for s in range(n_states):
    for a_idx in range(n_actions):
        row = s * n_actions + a_idx
        P_mat[row, s] = 1.0 / n_actions
policy_operator = P_mat

uniform_random_policy = np.copy(policy_operator)


# Define target state distribution as a uniform distribution over all states
nu_target = np.ones(n_states) / n_states
nu_target = nu_target.reshape((-1,1))

# Starting distribution nu0: uniform over top-left 2x2 corner of left room.
# Let's define top-left corner as x in {0,1}, y in {0,1} (choose "top-left" as small coords).
# This yields 4 states.
top_left_coords = [(0,0),(1,0),(0,1),(1,1)]
nu0 = np.zeros(n_states)
for c in top_left_coords:
    nu0[state_to_idx[c]] = 1.0 / len(top_left_coords)

nu0 = nu0.reshape((-1,1))

# Parameters
gamma = 0.9
eta = 0.1
n_updates = 10000
gradient_type = 'forward'  # 'forward' or 'reverse'


def M_pi_operator(P, T):
    """
    M_π = T ∘ Π_π
    Maps state distribution to next state distribution under policy
    """

    return T@P

def compute_discounted_occupancy(nu0, P, T=T_operator, gamma=0.99, max_iter=None):
    n = len(nu0)
    I = np.eye(n)
    M = T @ P
    if max_iter is None:
        return (1-gamma) * np.linalg.inv(I - gamma * M) @ nu0
    nu = nu0.copy()
    for _ in range(max_iter):
        nu_new = nu0 + gamma * M @ nu
        if np.allclose(nu, nu_new, atol=1e-8):
            break
        nu = nu_new
    return (1-gamma) * nu

def kl_divergence(p, q):
    """KL(p||q) with numerical stability"""
    eps = 1e-10
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    # Add normalization to ensure they sum to 1 after clipping
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))

def compute_gradient_kl(nu0, nu_target, P, T = T_operator, gamma = 0.99, gradient_type='reverse'):

    n = len(nu0)
    I = np.eye(n)
    M = M_pi_operator(P, T)
    if gradient_type == 'reverse':
        r = nu0/((I - gamma * M) @ nu_target)
        gradient = gamma * T.T @ r @ nu_target.T
    elif gradient_type == 'forward':

        nu_pi_over_nu_target = np.clip(np.linalg.solve(I - gamma * M, nu0) / nu_target, a_min=1e-10, a_max=None)
        # check all > 0
        assert np.all(nu_pi_over_nu_target > 0), "nu_pi_over_nu_target has non-positive entries"
        log_nu_pi_over_nu_target = np.log(nu_pi_over_nu_target)
        gradient = gamma * np.linalg.solve(I - gamma * M, T).T
        gradient = gradient @ (np.ones_like(log_nu_pi_over_nu_target) + log_nu_pi_over_nu_target)
        gradient = gradient @ np.linalg.solve(I - gamma * M, nu0).T
    else:
        raise ValueError(f"Unknown gradient type: {gradient_type}")

    return gradient

# def compute_gradient_kl(nu0, nu_target, P, T = T_operator, gamma = 0.99, gradient_type='reverse'):

#     n = len(nu0)
#     I = np.eye(n)
#     M = M_pi_operator(P, T)
#     if gradient_type == 'reverse':
#         r = nu0/((I - gamma * M) @ nu_target)
#         gradient = gamma * T.T @ r @ nu_target.T
#     elif gradient_type == 'forward':
#         # Add regularization for numerical stability
#         reg = 1e-8
#         i_gammaTP_inv = np.linalg.inv(I - gamma * M + reg * I)
        
#         nu_pi = i_gammaTP_inv @ nu0
#         # Clip and normalize nu_pi before log
#         eps = 1e-10
#         nu_pi = np.clip(nu_pi, eps, 1.0)
#         nu_pi = nu_pi / nu_pi.sum()
#         nu_target_safe = np.clip(nu_target, eps, 1.0)
#         nu_target_safe = nu_target_safe / nu_target_safe.sum()
        
#         log_nu_pi_over_nu_target = np.log(nu_pi / nu_target_safe)
#         gradient = gamma * T.T @ i_gammaTP_inv.T
#         gradient = gradient @ (np.ones_like(log_nu_pi_over_nu_target) + log_nu_pi_over_nu_target)
#         gradient = gradient @ nu0.T @ i_gammaTP_inv.T
#     else:
#         raise ValueError(f"Unknown gradient type: {gradient_type}")

#     return gradient

# def mirror_descent_update(policy_operator, gradient, eta):
#     """
#     Mirror Descent update with KL regularization:
#     π_{t+1}(a|s) = π_t(a|s) * exp(-η ∇_{a,s} f) / Z_s
#     """
#     new_policy_operator = policy_operator * np.exp(-eta * gradient)
#     # Normalize per state to ensure valid probability distribution
#     new_policy_operator = new_policy_operator / (new_policy_operator.sum(axis=1, keepdims=True)) # + 1e-10)
#     return new_policy_operator

def mirror_descent_update(policy_operator, gradient, eta):
    """
    Mirror Descent update: π_{t+1}(a|s) ∝ π_t(a|s) * exp(-η ∇_{a,s} f)
    """
    n_states = policy_operator.shape[1]
    n_actions = policy_operator.shape[0] // n_states
    
    policy_3d = policy_operator.reshape((n_states, n_actions, n_states))
    gradient_3d = gradient.reshape((n_states, n_actions, n_states))
    
    new_policy_3d = policy_3d * np.exp(-eta * gradient_3d)
    
    # Normalize the block-diagonal elements
    for s in range(n_states):
        policy_s_actions = new_policy_3d[s, :, s]
        new_policy_3d[s, :, s] = policy_s_actions / (policy_s_actions.sum() + 1e-10)
    
    return new_policy_3d.reshape((n_states * n_actions, n_states))
# ==================== OPTIMIZATION ====================

print("Starting optimization...")
print("\n" + "="*60)
print("OPTIMIZATION PARAMETERS")
print("="*60)
print(f"Discount factor γ: {gamma}")
print(f"Learning rate η: {eta}")
print(f"Number of updates: {n_updates}")
print(f"Gradient type: {gradient_type}")
print("="*60 + "\n")

kl_history = []

for iteration in range(n_updates):
    if gradient_type == 'forward':
        nu_pi = compute_discounted_occupancy(nu0, policy_operator, T_operator, gamma)
        kl = kl_divergence(nu_pi, nu_target)
    elif gradient_type == 'reverse':
        n = len(nu0)
        I = np.eye(n)
        M = M_pi_operator(policy_operator, T_operator)
        kl = kl_divergence(nu0, (I - gamma * M) @ nu_target)
    kl_history.append(kl)
    
    
    # Compute gradient
    gradient = compute_gradient_kl(nu0, nu_target, policy_operator, T_operator,  gamma, gradient_type)
    
    # Update policy
    policy_operator = mirror_descent_update(policy_operator, gradient, eta)
    
    if iteration % 10 == 0:
        print(f"Iter {iteration:3d}: KL = {kl:.6f}")

print("\nOptimization complete!")

# ==================== VISUALIZATION ====================

# Compute final occupancy
nu_final = compute_discounted_occupancy(nu0, policy_operator, T_operator, gamma)

# Create grid for visualization
grid_width = 11
grid_height = 5

def state_dist_to_grid(nu):
    """Convert state distribution to 2D grid"""
    grid = np.zeros((grid_height, grid_width))
    for s_idx in range(n_states):
        x, y = idx_to_state[s_idx]
        grid[y, x] = nu[s_idx]
    return grid

# Create visualizations
fig = plt.figure(figsize=(20, 15))

# Add parameter table as text in the top-left corner
param_text = f"Parameters:\nγ = {gamma}\nη = {eta}\nUpdates = {n_updates}\nGradient = {gradient_type}"
fig.text(0.02, 0.98, param_text, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# First row: 3 columns
ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=1)
ax2 = plt.subplot2grid((3, 4), (0, 1), colspan=1)
ax3 = plt.subplot2grid((3, 4), (0, 2), colspan=1)

# Second row: 3 columns (first 2 used, last one spans 2)
ax4 = plt.subplot2grid((3, 4), (1, 0), colspan=1)
ax5 = plt.subplot2grid((3, 4), (1, 1), colspan=1)
ax6 = plt.subplot2grid((3, 4), (1, 2), colspan=2)

# Third row: 4 columns
ax7 = plt.subplot2grid((3, 4), (2, 0), colspan=1)
ax8 = plt.subplot2grid((3, 4), (2, 1), colspan=1)
ax9 = plt.subplot2grid((3, 4), (2, 2), colspan=1)
ax10 = plt.subplot2grid((3, 4), (2, 3), colspan=1)

# 1. Initial distribution
grid_nu0 = state_dist_to_grid(nu0)
im1 = ax1.imshow(grid_nu0, cmap='YlOrRd', interpolation='nearest')
ax1.set_title('Initial Distribution ν₀')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_xticks(np.arange(-0.5, grid_width, 1), minor=True)
ax1.set_yticks(np.arange(-0.5, grid_height, 1), minor=True)
ax1.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.5)
plt.colorbar(im1, ax=ax1)

# 2. Target distribution
grid_target = state_dist_to_grid(nu_target)
im2 = ax2.imshow(grid_target, cmap='YlOrRd', interpolation='nearest')
ax2.set_title('Target Distribution ν*')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_xticks(np.arange(-0.5, grid_width, 1), minor=True)
ax2.set_yticks(np.arange(-0.5, grid_height, 1), minor=True)
ax2.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.5)
plt.colorbar(im2, ax=ax2)

# 3. Final learned distribution
grid_final = state_dist_to_grid(nu_final)
im3 = ax3.imshow(grid_final, cmap='YlOrRd', interpolation='nearest')
ax3.set_title('Final Discounted Occupancy')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_xticks(np.arange(-0.5, grid_width, 1), minor=True)
ax3.set_yticks(np.arange(-0.5, grid_height, 1), minor=True)
ax3.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.5)
plt.colorbar(im3, ax=ax3)

# 4. Policy visualization with arrows showing most probable actions
policy_matrix = policy_operator.reshape((n_states, n_actions, n_states))
policy_per_state = np.zeros((n_states, n_actions))
for s in range(n_states):
    policy_per_state[s, :] = policy_matrix[s, :, s]

# Create grid visualization with arrows
action_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}  # up, down, left, right
ax4.set_xlim(-0.5, grid_width - 0.5)
ax4.set_ylim(grid_height - 0.5, -0.5)
ax4.set_aspect('equal')
ax4.set_title('Policy Actions (arrows = most probable)')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.grid(True, alpha=0.3)

for s_idx in range(n_states):
    x, y = idx_to_state[s_idx]
    # Find actions with maximum probability
    max_prob = np.max(policy_per_state[s_idx])
    max_actions = np.where(np.isclose(policy_per_state[s_idx], max_prob, atol=1e-6))[0]
    
    # Create text with all arrows for max probability actions
    arrow_text = ''.join([action_symbols[a] for a in max_actions])
    
    # Draw background cell
    rect = plt.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8, 
                         facecolor='lightgray', edgecolor='black', linewidth=0.5)
    ax4.add_patch(rect)
    
    # Add arrows
    ax4.text(x, y, arrow_text, ha='center', va='center', 
                   fontsize=12, fontweight='bold')

# 5. Policy visualization (mini histograms per state)
ax5.set_xlim(-0.5, grid_width - 0.5)
ax5.set_ylim(grid_height - 0.5, -0.5)
ax5.set_aspect('equal')
ax5.set_title('Policy Probabilities per State')
ax5.set_xlabel('x')
ax5.set_ylabel('y')
ax5.grid(True, alpha=0.3)

# Colors for each action
action_colors = ['red', 'blue', 'green', 'orange']  # up, down, left, right

for s_idx in range(n_states):
    x, y = idx_to_state[s_idx]
    
    # Draw background cell
    rect = plt.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8, 
                         facecolor='lightgray', edgecolor='black', linewidth=0.5)
    ax5.add_patch(rect)
    
    # Get probabilities for this state
    probs = policy_per_state[s_idx]
    
    # Draw mini bar chart (4 bars side by side)
    bar_width = 0.15
    bar_spacing = 0.2
    start_x = x - 1.5 * bar_spacing
    max_bar_height = 0.7  # Maximum height in cell units
    
    for a_idx in range(n_actions):
        bar_x = start_x + a_idx * bar_spacing
        bar_height = probs[a_idx] * max_bar_height  # Proportional to probability
        
        # Draw bar from bottom of cell
        bar_rect = plt.Rectangle((bar_x - bar_width/2, y + 0.35 - bar_height), 
                                bar_width, bar_height,
                                facecolor=action_colors[a_idx], 
                                edgecolor='black', linewidth=0.3)
        ax5.add_patch(bar_rect)

# Add legend for actions
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=action_colors[i], label=actions[i]) 
                   for i in range(n_actions)]
ax5.legend(handles=legend_elements, loc='upper right', fontsize=8)

# 6. KL divergence history (spans 2 columns)
ax6.plot(kl_history)
ax6.set_xlabel('Iteration')
ax6.set_ylabel('KL(ν₀ || Q)')
ax6.set_title('KL Divergence vs Iteration')
ax6.grid(True)

# 7-10. Heatmaps per action (third row - 4 columns)
axes_third_row = [ax7, ax8, ax9, ax10]
for a_idx in range(n_actions):
    # Create grid for this action's probabilities
    grid_action = np.zeros((grid_height, grid_width))
    for s_idx in range(n_states):
        x, y = idx_to_state[s_idx]
        grid_action[y, x] = policy_per_state[s_idx, a_idx]
    
    im = axes_third_row[a_idx].imshow(grid_action, cmap='YlOrRd', interpolation='nearest', vmin=0, vmax=1)
    axes_third_row[a_idx].set_title(f'π({actions[a_idx]}|s)')
    axes_third_row[a_idx].set_xlabel('x')
    axes_third_row[a_idx].set_ylabel('y')
    axes_third_row[a_idx].set_xticks(np.arange(-0.5, grid_width, 1), minor=True)
    axes_third_row[a_idx].set_yticks(np.arange(-0.5, grid_height, 1), minor=True)
    axes_third_row[a_idx].grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.colorbar(im, ax=axes_third_row[a_idx])

plt.tight_layout()
plt.savefig('/home/mprattico/dist_tmp/distribution_matching_results.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved to: /home/mprattico/dist_tmp/distribution_matching_results.png")

plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Number of states: {n_states}")
print(f"Number of actions: {n_actions}")
print(f"Discount factor γ: {gamma}")
print(f"Learning rate η: {eta}")
print(f"Number of updates: {n_updates}")
print(f"Gradient type: {gradient_type}")
print(f"\nInitial KL: {kl_history[0]:.6f}")
print(f"Final KL: {kl_history[-1]:.6f}")
print(f"KL reduction: {kl_history[0] - kl_history[-1]:.6f}")
print(f"\nTarget distribution (uniform): mean={nu_target.mean():.6f}, std={nu_target.std():.6f}")
print(f"Final distribution: mean={nu_final.mean():.6f}, std={nu_final.std():.6f}")
print(f"Distribution L2 distance: {np.linalg.norm(nu_final - nu_target):.6f}")
