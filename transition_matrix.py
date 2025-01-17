import jax
import jax.numpy as jnp
from jax import random, vmap
from tqdm import tqdm

def generate_all_beliefs(l):
    """
    Generate all possible belief states for l propositions.
    Each proposition can be -1 (false), 0 (unknown), or 1 (true).
    Returns a matrix of shape (3^l, l) where each row is a possible belief state.
    """
    # Create base values: -1, 0, 1
    values = jnp.array([-1, 0, 1])
    
    # Use meshgrid to generate all combinations
    grids = jnp.meshgrid(*[values for _ in range(l)], indexing='ij')
    
    # Stack and reshape to get final matrix
    all_beliefs = jnp.stack(grids, axis=-1).reshape(-1, l)
    
    return all_beliefs

@jax.jit
def consensus_operator(belief_1, belief_2):
    """
    Vectorised fusion operator that works with JAX.
    The fusion process is defined as follows:
    - If the truth value of a proposition is the same for both beliefs, then
      the truth value of the proposition remains the same for the fused belief.
    - If one has a certain truth value (0 or 1) and the other is unknown, then
      the fused belief adopts the certain truth value.
    - If both have certain truth values but they are conflicting, then the fused
      belief adopts the unknown truth value.
    
    Args:
        beliefs1: shape (N, l) or (l,) - first set of belief vectors
        beliefs2: shape (N, l) or (l,) - second set of belief vectors
    
    Returns:
        Fused beliefs of same shape as inputs
    """
    sum_result = belief_1 + belief_2
    return jnp.where(
        sum_result < 0,
        -1,
        jnp.where(
            sum_result == 0,
            0,
            1
        )
    )


def create_transition_matrix(l):
    """
    Create the transition matrix for the fusion process.
    This is a 3^l x 3^l matrix, where each row and column corresponds to a
    belief state.
    The probability of transitioning from one belief state to another is given
    by the number of ways to transition from one belief state to another,
    according to the fusion operator.
    """
    # Generate all possible belief states
    belief_set = generate_all_beliefs(l)
    n_states = 3**l

    # Create the transition matrix
    transition_matrix = jnp.zeros((n_states, n_states))
    
    # Calculate the transition matrix
    for i, belief_1 in tqdm(enumerate(belief_set), total=len(belief_set), desc="Computing fusions for belief 1", leave=True):
        for j, belief_2 in tqdm(enumerate(belief_set), total=len(belief_set), desc="with belief 2", leave=False):
            fused_belief = consensus_operator(belief_1, belief_2)
            # Find the index of the fused belief in belief_set
            fused_idx = jnp.where((belief_set == fused_belief).all(axis=1))[0][0]
            transition_matrix = transition_matrix.at[i, fused_idx].add(1)

    # Safe normalization: avoid division by zero
    row_sums = jnp.sum(transition_matrix, axis=1)
    # Add small epsilon to avoid division by zero
    row_sums = jnp.where(row_sums == 0, 1.0, row_sums)
    
    # Normalize
    transition_matrix = transition_matrix / row_sums[:, None]

    return transition_matrix



if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate belief states and transition matrix')
    parser.add_argument('-l', type=int, default=3,
                       help='Language size (default: 3)')
    
    args = parser.parse_args()
    l = args.l

    # Generate all possible belief states
    belief_set = generate_all_beliefs(l)
    print(belief_set)

    transition_matrix = create_transition_matrix(l)
    print(transition_matrix)