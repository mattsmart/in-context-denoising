import os
import sys

DIR_CURRENT = os.path.dirname(__file__)
DIR_PARENT = os.path.dirname(DIR_CURRENT)
sys.path.append(DIR_PARENT)

DIR_OUT = DIR_PARENT + os.sep + 'output'
DIR_DATA = DIR_PARENT + os.sep + 'input'
DIR_MODELS = DIR_PARENT + os.sep + 'models'
DIR_RUNS = DIR_PARENT + os.sep + 'runs'

for core_dir in [DIR_OUT, DIR_DATA, DIR_MODELS, DIR_RUNS]:
    if not os.path.exists(core_dir):
        os.mkdir(core_dir)

# settings for dataset construction + training (these apply to all three dataset cases below)
CONTEXT_LEN = 100
DIM_N = 32
NUM_W_IN_DATASET = 1000
CONTEXT_EXAMPLES_PER_W = 1
SAMPLES_PER_CONTEXT_EXAMPLE = 1
TEST_RATIO = 0.2

# Three cases implemented for dataset generation
#   1) 'Linear'     - samples are drawn from a linear subspace
#   2) 'Clustering' - samples are drawn from a mixture of Gaussians
#   3) 'Manifold'   - samples are drawn from a manifold (d-dim spheres)

DATASET_CASE = 0  # 0: 'Linear', 1: 'Clustering', 2: 'Manifold'
assert DATASET_CASE in [0, 1, 2]

DATASET_CASES = {
    0: 'Linear',
    1: 'Clustering',
    2: 'Manifold (sphere)'
}

# default kwargs for dataset generation for each case 0: 'Linear', 1: 'Clustering', 2: 'Manifold'
DATAGEN_GLOBALS = {
    0: dict(
        sigma2_corruption=0.5,                    # applies to all cases
        style_corruption_orthog=False,            # if True, the noise is only in orthogonal directions
        style_origin_subspace=True,               # if True, the subspace must contain origin (not affine)
        style_subspace_dimensions='random',       # int or 'random'
        # parameters specific to the Linear case
        sigma2_pure_context=2.0,                  # controls radius of ball of pure of samples (default: 2.0)
        corr_scaling_matrix=None,                 # x = Pz; y = A x  \tilde x = A (x + z); (normally identity I_n)
    ),
    1: dict(
        sigma2_corruption=0.5,  # applies to all cases
        style_corruption_orthog=False,     # mandatory for this case
        style_origin_subspace=True,        # mandatory for this case
        style_subspace_dimensions='full',  # full means dim-n gaussian balls in R^n
        # parameters specific to the Clustering case
        style_cluster_mus='unit_norm',
        style_cluster_vars='isotropic',
        num_cluster=3,
        cluster_var=0.01,                   # controls radius of ball of pure of samples (default: 2.0); could be a list in Clustering case
    ),
    2: dict(
        sigma2_corruption=0.5,                # applies to all cases
        style_corruption_orthog=False,        # mandatory for this case
        style_origin_subspace=True,           # mandatory for this case
        style_subspace_dimensions='random',   # int or 'random'
        # parameters specific to the Manifold case
        radius_sphere=1.0,                    # controls radius of sphere (d-dim manifold S in R^n)
    ),
}

# asserts for Linear case
assert DATAGEN_GLOBALS[0]['style_subspace_dimensions'] in ['random'] or isinstance(DATAGEN_GLOBALS[0]['style_subspace_dimensions'], int)
assert DATAGEN_GLOBALS[0]['style_corruption_orthog'] is False
assert DATAGEN_GLOBALS[0]['style_origin_subspace'] is True

# asserts for Clustering case
assert DATAGEN_GLOBALS[1]['style_corruption_orthog'] is False
assert DATAGEN_GLOBALS[1]['style_origin_subspace'] is True
assert DATAGEN_GLOBALS[1]['style_subspace_dimensions'] in ['full']  #  'full' means dim-n gaussian balls in R^n
assert DATAGEN_GLOBALS[1]['style_cluster_vars'] in ['isotropic']
assert DATAGEN_GLOBALS[1]['num_cluster'] in ['random'] or isinstance(DATAGEN_GLOBALS[1]['num_cluster'], int)

# asserts for Manifold case
assert DATAGEN_GLOBALS[2]['style_corruption_orthog'] is False
assert DATAGEN_GLOBALS[2]['style_origin_subspace'] is True
assert DATAGEN_GLOBALS[2]['style_subspace_dimensions'] in ['random'] or isinstance(DATAGEN_GLOBALS[2]['style_subspace_dimensions'], int)
assert DATAGEN_GLOBALS[2]['radius_sphere'] in ['random'] or isinstance(DATAGEN_GLOBALS[2]['radius_sphere'], float)

# settings for visualization
COLOR_TARGET = '#97C6CF'
COLOR_PRED = '#89C09F'
COLOR_INPUT = '#E9B24C'
