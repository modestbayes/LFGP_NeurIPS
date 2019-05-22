import numpy as np


def build_covariance_blocks(F_covariance_list, loading_matrix, Y_variance):
    """
    Build covariance matrix for long vector of all columns of Y stacked together.

    Args
        F_covariance_list: (list) of [t, t] covariance matrices
        loading_matrix: (numpy array) [r, q] linear transformation between F and Y
        Y_sigma_list: (numpy array) [q] variance parameters for columns of Y
    """
    r = len(F_covariance_list)
    t = F_covariance_list[0].shape[0]
    q = loading_matrix.shape[1]
    block_FF_rows = []  # covariance for columns of F
    for i in range(r):
        current_row = np.zeros((t, r * t))
        current_row[:, (i * t):(i * t + t)] = F_covariance_list[i]
        block_FF_rows.append(current_row)
    block_FF = np.vstack(block_FF_rows)
    block_FY_rows = []  # covariance between columns of F and columns of Y
    for i in range(r):
        current_row = np.zeros((t, q * t))
        for j in range(q):
            current_row[:, (j * t):(j * t + t)] = loading_matrix[i, j] * F_covariance_list[i]
        block_FY_rows.append(current_row)
    block_FY = np.vstack(block_FY_rows)
    block_YF = np.transpose(block_FY)
    block_YY_rows = []   # covariance between columns of Y
    for i in range(q):
        current_row = np.zeros((t, q * t))
        for j in range(q):
            for k in range(r):
                current_row[:, (j * t):(j * t + t)] += F_covariance_list[k] * loading_matrix[k, i] * loading_matrix[
                    k, j]
            if i == j:
                current_row[:, (j * t):(j * t + t)] += np.eye(t) * Y_variance[i]  # diagonal variance
        block_YY_rows.append(current_row)
    block_YY = np.vstack(block_YY_rows)
    return block_FF, block_FY, block_YF, block_YY


def factor_covariance_blocks(F_covariance_list, loading_matrix, Y_variance, factor_index):
    """
    Build covariance matrix for long vector of all columns of Y stacked together.

    Args
        F_covariance_list: (list) of [t, t] covariance matrices
        loading_matrix: (numpy array) [r, q] linear transformation between F and Y
        Y_sigma_list: (numpy array) [q] variance parameters for columns of Y
    """
    r = len(F_covariance_list)
    t = F_covariance_list[0].shape[0]
    q = loading_matrix.shape[1]
    block_YY = np.zeros((q * t, q * t))
    # covariance for columns of F
    block_FF = F_covariance_list[factor_index]
    # covariance between columns of F and columns of Y
    block_FY_rows = []
    current_row = np.zeros((t, q * t))
    for j in range(q):
        current_row[:, (j * t):(j * t + t)] = loading_matrix[factor_index, j] * F_covariance_list[factor_index]
    block_FY = current_row
    block_YF = np.transpose(block_FY)
    # covariance between columns of Y
    block_YY_rows = []
    for i in range(q):
        current_row = np.zeros((t, q * t))
        for j in range(q):
            current_row[:, (j * t):(j * t + t)] += F_covariance_list[factor_index] \
                                                   * loading_matrix[factor_index, i] * loading_matrix[factor_index, j]
            if i == j:
                current_row[:, (j * t):(j * t + t)] += np.eye(t) * Y_variance[i]  # diagonal variance
        block_YY_rows.append(current_row)
    block_YY = np.vstack(block_YY_rows)
    return block_FF, block_FY, block_YF, block_YY


def sample_conditional_F(Y, F_covariance_list, loading_matrix, Y_variance):
    """
    Sample from conditional distribution of F given everything else.

    Args
        Y: (numpy array) [t, q] observed multivariate time series
        block_FF, block_FY, block_YF, block_YY: (numpy array) blocks in the covariance of joint distribution
    """
    block_FF, block_FY, block_YF, block_YY = build_covariance_blocks(F_covariance_list, loading_matrix, Y_variance)
    t, q = Y.shape
    r = int(block_FF.shape[0] / t)
    Y_stack = np.transpose(Y).reshape(t * q)  # stack columns of Y
    block_YY_inverse = np.linalg.inv(block_YY)
    prod = np.matmul(block_FY, block_YY_inverse)
    mu = np.matmul(prod, Y_stack)
    covariance = block_FF - np.matmul(prod, block_YF)
    F_stack = np.random.multivariate_normal(mu, covariance)
    F_sample = np.transpose(F_stack.reshape((r, t)))  # de-stack columns of F
    return F_sample


def conditional_F_dist(F_covariance_list, loading_matrix, Y_variance):
    block_FF, block_FY, block_YF, block_YY = build_covariance_blocks(F_covariance_list, loading_matrix, Y_variance)
    block_YY_inverse = np.linalg.inv(block_YY)
    prod = np.matmul(block_FY, block_YY_inverse)
    covariance = block_FF - np.matmul(prod, block_YF)
    return prod, covariance


def sample_conditonal_F_dist(Y, prod, covariance):
    t, q = Y.shape
    r = int(covariance.shape[0] / t)
    Y_stack = np.transpose(Y).reshape(t * q)  # stack columns of Y
    mu = np.matmul(prod, Y_stack)
    F_stack = np.random.multivariate_normal(mu, covariance)
    F_sample = np.transpose(F_stack.reshape((r, t)))  # de-stack columns of F
    return F_sample


def iterative_conditional_F(Y, F_covariance_list, loading_matrix, Y_variance):
    """
    Sample conditional distribution of F by each factor as iterative regression.
    """
    r = len(F_covariance_list)
    t, q = Y.shape
    Y_stack = np.transpose(Y).reshape(t * q)  # stack columns of Y
    residuals = Y_stack.copy()
    F_stack = np.zeros(t * r)
    for i in range(r):
        block_FF, block_FY, block_YF, block_YY = factor_covariance_blocks(F_covariance_list, loading_matrix, Y_variance, i)
        block_YY_inverse = np.linalg.inv(block_YY)
        prod = np.matmul(block_FY, block_YY_inverse)
        mu = np.matmul(prod, residuals)
        covariance = block_FF - np.matmul(prod, block_YF)
        conditional = np.random.multivariate_normal(mu, covariance)
        F_stack[(i * t):(i * t + t)] = conditional
        hat = np.matmul(conditional.reshape((t, 1)), loading_matrix[i, :].reshape((1, q)))
        residuals = residuals - np.transpose(hat).reshape(t * q)
    F_sample = np.transpose(F_stack.reshape((r, t)))  # de-stack columns of F
    return F_sample


def conditional_factor_dist(F_covariance_list, loading_matrix, Y_variance, factor_index):
    block_FF, block_FY, block_YF, block_YY = factor_covariance_blocks(F_covariance_list, loading_matrix, Y_variance, factor_index)
    block_YY_inverse = np.linalg.inv(block_YY)
    prod = np.matmul(block_FY, block_YY_inverse)
    covariance = block_FF - np.matmul(prod, block_YF)
    return prod, covariance


def sample_conditonal_factor_dist(res, prod, covariance):
    t, q = res.shape
    r = int(covariance.shape[0] / t)
    Y_stack = np.transpose(res).reshape(t * q)  # stack columns of Y
    mu = np.matmul(prod, Y_stack)
    F_sample = np.random.multivariate_normal(mu, covariance)
    return F_sample


def kronecker_A(loading_matrix, factor_index):
    """
    Covariance between columns of Y with respect to a specific factor.
    """
    r, q = loading_matrix.shape
    A = np.zeros((q, q))
    for i in range(q):
        for j in range(q):
            A[i, j] = loading_matrix[factor_index, i] * loading_matrix[factor_index, j]
    return A


def quick_inverse(A, cov, Y_variance):
    """
    Calculate inverse of the sum of kronecker product and identity matrix.
    """
    q = A.shape[0]
    t = cov.shape[0]
    var = np.zeros(q * t)
    for i in range(q):
        var[(i * t):(i * t + t)] = Y_variance[i]
    v1, W1 = np.linalg.eigh(A)
    v2, W2 = np.linalg.eigh(cov)
    S_diag = np.zeros(q * t)
    for i in range(q):
        S_diag[(i * t):(i * t + t)] = v1[i] * v2
    front = np.kron(W1, W2)
    back = np.transpose(front)
    inverse = np.matmul(front, np.diag(1.0 / (var + S_diag)))
    inverse = np.matmul(inverse, back)
    return inverse
