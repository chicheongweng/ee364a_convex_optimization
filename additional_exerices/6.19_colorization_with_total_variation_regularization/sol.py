import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# Load the original image
img = mpimg.imread("flower.png")
img = img[:, :, 0:3]  # Ensure it's RGB
m, n, _ = img.shape

# Generate the monochrome image
M = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

# Randomly select known indices
np.random.seed(5)
known_ind = np.where(np.random.rand(m, n) >= 0.90)

# Extract known color values
R_known = img[:, :, 0][known_ind]
G_known = img[:, :, 1][known_ind]
B_known = img[:, :, 2][known_ind]

# Initialize the given image with known values
R_given = np.copy(M)
R_given[known_ind] = R_known
G_given = np.copy(M)
G_given[known_ind] = G_known
B_given = np.copy(M)
B_given[known_ind] = B_known

# Save the monochrome image with known pixels colored
def save_img(filename, R, G, B):
    img = np.stack((np.array(R), np.array(G), np.array(B)), axis=2)
    plt.tick_params(
        axis="both",
        which="both",
        labelleft=False,
        labelbottom=False,
        bottom=False,
        top=False,
        right=False,
        left=False,
    )
    plt.imshow(img)
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.0)


save_img("flower_mono.png", R_given, G_given, B_given)

# Total variation regularization function
def total_variation(X, M, known_ind, R_known, G_known, B_known, m, n):
    R = X[: m * n].reshape((m, n))
    G = X[m * n : 2 * m * n].reshape((m, n))
    B = X[2 * m * n :].reshape((m, n))

    # Total variation term
    tv = 0
    for i in range(m - 1):
        for j in range(n - 1):
            grad = np.array(
                [
                    R[i, j] - R[i, j + 1],
                    G[i, j] - G[i, j + 1],
                    B[i, j] - B[i, j + 1],
                    R[i, j] - R[i + 1, j],
                    G[i, j] - G[i + 1, j],
                    B[i, j] - B[i + 1, j],
                ]
            )
            tv += np.linalg.norm(grad)

    # Consistency with monochrome image
    consistency = np.sum((0.299 * R + 0.587 * G + 0.114 * B - M) ** 2)

    # Consistency with known values
    R[known_ind] = R_known
    G[known_ind] = G_known
    B[known_ind] = B_known

    return tv + consistency


# Optimization
def colorize(M, R_known, G_known, B_known, known_ind, m, n):
    # Initial guess (flattened arrays)
    R_init = np.copy(M).flatten()
    G_init = np.copy(M).flatten()
    B_init = np.copy(M).flatten()
    X_init = np.concatenate([R_init, G_init, B_init])

    # Bounds for the optimization (values must be in [0, 1])
    bounds = [(0, 1)] * len(X_init)

    # Minimize the total variation
    result = minimize(
        total_variation,
        X_init,
        args=(M, known_ind, R_known, G_known, B_known, m, n),
        bounds=bounds,
        method="L-BFGS-B",
    )

    # Extract the optimized R, G, B channels
    X_opt = result.x
    R_opt = X_opt[: m * n].reshape((m, n))
    G_opt = X_opt[m * n : 2 * m * n].reshape((m, n))
    B_opt = X_opt[2 * m * n :].reshape((m, n))

    return R_opt, G_opt, B_opt


# Perform colorization
R_opt, G_opt, B_opt = colorize(M, R_known, G_known, B_known, known_ind, m, n)

# Save the colorized image
save_img("flower_colorized.png", R_opt, G_opt, B_opt)
