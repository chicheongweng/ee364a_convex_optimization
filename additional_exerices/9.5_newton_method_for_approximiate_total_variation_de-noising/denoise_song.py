import numpy as np
from scipy.linalg import solve_banded
from pydub import AudioSegment

# Parameters
mu = 50
epsilon = 1e-3  # Small value for smooth approximation
alpha = 0.01
beta = 0.5
tolerance = 1e-3

def gradient(x, x_cor, mu, epsilon):
    """
    Compute the gradient of the objective function using the original TV formulation.
    """
    n = len(x)
    grad = 2 * (x - x_cor)  # Gradient of the fidelity term
    for i in range(n - 1):
        diff = x[i + 1] - x[i]
        denom = np.sqrt(epsilon**2 + diff**2)
        grad[i] -= mu * 2 * diff / denom
        grad[i + 1] += mu * 2 * diff / denom
    return grad

def hessian(x, mu, epsilon):
    """
    Construct the tridiagonal Hessian matrix in banded form using the original TV formulation.
    """
    n = len(x)
    diag = 2 * np.ones(n)  # Diagonal entries from the fidelity term
    off_diag = np.zeros(n - 1)  # Off-diagonal entries

    for i in range(n - 1):
        diff = x[i + 1] - x[i]
        denom = (epsilon**2 + diff**2)**1.5
        diag[i] += mu * (2 / np.sqrt(epsilon**2 + diff**2) - 2 * diff**2 / denom)
        diag[i + 1] += mu * (2 / np.sqrt(epsilon**2 + diff**2) - 2 * diff**2 / denom)
        off_diag[i] -= mu * 2 * diff / denom

    return diag, off_diag

def newton_step(x, x_cor, mu, epsilon):
    """
    Compute the Newton step using the banded Hessian structure.
    """
    grad = gradient(x, x_cor, mu, epsilon)
    diag, off_diag = hessian(x, mu, epsilon)
    # Construct the banded matrix for solve_banded
    ab = np.zeros((3, len(x)))
    ab[1] = diag
    ab[0, 1:] = off_diag
    ab[2, :-1] = off_diag
    delta_x = solve_banded((1, 1), ab, -grad)
    return delta_x, grad

def line_search(x, x_cor, mu, epsilon, delta_x):
    """
    Perform backtracking line search to find the step size.
    """
    t = 1
    while True:
        new_x = x + t * delta_x
        psi_new = np.sum((new_x - x_cor)**2) + mu * np.sum(np.sqrt(epsilon**2 + (new_x[1:] - new_x[:-1])**2)) - mu * epsilon * (len(new_x) - 1)
        psi_old = np.sum((x - x_cor)**2) + mu * np.sum(np.sqrt(epsilon**2 + (x[1:] - x[:-1])**2)) - mu * epsilon * (len(x) - 1)
        if psi_new <= psi_old + alpha * t * np.dot(gradient(x, x_cor, mu, epsilon), delta_x):
            break
        t *= beta
    return t

def newton_method(x_cor, mu, epsilon, tolerance):
    """
    Perform Newton's method to minimize the objective function.
    """
    x = np.zeros_like(x_cor)
    while True:
        delta_x, grad = newton_step(x, x_cor, mu, epsilon)
        lambda_sq = np.dot(delta_x, grad)
        if lambda_sq / 2 <= tolerance:
            break
        t = line_search(x, x_cor, mu, epsilon, delta_x)
        x += t * delta_x
    return x

def denoise_partition(chunk, mu, epsilon, tolerance):
    """
    Denoise a single partition of the audio signal using Newton's method.
    """
    return newton_method(chunk, mu, epsilon, tolerance)

def denoise_song(input_file, output_file, partitions, mu=50, epsilon=1e-3, tolerance=1e-3):
    """
    Denoise an audio file by partitioning it into chunks, denoising each chunk, and combining the results.
    
    Parameters:
        input_file (str): Path to the noisy input audio file.
        output_file (str): Path to save the denoised audio file.
        partitions (int): Number of chunks to divide the audio file into.
        mu (float): Regularization parameter for total variation.
        epsilon (float): Small parameter for smooth approximation.
        tolerance (float): Stopping criterion for Newton's method.
    """
    # Load the noisy audio file
    audio = AudioSegment.from_file(input_file)
    samples = np.array(audio.get_array_of_samples())
    total_samples = len(samples)
    chunk_size = total_samples // partitions

    denoised_samples = []

    # Process each partition
    for i in range(partitions):
        print(f"Processing chunk {i + 1} of {partitions}...")
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < partitions - 1 else total_samples
        chunk = samples[start:end]
        denoised_chunk = denoise_partition(chunk, mu, epsilon, tolerance)
        denoised_samples.append(denoised_chunk)

    # Combine all denoised chunks
    denoised_samples = np.concatenate(denoised_samples).astype(np.int16)

    # Save the denoised audio file
    denoised_audio = audio._spawn(denoised_samples.tobytes())
    denoised_audio.export(output_file, format="mp3")

# Example usage
denoise_song("song_with_noise.mp3", "song_denoised.mp3", partitions=10, mu=50, epsilon=1e-3, tolerance=1e-3)
