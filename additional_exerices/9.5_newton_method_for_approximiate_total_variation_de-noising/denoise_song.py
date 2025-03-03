import os
import numpy as np
from scipy.linalg import solve_banded
from pydub import AudioSegment

# Parameters
mu = 5  # Reduced regularization weight
epsilon = 1e-2  # Increased epsilon for numerical stability
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
        diff = np.clip(diff, -1e6, 1e6)  # Clamp diff to avoid overflow
        denom = np.sqrt(epsilon**2 + diff**2)
        denom = max(denom, 1e-10)  # Prevent division by zero
        #print(f"diff[{i}]: {diff}, denom[{i}]: {denom}")  # Debugging output
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
        diff = np.clip(diff, -1e6, 1e6)  # Clamp diff to avoid overflow
        denom = (epsilon**2 + diff**2)**1.5
        denom = max(denom, 1e-10)  # Prevent division by zero
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
    print("Gradient (first 10):", grad[:10])  # Debugging gradient
    print("Delta x (first 10):", delta_x[:10])  # Debugging delta_x
    return delta_x, grad

def newton_method(x_cor, mu, epsilon, tolerance):
    """
    Perform Newton's method to minimize the objective function.
    """
    x = x_cor.copy()  # Initialize with the noisy signal
    while True:
        delta_x, grad = newton_step(x, x_cor, mu, epsilon)
        lambda_sq = np.dot(delta_x, grad)
        if lambda_sq / 2 <= tolerance:
            break
        t = line_search(x, x_cor, mu, epsilon, delta_x)
        x += t * delta_x
    return x

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

def denoise_partition(chunk, mu, epsilon, tolerance):
    """
    Denoise a single partition of the audio signal using Newton's method.
    """
    return newton_method(chunk, mu, epsilon, tolerance)

def denoise_song(input_file, output_file, partitions, mu=5, epsilon=1e-2, tolerance=1e-3):
    """
    Denoise an audio file by partitioning it into chunks, denoising each chunk, and combining the results.
    """
    # Load the noisy audio file
    audio = AudioSegment.from_file(input_file)
    samples = np.array(audio.get_array_of_samples())
    total_samples = len(samples)
    chunk_size = total_samples // partitions

    # Handle stereo audio (multiple channels)
    channels = audio.channels
    sample_width = audio.sample_width
    frame_rate = audio.frame_rate

    print(f"Input audio properties: frame_rate={frame_rate}, sample_width={sample_width}, channels={channels}")

    if channels > 1:
        samples = samples.reshape((-1, channels))

    denoised_samples = []

    # Process each partition
    for i in range(partitions):
        print(f"Processing chunk {i + 1} of {partitions}...")
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < partitions - 1 else total_samples
        chunk = samples[start:end]
        if channels > 1:
            denoised_chunk = np.array([denoise_partition(chunk[:, ch], mu, epsilon, tolerance) for ch in range(channels)]).T
        else:
            denoised_chunk = denoise_partition(chunk, mu, epsilon, tolerance)
        denoised_samples.append(denoised_chunk)

    # Combine all denoised chunks
    denoised_samples = np.concatenate(denoised_samples, axis=0).astype(np.int16)

    # Debugging: Check the denoised samples
    print("Denoised samples (first 100):", denoised_samples[:100])
    print("Max sample value:", np.max(denoised_samples))
    print("Min sample value:", np.min(denoised_samples))

    # Normalize the samples
    max_val = np.max(np.abs(denoised_samples))
    if max_val > 0:
        denoised_samples = (denoised_samples / max_val * 32767).astype(np.int16)

    # Flatten for mono audio
    if channels == 1:
        denoised_samples = denoised_samples.flatten()

    # Save the denoised audio file
    denoised_audio = AudioSegment(
        denoised_samples.tobytes(),
        frame_rate=frame_rate,
        sample_width=sample_width,
        channels=channels
    )
    denoised_audio.export(output_file, format="mp3")

    # Debugging: Check the output file
    print("Exported file size:", os.path.getsize(output_file))
    print("Denoising complete. Output saved to:", output_file)

# Example usage
denoise_song("song_with_noise.mp3", "song_denoised.mp3", partitions=10, mu=5, epsilon=1e-2, tolerance=1e-3)
