import numpy as np
from scipy.linalg import solve
from pydub import AudioSegment

# Gradient of the approximate total variation function
def grad_phi_atv(x, epsilon):
    n = len(x)
    grad = np.zeros(n)
    for i in range(n - 1):
        diff = x[i + 1] - x[i]
        grad[i] -= diff / np.sqrt(epsilon**2 + diff**2)
        grad[i + 1] += diff / np.sqrt(epsilon**2 + diff**2)
    return grad

# Hessian of the approximate total variation function
def hess_phi_atv(x, epsilon):
    n = len(x)
    diagonals = [np.zeros(n), np.zeros(n - 1), np.zeros(n - 1)]
    for i in range(n - 1):
        diff = x[i + 1] - x[i]
        denom = (epsilon**2 + diff**2)**(3 / 2)
        diagonals[0][i] += epsilon**2 / denom
        diagonals[0][i + 1] += epsilon**2 / denom
        diagonals[1][i] -= epsilon**2 / denom
        diagonals[2][i] -= epsilon**2 / denom
    return np.diag(diagonals[0]) + np.diag(diagonals[1], 1) + np.diag(diagonals[2], -1)

# Objective function
def psi(x, xcor, mu, epsilon):
    norm_term = np.sum((x - xcor)**2)
    atv_term = np.sum(np.sqrt(epsilon**2 + (x[1:] - x[:-1])**2)) - epsilon * (len(x) - 1)
    return norm_term + mu * atv_term

# Denoise a single partition
def denoise_partition(xcor, epsilon, mu, alpha, beta, tol, max_iter):
    x = np.zeros_like(xcor)
    lambda2 = 1
    iter_count = 0
    while lambda2 / 2 > tol and iter_count < max_iter:
        grad = 2 * (x - xcor) + mu * grad_phi_atv(x, epsilon)
        hess = 2 * np.eye(len(x)) + mu * hess_phi_atv(x, epsilon)
        delta_x = solve(hess, -grad)

        # Line search
        t = 1
        while psi(x + t * delta_x, xcor, mu, epsilon) > psi(x, xcor, mu, epsilon) + alpha * t * grad @ delta_x:
            t *= beta

        # Update
        x += t * delta_x
        lambda2 = grad @ solve(hess, grad)
        iter_count += 1

    return x

# Newton's method for song denoising
def denoise_song(input_song_path, output_song_path, partitions=4, epsilon=0.001, mu=50, alpha=0.01, beta=0.5, tol=1e-8, max_iter=100):
    # Load the noisy song
    song = AudioSegment.from_mp3(input_song_path)
    song = song.set_channels(1)  # Convert to mono
    samples = np.array(song.get_array_of_samples(), dtype=np.float32)
    samples /= np.max(np.abs(samples))  # Normalize

    # Calculate partition length
    partition_length = len(samples) // partitions

    denoised_samples = np.zeros_like(samples)
    start = 0

    print(f"Starting Newton's method for song denoising with {partitions} partitions...")
    for i in range(partitions):
        end = min(start + partition_length, len(samples))
        partition = samples[start:end]

        print(f"Processing partition {i + 1}/{partitions} from sample {start} to {end}...")
        denoised_partition = denoise_partition(partition, epsilon, mu, alpha, beta, tol, max_iter)
        denoised_samples[start:end] = denoised_partition

        start = end

    denoised_samples = np.clip(denoised_samples, -1, 1)  # Clip values to [-1, 1]

    # Convert back to 16-bit PCM
    denoised_samples = (denoised_samples * 32767).astype(np.int16)

    # Save the denoised song
    denoised_song = AudioSegment(
        denoised_samples.tobytes(),
        frame_rate=song.frame_rate,
        sample_width=denoised_samples.dtype.itemsize,
        channels=1
    )
    denoised_song.export(output_song_path, format="mp3", bitrate="192k")
    print(f"Denoising complete. Denoised song saved as {output_song_path}.")

# Example usage
denoise_song('song_with_noise.mp3', 'song_denoised.mp3', partitions=16)