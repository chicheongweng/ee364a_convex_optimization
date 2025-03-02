import numpy as np  
from scipy.sparse import diags  
from scipy.linalg import solve  
from PIL import Image  
import matplotlib.pyplot as plt  

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
    return diags(diagonals, [0, 1, -1]).toarray()  

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

# Newton's method with overlapping partitions for image denoising  
def denoise_image(input_image_path, output_image_path, partitions=4, epsilon=0.001, mu=50, alpha=0.01, beta=0.5, tol=1e-8, max_iter=100):
    # Load and flatten the noisy image  
    img_cor = np.array(Image.open(input_image_path).convert('L')) / 255.0  # Convert to grayscale and normalize  
    height, width = img_cor.shape  

    # Calculate partition width and overlap based on the number of partitions
    partition_width = width // partitions
    overlap = max(partition_width // 8, 1)  # Set overlap as 1/8th of partition width, with a minimum of 1 pixel

    # Partition the image into overlapping vertical slices  
    denoised_image = np.zeros_like(img_cor)  
    weight_matrix = np.zeros_like(img_cor)  # To handle blending  
    start = 0  
    iteration = 0

    print(f"Starting Newton's method for image denoising with {partitions} partitions...")  
    while start < width:
        iteration = iteration + 1
        print(f"Iteration {iteration}: Processing partition...")  # Print current iteration number
        end = min(start + partition_width, width)  
        if start > 0:  
            start_overlap = start - overlap  
        else:  
            start_overlap = start  

        if end < width:  
            end_overlap = end + overlap  
        else:  
            end_overlap = end  

        # Extract the partition with overlap  
        partition = img_cor[:, start_overlap:end_overlap].flatten()  

        print(f"Processing partition from column {start_overlap} to {end_overlap}...")  
        denoised_partition = denoise_partition(partition, epsilon, mu, alpha, beta, tol, max_iter)  
        denoised_partition = denoised_partition.reshape((height, end_overlap - start_overlap))  

        # Add the denoised partition to the result  
        denoised_image[:, start_overlap:end_overlap] += denoised_partition  
        weight_matrix[:, start_overlap:end_overlap] += 1  # Track how many times each pixel is updated  

        start = end  

    # Normalize by the weight matrix to blend overlapping regions  
    denoised_image /= weight_matrix  
    denoised_image = np.clip(denoised_image, 0, 1)  # Clip values to [0, 1]  

    # Save the denoised image  
    denoised_image_uint8 = (denoised_image * 255).astype(np.uint8)  # Convert to 8-bit grayscale  
    Image.fromarray(denoised_image_uint8).save(output_image_path, format='JPEG', quality=95)  
    print(f"Denoising complete. Denoised image saved as {output_image_path}.")  

    # Plot the original and denoised images  
    plt.figure(figsize=(10, 5))  
    plt.subplot(1, 2, 1)  
    plt.imshow(img_cor, cmap='gray')  
    plt.title('Noisy Image')  
    plt.axis('off')  

    plt.subplot(1, 2, 2)  
    plt.imshow(denoised_image, cmap='gray')  
    plt.title('Denoised Image')  
    plt.axis('off')  

    plt.show()  

# Example usage  
denoise_image('sf_with_noise.jpeg', 'sf_denoised.jpeg', partitions=64)
