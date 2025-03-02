pkg load image

% Load the image
img = imread('sf.jpeg');

% Convert the image to double for processing
img_double = im2double(img);

% Generate white noise with the same size as the image
noise = randn(size(img_double)) * 0.1; % Adjust the multiplier (0.1) for noise intensity

% Add the noise to the image
img_with_noise = img_double + noise;

% Clip the values to stay within the valid range [0, 1]
img_with_noise = max(min(img_with_noise, 1), 0);

% Convert the image back to uint8 for saving
img_with_noise_uint8 = im2uint8(img_with_noise);

% Save the noisy image
imwrite(img_with_noise_uint8, 'sf_with_noise.jpeg');

% Display the original and noisy images
figure;
subplot(1, 2, 1);
imshow(img);
title('Original Image');

subplot(1, 2, 2);
imshow(img_with_noise_uint8);
title('Image with White Noise');
