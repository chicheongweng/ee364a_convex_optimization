% Load the audio package (if not already loaded)  
pkg load audio;  

% Read the input audio file  
[audio, fs] = audioread('song.mp3'); % 'audio' contains the audio data, 'fs' is the sampling rate  

% Generate white noise with the same length as the audio  
noise = 0.20 * randn(size(audio)); % Adjust the multiplier (0.01) to control noise intensity  

% Add the white noise to the original audio  
audio_with_noise = audio + noise;  

% Normalize the audio to prevent clipping  
audio_with_noise = audio_with_noise / max(abs(audio_with_noise));  

% Save the resulting audio to a new file  
audiowrite('song_with_noise.mp3', audio_with_noise, fs);  

% Display a message indicating the process is complete  
disp('White noise added and saved to song_with_noise.mp3'); 
