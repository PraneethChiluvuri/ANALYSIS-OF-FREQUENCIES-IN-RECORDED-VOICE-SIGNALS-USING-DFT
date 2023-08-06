% Set the desired recording frequency
recordingFreq = 16000;

% Create an audiorecorder object
recorder = audiorecorder(recordingFreq, 16, 1);

% Print the message to start recording
disp('Starting recording...');

% Start recording
record(recorder);

% Wait for recording to finish
pause(6);

% Stop recording
stop(recorder);

% Print the message to end recording
disp('Recording stopped.');

% Get the recorded audio data
audioData = getaudiodata(recorder);

% Save the recorded audio data to a file in wav format
audiowrite('A B C D E F G H I K L M N O 16000.wav', audioData, recordingFreq);


















% Code for plotting 500Hz and 2000Hz
% Read the audio file
filename ="C:\Users\Praneeth Chiluvuri\OneDrive\Desktop\SnS\Praneeth Sentence 1 16000_denoised.wav";
[x, Fs] = audioread(filename);

% Define the length of the signal
N = length(x);

% Define the time vector
t = (0:N-1)*(1/Fs);

% Define the frequency vector
f = (0:N-1)*(Fs/N);

% Compute the DFT matrix using the formula studied in class
W = exp(-1i*2*pi*(0:N-1)'*(0:N-1)/N);

% Compute the DFT using matrix multiplication
X = W*x;

% Plot the magnitude of the DFT
plot(f, abs(X));
xlabel('Frequency (Hz)');
ylabel('Magnitude');






% code for plotting 16000Hz

% Load audio file
filename = "C:\Users\Praneeth Chiluvuri\OneDrive\Desktop\SnS\Praneeth Sentence 1 16000_denoised.wav";
[x, Fs] = audioread(filename);

% Compute length of audio signal
N = length(x);

% Compute DFT of audio signal
X = zeros(1, N);
for k = 0:N-1
    for n = 0:N-1
        X(k+1) = X(k+1) + x(n+1)*exp(-1i*2*pi*k*n/N);
    end
end

% Compute frequency axis
f = (0:N-1)*Fs/N;

% Plot DFT magnitude vs frequency
plot(f, abs(X));
xlabel('Frequency (Hz)');
ylabel('DFT Magnitude');
xlim([0 Fs/2]);









% Code for Denoising

% Load the audio input
[input, Fs] = audioread("C:\Users\Praneeth Chiluvuri\Downloads\dfghk500.wav");

% Define the parameters for the denoising function
frameSize = 1024; % Size of each analysis frame
hopSize = 256; % Number of samples to hop between frames
nFFT = frameSize; % Size of FFT
nBins = nFFT/2 + 1; % Number of frequency bins

% Apply the denoising function to each frame of the audio input
output = zeros(size(input));
for i = 1:hopSize:(length(input)-frameSize)
    % Extract the current frame
    frame = input(i:i+frameSize-1);
    
    % Perform denoising on the current frame
    % You'll need to replace this with your own denoising function
    % that works for your specific application.
    % Here, we're just copying the input frame to the output frame.
    outputFrame = frame;
    
    % Overlap-add the denoised frame to the output
    output(i:i+frameSize-1) = output(i:i+frameSize-1) + outputFrame;
end

% Save the denoised audio
audiowrite('praneeth sentence 1 - 500Hz.wav', output, Fs);

% Plot the DFT of the original and denoised audio
dft_input = abs(fft(input, nFFT));
dft_output = abs(fft(output, nFFT));

freq_axis = linspace(0, Fs/2, nBins);

plot(freq_axis, dft_input(1:nBins), 'b');
hold on;
plot(freq_axis, dft_output(1:nBins), 'r');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
legend('Input', 'Output');