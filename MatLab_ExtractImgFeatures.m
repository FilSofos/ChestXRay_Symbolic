%% Feature Extraction for Pneumonia/Normal Images
% Reads X-rays, calculates features, and exports to CSV.

clear; clc;

% Suppose the folder NORMAL contains all NORMAL X-Ray images
% Do exactly the same for folder PNEUMONIA
dataDir = 'NORMAL'; 
outputFile = 'normal_test2.csv';
fileSpecs = dir(fullfile(dataDir, 'NORMAL*.jp*g')); % Handles normal*.jpg and .jpeg
%fileSpecs = dir(fullfile(dataDir, 'BACTERIA*.jp*g')); % Handles bacteria*.jpg and .jpeg
%fileSpecs = dir(fullfile(dataDir, 'VITUS*.jp*g')); % Handles virus*.jpg and .jpeg


numFiles = length(fileSpecs);

fprintf('Processing %d images...\n', numFiles);
tic;

numFeatures = 11;
results = cell(numFiles, numFeatures); 

for i = 1:numFiles
    imgName = fileSpecs(i).name;
    imgPath = fullfile(dataDir, imgName);
    
    % GEOMETRY & QUALITY CHECK
    info = imfinfo(imgPath);
    aspectRatio = info.Width / info.Height;
    
    % Skip stretched images
    if aspectRatio < 0.7 || aspectRatio > 1.5
        fprintf('Skipping %s: Stretched Aspect Ratio (%.2f)\n', imgName, aspectRatio);
        continue;
    end
    
    I_raw = imread(imgPath);
    if size(I_raw, 3) == 3, I_raw = rgb2gray(I_raw); end
    
    % Skip extremely dark or "flat" images (Poor exposure)
    if mean(I_raw(:)) < 10 || std(double(I_raw(:))) < 5
        fprintf('Skipping %s: Poor Image Quality/Contrast\n', imgName);
        continue;
    end

    % Resize all images to 1024X1024
    I = im2double(imresize(I_raw, [1024 1024]));
    
    % Lung Masking - internal lung region
    rows = floor(0.15*1024):floor(0.85*1024);
    cols = floor(0.10*1024):floor(0.90*1024);
    I_focused = I(rows, cols);
    I_focused = adapthisteq(I_focused, 'ClipLimit', 0.02); 
    
    % DENSITY & HAZE
    threshold = graythresh(I_focused);
    haze_mask = (I_focused > (threshold * 0.7)) & (I_focused < (threshold * 1.2));
    hazeRatio = sum(haze_mask(:)) / numel(I_focused);
    
    % Solidity
    solidityRatio = sum(I_focused(:) > 0.85) / numel(I_focused);

    % Standard Features
    E_map = entropyfilt(I_focused);
    entropyMax = max(E_map(:)); 
    entropy90th = prctile(E_map(:), 90);
    
    glcm = graycomatrix(I_focused, 'Offset', [0 1], 'Symmetric', true);
    stats = graycoprops(glcm, {'Contrast', 'Homogeneity'});
    
    BW = edge(I_focused, 'canny'); 
    fractalDim = get_fractal_dim(BW);

    results(i, :) = {imgName, mean(E_map(:)), entropyMax, entropy90th, ...
                     hazeRatio, solidityRatio, stats.Contrast, ...
                     stats.Homogeneity, skewness(I_focused(:)), ...
                     kurtosis(I_focused(:)), fractalDim};
end

% Write table
T = cell2table(results, 'VariableNames', ...
    {'Filename', 'EntropyMean', 'EntropyMax', 'Entropy90', 'Haze', ...
     'Solidity','Contrast', 'Homogeneity', 'Skewness', 'Kurtosis', 'FractalDim'});
writetable(T, outputFile);

elapsedTime = toc;
fprintf('Done! %d images in %.2f seconds.\n', numFiles, elapsedTime);

function D = get_fractal_dim(BW)
    BW = logical(BW);
    [h, w] = size(BW);
    
    % This is the standard for 1024x1024 images
    steps = 2.^(1:8); 
    N = zeros(size(steps));
    
    for k = 1:length(steps)
        s = steps(k);
        fun = @(block_struct) any(block_struct.data(:));
        reduced = blockproc(BW, [s s], fun);
        N(k) = sum(reduced(:));
    end
    
    % Remove any zeros to avoid log(0) which leads to NaNs
    valid = N > 0;
    if sum(valid) > 1
        % Linear regression on the log-log plot
        % The slope is the Minkowski-Bouligand Dimension
        x = log(1./steps(valid));
        y = log(N(valid));
        coeffs = polyfit(x, y, 1);
        D = coeffs(1);
    else
        D = 0;
    end
end