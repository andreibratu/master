function ex1_1(image_path)
    % Describe the image founded at given image path.

    % Meta information about photo
    % imfinfo(image_path)

    % Size of image
    s = dir(image_path);
    sprintf("Size of the image in bytes: %i", s.bytes)

    % Display the image      
    read_image = imread(image_path);
    figure("Name", "RGB Image");
    imshow(read_image);

    % Image to grayscale
    grayscale = rgb2gray(read_image);
    figure("Name", "Grayscale Image"); % Create new figure for grayscale image
    imshow(grayscale)

    % Min max values of grayscale image
    minV = min(reshape(grayscale, 1, []));
    maxV = max(reshape(grayscale, 1, []));
    sprintf("Min and max value in grayscale image: %i %i", minV, maxV)

    % Gaussian smoothing filter
    for smoothV = 2:2:8
        smoothI = imgaussfilt(grayscale, smoothV);
        figure("Name", sprintf("Gaussian grayscale with STD %i", smoothV));
        imshow(smoothI);
    end

    

