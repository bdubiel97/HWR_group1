function saveImage(image, name, output_size)
    %1. Size of the image within the defined range
    ratio = max(size(image));
    ratio = 0.8*output_size/ratio;
    image = imresize(image, floor(ratio.*size(image)), 'bilinear');
       
    [image, ratio] = centerObject(image, output_size);
    isBlob = checkBlob(image);

    %2. Dots-like objects treated as failure
    if ((0.7 < ratio) || (ratio < 0.035)) || isBlob
        path = fullfile(pwd, "output_failures");
    else
        path = fullfile(pwd, "output"); 
    end
    
    % Make sure image is a black letter on a white background
    image = ~image;
    
    % Write segmented image to file
    disp(fullfile(path, name));
    imwrite(image, fullfile(path, name));
end
