function saveImage(image, name, output_size)
    
    %1. Size of the image within the defined range
    if ~all(size(image) < output_size) 
        ratio = max(size(image));
        ratio = 100/ratio;
        image = imresize(image, floor(ratio*size(image)), 'bilinear');
    end
       
    [image, ratio] = centerObject(image, output_size);

    %2. Dots-like objects treated as failure
    if ratio < 0.035 
        path = fullfile(pwd, "output_failures");
    else
        path = fullfile(pwd, "output"); 
    end
    
    image = ~image;
    disp(fullfile(path, name));
    imwrite(image, fullfile(path, name));
end
