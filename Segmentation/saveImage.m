function saveImage(image, name, output_size)
    
    %1. Size of the image within the defined range
    if all(size(image) < output_size) 
        [image, ratio] = centerObject(image, output_size);
        
        %2. Dots-like objects treated as failure
        if ratio < 0.035 
            path = fullfile(pwd, "output_failures");
        else
            path = fullfile(pwd, "output");
        end
            
    %1. Size of the image extending the defined range  
    else 
        image = imresize(image, 0.5, 'bilinear');
        
        %2. Size of the image after resizing within the defined range
        if (output_size - size(image)) > 0 
            [image, ratio] = centerObject(image, output_size);
            
            %3. Dots-like objects treated as failure
            if ratio < 0.035 
                path = fullfile(pwd, "output_failures");

            else
                isLamed = checkLamed(image); 
                %4. Saving "lamed", the rest excluded
                if isLamed == 1
                    path = fullfile(pwd, "output"); 
                else
                    path = fullfile(pwd, "output_failures");
                end

            end

        %2. Size of the image after resizing extending the defined range
        else 
           path = fullfile(pwd, "output_failures"); 
            
        end
        
    end
    image = ~image;
    disp(fullfile(path, name));
    imwrite(image, fullfile(path, name));

    
end
