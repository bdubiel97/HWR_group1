function isBlob = checkBlob(image)

    % Function for checking if image after resizing is lamed
    info = regionprops(image,'Circularity'); %&& info.Circularity > 0.2
    
    if info.Circularity > 0.65
        isBlob = 1;
    else
        isBlob = 0;
    end
    
end