function isLamed = checkLamed(image)

    % Function for checking if image after resizing is lamed
    info = regionprops(image,'Eccentricity','Circularity'); %&& info.Circularity > 0.2
    
    if info.Eccentricity > 0.93 
        isLamed = 1;
    else
        isLamed = 0;
    end
    
end