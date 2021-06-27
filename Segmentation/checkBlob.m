function isBlob = checkBlob(image)
    % Classify letters as a blob if they are sufficiently round
    % These are probably just noise
    info = regionprops(image, 'Circularity');
    if (min(size(info)) < 1) || (info.Circularity < 0.65)
        isBlob = 0;
    else
        isBlob = 1;
    end
end