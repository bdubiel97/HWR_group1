function isBlob = checkBlob(image)
    % Classify letters as a blob if they are sufficiently round
    % These are probably just noise
    info = regionprops(image, 'Circularity');
    if (min(size(info)) < 1) || (info.Circularity < 0.65)
        isBlob = false;
    else
        isBlob = true;
    end
end