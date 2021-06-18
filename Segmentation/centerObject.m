function [centeredObject, ratio] = centerObject(image, output_size)

% Function for cleaning the object before saving:
%   1st: Centering
%   2nd: Removing objects on the edge (by leaving only the biggest one)

    % 1st: Centering the letter
    [row_indices, col_indices, ~] = find(image==1);
    cog = round([mean(row_indices), mean(col_indices)]);
    to_pad = max(0, min(output_size - size(image), output_size/2 - cog));
    image = padarray(image, to_pad, 0, 'pre');
    image = padarray(image, output_size - size(image), 0, 'post');
        
    % 2nd: Leaving only the biggest object in the imaghe
    centeredObject = bwareafilt(image, 1);
        
    % Calculation of the white to black pixels ratio
    white_pixels = nnz(image);
    black_pixels = nnz(~image);
    ratio = white_pixels/black_pixels;


end