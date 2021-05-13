function saveImage(image, name, output_size)
    path = fullfile(pwd, 'output');
    name = strcat(name, '.jpg');
    if all(size(image) < output_size)
        [row_indices, col_indices, ~] = find(image==1);
        cog = round([mean(row_indices), mean(col_indices)]);
        to_pad = max(0, min(output_size - size(image), output_size/2 - cog));

        image = padarray(image, to_pad, 0, 'pre');
        image = padarray(image, output_size - size(image), 0, 'post');
        
        imwrite(image, fullfile(path, name));
    else
        imwrite(image, fullfile(path, 'failures', name));
        fprintf("Image %s too big: %d %d\n", name, size(image, 1), size(image, 2));
    end
end