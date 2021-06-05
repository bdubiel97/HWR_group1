function saveImage(image, name, output_size)
    if all(size(image) < output_size)
        path = fullfile(pwd, "output");
        [row_indices, col_indices, ~] = find(image==1);
        cog = round([mean(row_indices), mean(col_indices)]);
        to_pad = max(0, min(output_size - size(image), output_size/2 - cog));

        image = padarray(image, to_pad, 0, 'pre');
        image = padarray(image, output_size - size(image), 0, 'post');
        image = ~image;
        
        disp(fullfile(path, name));
        imwrite(image, fullfile(path, name));
    else
        path = fullfile(pwd, "output_failures");
        image = ~image;
        imwrite(image, fullfile(path, name));
        fprintf("Image %s too big: %d %d\n", name, size(image, 1), size(image, 2));
    end
end
