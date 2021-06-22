function [files] = getInputFiles(directory)
    files = dir(directory);
    files = files(3:end, :);
    disp("Loading:");
    for i = 1:size(files, 1)
        disp(files(i).name);
    end
end