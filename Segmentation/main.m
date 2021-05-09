function main(debug)
    files = getInputFiles("input");
    for i = 1:size(files, 1)
        file = fullfile(files(i).folder,files(i).name);
        segmentation(file);
        if debug
            pause;
        end
    end
end