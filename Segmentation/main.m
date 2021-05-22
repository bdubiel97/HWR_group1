function main(debug, output_size, showImages)
    files = getInputFiles("input");
    
    if ~exist('output', 'dir')
        mkdir("output");
    end
    if ~exist(fullfile("output_failures"), 'dir')
        mkdir(fullfile("output_failures"));
    end
    
    for i = 1:size(files, 1)
        segmentation(files(i).folder, files(i).name, output_size, showImages);
        if debug
            pause;
        end
    end
end