function main(debug, output_size, showImages)
    files = getInputFiles("input");
    
    % Create output folders when necessary
    if ~exist('output', 'dir')
        mkdir("output");
    end

    if ~exist(fullfile("output_failures"), 'dir')
        mkdir(fullfile("output_failures"));
    end
    
    for i = 1:size(files, 1)
        name = strsplit(files(i).name, '.');
        folder = fullfile("output", name(1));
        if ~exist(fullfile(folder), 'dir')
            mkdir(fullfile(folder));
        end

        folder = fullfile("output_failures", name(1));
        if ~exist(fullfile(folder), 'dir')
            mkdir(fullfile(folder));
        end
        
        segmentation(files(i).folder, files(i).name, output_size, showImages);
        if debug
            pause;
        end
    end
end