function main(input_folder, output_folder, output_size, debug, showImages)
    input_folder = string(input_folder);
    output_folder = string(output_folder);
    files = getInputFiles(input_folder);
    
    % Create output folders when necessary
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end
    
    for i = 1:size(files, 1)
        name = strsplit(files(i).name, '.');
        folder = fullfile(output_folder, name(1));
        if ~exist(fullfile(folder), 'dir')
            mkdir(fullfile(folder));
        end
        
        segmentation(files(i).folder, files(i).name, output_size, debug, showImages);
        if debug
            pause;
        end
    end
end