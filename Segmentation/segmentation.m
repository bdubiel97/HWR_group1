function segmentation(path, file, output_size, showImages)
    %% 1.Reading img
    I = im2double(imread(fullfile(path, file)));
    if showImages
        imshow(I); %imhist(I);
    end

    %% 2.Binarization
    I2 = 1-imbinarize(I); 
    if showImages
        figure(2);
        imshow(I2); title('binarized');
    end

    %% 3.Opening
    I3 = imopen(I2,ones(4)); 
    if showImages
        figure(3); 
        imshow(I3); title('opening');
    end

    %% 5.Separating
    [L,N]=bwlabel(I3); %function for labelling separate binarized objects
    Acc_big = logical(I3*0); %initialization of the accumulated images
    Acc_small = logical(I3*0);

    %literate for all the objects
    for k = 1: N
        %compute the mask of the kth object
        Mk = L == k; 

        %Compute the area of the kth object
        Ak = nnz(Mk);

        %Accumulation
        if Ak > 500 %if object is smaller than 500pixels, treat as a noise and exclude
          Acc_big = or(Acc_big,Mk);
        else
          Acc_small = or(Acc_small,Mk);
        end

        if showImages
            figure(4)
            subplot(1,3,1); imshow(Mk); title(['M' num2str(k)])
            subplot(1,3,2); imshow(Acc_big); title('Acc big' )
            subplot(1,3,3); imshow(Acc_small); title('Acc small' )
            pause(0.1)
        end
    end

    if showImages
        figure(5); imshow(Acc_big); title('Acc big' );
        figure(6); imshow(Acc_small); title('Acc small' );
    end

    %% 6. Opening
    I7 = imopen(Acc_big, ones(5));
    if showImages
        figure(9); imshow(I7); title('Opening 5');
    end

    %% 7. Boundig Box
    info = regionprops(I7,'Boundingbox') ;
    if showImages
        imshow(I7)
        hold on
    end
    
    max_x = -Inf;
    max_y = -Inf;
    for k = 1 : length(info)
        BB = info(k).BoundingBox;
        if k > 1
            bbc = ceil(BB); bbc(4) = bbc(4) + bbc(2)-1; bbc(3) = bbc(3) + bbc(1)-1;
            char = I2(bbc(2):bbc(4), bbc(1):bbc(3));
            max_x = max(size(char, 1), max_x);
            max_y = max(size(char, 2), max_y);
            name = strsplit(file, '.');
            name = strcat(fullfile(name(1), strcat(int2str(k), ".jpg")));
            saveImage(char, name(1), output_size);
        end
        if showImages
            rectangle('Position', [BB(1),BB(2),BB(3),BB(4)],'EdgeColor','r','LineWidth',1) ;
        end
    end
    fprintf("Largest character segmented in this image: %d x %d\n", max_x, max_y);
end