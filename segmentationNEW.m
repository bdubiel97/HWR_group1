function segmentationNEW(path, file, output_size, showImages)
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

    %% 3.Pre-processing (opening for now)
    I3 = imopen(I2,ones(4)); 
    if showImages
        figure(3); 
        imshow(I3); title('opening');
    end

    %% 4.Discarding noise-like objects
    [L,N]=bwlabel(I3);              %function for labelling separate binarized objects
    Acc_big = logical(I3*0);        %initialization of the accumulated images
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

    %% 5. Post-processing (opening for now)
    I4 = imopen(Acc_big, ones(5));
    if showImages
        figure(9); imshow(I4); title('Opening 5');
    end

    %% 6. 1st Saving (merged vs clean)
    info = regionprops(I4,'Boundingbox','Area','Perimeter','Centroid') ;
    [L,N]=bwlabel(I4);
    AccMerged = logical(I4*0);
    AccSegmented = logical(I4*0);
    count_merged = 0;
    count_clean = 0;
    
    if showImages
        imshow(I4)
        hold on
    end
    
    for k = 1 : length(info)
        
        BB = info(k).BoundingBox;
        info(k).image = imcrop(I4,info(k).BoundingBox);               %cropping bb for each obj
        info(k).FormFactor = 4*pi*info(k).Area / info(k).Perimeter^2; %computing ff for each obj
        Mk = L == k;
        
        %Segmentation of clear and merged letters
        if info(k).FormFactor < 0.2
            %Merged letters
            AccMerged = or(AccMerged, Mk);
            count_merged = count_merged + 1;
       
        else 
            %Segmented letters
            AccSegmented = or(AccSegmented, Mk);
            count_clean = count_clean + 1;
            info(k).image = imcrop(I4,info(k).BoundingBox); %cropping bb for each obj
            name = strsplit(file, '.');
            name = strcat(fullfile(name(1), strcat(int2str(k), "-clear", ".jpg")));
            saveImage(info(k).image, name(1), output_size); %CHAR IS THE IMG
            
        end
                
        if showImages
            rectangle('Position', [BB(1),BB(2),BB(3),BB(4)],'EdgeColor','r','LineWidth',1) ;
        end
    end
    %% 7. 2nd saving (lamed vs rest)
    info2 = regionprops(AccMerged,'Boundingbox','Area','Perimeter','Centroid');
    count_lamed = 0;
    for k = 1: length(info2)
        
        BB2 = info2(k).BoundingBox;
        info2(k).image = imcrop(AccMerged,info2(k).BoundingBox);%cropping bb for each obj
        X = info2(k).image;
        height = size(X,1);
        width =  size(X,2);
        ratio = width/height;
        
        %Rotation of the images with lamed
        if ratio < 0.95
           
            count_lamed = count_lamed + 1;
            %CALC: CentrumOfGravity + Centroid
            info3 = regionprops(X,'Centroid'); %NEW
            [row_indices, col_indices, ~] = find(X==1); %NEW
            cog = round([mean(row_indices), mean(col_indices)]); %NEW
            cntr = info3.Centroid; %NEW
            
            %Finding the angle
            coefficients = polyfit([cntr(1), cog(1)], [cntr(2), cog(2)], 1);
            a = abs(coefficients(1));
           
            %Rotation by angle
            X=imrotate(X,-a*180/pi);
            
        end
        
        %Horizontal and Vertical profile of the image
        [rows, columns] = size(X); %OLD
        verticalProfile = sum(X, 2);
        horizontalProfile = sum(X, 1);
        amplitude = 1; % Scaling factor
        y = rows - amplitude * horizontalProfile;
        
        %Finding the min. value for intersection
        x1 = round(size(horizontalProfile,2)* (0.2));
        x2 = round(size(horizontalProfile,2)* (0.8));
        y_temp =y(:,x1:x2);
        [ymax, idx] = max(y_temp);
        x_coords = idx + x1;
        
        %Dividing the image into two parts ([xmin ymin width height];)
        I_segmented_L = imcrop(X, [0,0,x_coords,size(verticalProfile,1)]);
        I_segmented_R = imcrop(X, [x_coords,0,size(horizontalProfile,2)- x_coords,size(verticalProfile,1)]);
        
        %Saving splitted images
        name = strsplit(file, '.');
        nameL = strcat(fullfile(name(1), strcat(int2str(k), "-L", ".jpg")));
        nameR = strcat(fullfile(name(1), strcat(int2str(k), "-R", ".jpg")));
        saveImage(I_segmented_L, nameL(1), output_size); 
        saveImage(I_segmented_R, nameR(1), output_size); 
    end
    
    
    
    
    
    
end