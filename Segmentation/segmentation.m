function segmentation(path, file, output_size, showImages)
    %% 1.Reading img
    I = im2double(imread(fullfile(path, file)));
    I_width = size(I,1); %new
    I_height = size(I,2); %new
    TotalPixels =  I_width * I_height; %new
    Threshold = TotalPixels* 0.0005633; %new
    if showImages
        imshow(I); 
    end

    %% 2.Binarization
    I2 = 1-imbinarize(I); 
    if showImages
        figure(2);
        imshow(I2); title('binarized');
    end

    %% 3.Opening 
    I3 = imopen(I2,ones(5)); 
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
        if Ak > Threshold  %new
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

    %% 5. Erode
    se = [0,1,1;1,1,1;0,1,0];
	I4 = imerode(Acc_big, se);
    
    if showImages
        figure(9); imshow(I4); title('Erode with se (3x3)');
    end

    %% 6. 1st Saving (merged vs clean)
    info = regionprops(I4,'Boundingbox','Area','Perimeter','Centroid') ;
    [L,N]=bwlabel(I4);
    AccMerged = logical(I4*0);
    AccSegmented = logical(I4*0);
    
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
        if info(k).FormFactor < 0.21
            %Merged letters
            AccMerged = or(AccMerged, Mk);
       
        else 
            %Segmented letters
            AccSegmented = or(AccSegmented, Mk);
            info(k).image = imcrop(I4,info(k).BoundingBox); 
            info(k).image = imdilate(info(k).image, se);
            name = strsplit(file, '.');
            coords = sprintf("-x=%.0f-y=%.0f-h=%d", info(k).BoundingBox(1), info(k).BoundingBox(2), info(k).BoundingBox(3));
            name = strcat(fullfile(name(1), strcat(int2str(k), "-clear", coords, ".jpg")));
            saveImage(info(k).image, name(1), output_size);
        end
                
        if showImages
            rectangle('Position', [BB(1),BB(2),BB(3),BB(4)],'EdgeColor','r','LineWidth',1) ;
        end
    end
    %% 7. Dilate
      
	AccMerged = imdilate(AccMerged, se);
    if showImages
        figure(9); imshow(AccMerged); title('Dilate with se (3x3)');
    end
    %% 8. 2nd saving (lamed vs rest)
    info2 = regionprops(AccMerged,'Boundingbox','Area','Perimeter','Centroid');
    for k = 1: length(info2)
        
        BB2 = info2(k).BoundingBox;
        info2(k).image = imcrop(AccMerged,info2(k).BoundingBox);%cropping bb for each obj
        X = info2(k).image;
        height = size(X,1);
        width =  size(X,2);
        ratio = width/height;
        
        %Rotation of the images with lamed
        if ratio < 0.9 %WAS
           
            %CALC: CentrumOfGravity + Centroid
            info3 = regionprops(X,'Centroid'); 
            [row_indices, col_indices, ~] = find(X==1);
            cog = round([mean(row_indices), mean(col_indices)]); 
            cntr = info3.Centroid; 
            
            %Finding the angle
            coefficients = polyfit([cntr(1), cog(1)], [cntr(2), cog(2)], 1);
            a = abs(coefficients(1));
            angle = -a*180/pi;
           
            %Rotation by angle
            X=imrotate(X,angle);
            
        end
        
        %Horizontal and Vertical profile of the image
        [rows, columns] = size(X); 
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
        coords = sprintf("-x=%.0f-y=%.0f-h=%d", info2(k).BoundingBox(1), info2(k).BoundingBox(2), info2(k).BoundingBox(3));
        nameL = strcat(fullfile(name(1), strcat(int2str(k), "-L", coords, ".jpg")));
        nameR = strcat(fullfile(name(1), strcat(int2str(k), "-R", coords, ".jpg")));
        saveImage(I_segmented_L, nameL(1), output_size); 
        saveImage(I_segmented_R, nameR(1), output_size); 
    end
end
