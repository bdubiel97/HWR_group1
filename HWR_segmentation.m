%% 1.Reading img
I = im2double(imread('P123-Fg002-R-C01-R01-binarized.jpg'));
imshow(I); %imhist(I);

%% 2.Binarization
I2 = autobin(I); 
figure(2);
imshow(I2); title('binarized');

%% 3.Opening
I3 = imopen(I2,ones(4)); 
figure(3); 
imshow(I3); title('opening');

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
    
    figure(4)
    subplot(1,3,1); imshow(Mk); title(['M' num2str(k)])
    subplot(1,3,2); imshow(Acc_big); title('Acc big' )
    subplot(1,3,3); imshow(Acc_small); title('Acc small' )
    
    pause(0.1)
    
end

figure(5); imshow(Acc_big); title('Acc big' );
figure(6); imshow(Acc_small); title('Acc small' );

%% 6. Opening
I7 = imopen(Acc_big, ones(5));
figure(9); imshow(I7); title('Opening 5');

%% 7. Boundig Box
info = regionprops(I7,'Boundingbox') ;
imshow(I7)
hold on
for k = 1 : length(info)
     BB = info(k).BoundingBox;
     rectangle('Position', [BB(1),BB(2),BB(3),BB(4)],'EdgeColor','r','LineWidth',1) ;
end



