function [Ibin] = autobin(I)
%BW = imbinarize(I, METHOD) binarizes image I with the threshold method
    %specified using METHOD. Available methods are (names can be
    %abbreviated):
% 'global'    - Global image threshold using Otsu's method, chosen to
             %  minimize the intraclass variance of the thresholded black
              % and white pixels. See GRAYTHRESH for details.
%    b = find(I==0);
%    w = find(I==1);
%    
%    if w > b
%        imbinarize(I,'bright');
%    else
%        imbinarize(I,'dark'); 
%    end

%compute the adequeta threshold level
t = graythresh(I);

%use computed threshold to binarize the image
Ibin = I > t;

%check if the number of white pixels is larger than the 
%number of black pixels. If so, background is white
%and must invert the image

nwp = nnz(Ibin);
nbp = size(I,1)*size(I,2) - nwp; %no of raws times no of col
%nbp = numel(Ibin) - nwp; %equivalence with the one above

if (nwp >nbp)
    Ibin = not(Ibin);
   
end

