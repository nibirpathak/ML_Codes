clear all
close all

ima = imread('C1_oligo_1.tif'); % read raw image 
bck = ima;
ima_a= imadjust(ima,[0 0.03]); % adjust contrast 
figure(1)
imshowpair(ima,ima_a,'montage') % show raw and conrast adjusted images 

bck_a = ima_a;

imd = imread('C1_dapi_1.tif'); % read raw image 

figure(2)
[ima_a,rect] = imcrop(ima_a); % cropping the image
[ima] = imcrop(ima,rect);
[imd] = imcrop(imd,rect);
figure(3)
imshowpair(ima,ima_a,'montage')

imgbinary = imbinarize(ima,'adaptive','ForegroundPolarity','bright','Sensitivity',0.56); % convert to binary

figure(4)
imshowpair(ima_a,imgbinary,'montage')

imgbinary = bwareaopen(imgbinary,800,8); % removes reions with less than 400 pixels 
figure(5)
imshowpair(ima_a,imgbinary,'montage') 

I4=imgbinary;
stats = regionprops(I4,'PixelList'); % returns the pixel list of each of the connected regions 
% getting the pixels of interest 
siz=size(I4);
L=size(stats);       % return the size of 'stats' ,number of cells 
N=L(1,1);           % the first element of L contains the number of connected regions 
m = zeros(N,1);
div = zeros(N,1);

% For dapi images 
imgbinary_d = imbinarize(imd,'adaptive','ForegroundPolarity','bright','Sensitivity',0.52); % convert to binary
temp=imgbinary_d;
for i=1:6                           % performs a series of erosion and dilation
SE = strel('square',1);
I3=imdilate(temp,SE);
SE = strel('square',3);
Id=imerode(I3,SE);
temp=Id;
end
imgbinary_d=temp;
imgbinary_d = bwareaopen(imgbinary_d,400,8); % removes reions with less than 400 pixels
figure(7)
imshowpair(imd,imgbinary_d,'montage') 
stats_d = regionprops(imgbinary_d ,'PixelList'); % returns the pixel list of each of the connected regions 


% get the background intensity 
figure(9)
[bck_a_crop,rect] = imcrop(ima_a); % cropping the image
[bck_g_crop] = imcrop(bck,rect);
figure(9)
imshowpair(bck_a_crop,bck_g_crop,'montage')
background=mean2(bck_g_crop);

for i=1:N
   M=size(stats(i).PixelList); % total number of pixels in one connected region
   m(i)= M(1,1);
   s=0;
   s=double(s);
   for j=1:m(i)
       col=stats(i).PixelList(j,1);    % collecting the row and col coordinates for each pixel in the connected region
       row=stats(i).PixelList(j,2);   
       if row>siz(1,1)                 % ensuring that the coordinates do not exceed the original image size
           row=siz(1,1);
       end
       if col>siz(1,2)
           col=siz(1,2);
       end
       if(I4(row,col)==1)
%           div(i)=div(i)+1;                % keep track of number of non-zero pixels
          s = (s+double(ima(row,col)));     % summing up the intensities of all the pixels in one connected region
       end
       
   end
   Intensity(i)= s;
   Avg_in(i)= double(s/m(i));                   % average intensity of one conected region
   Area(i)= m(i)*6.5*6.5*10e-12/(20*20);        % i pixel = 6.5 um , Mag= 20X
end
Ld=size(stats_d);       % return the size of 'stats_d' ,number of nuclei
Nd=Ld(1,1);   % no of nuclei 
Diff=Nd-N;

% BS=size(bck_g_crop);
% Intensity_real= Intensity-background*BS(1)*BS(2); 
Intensity_real= Intensity-background; 
I_per_Area= Intensity_real./Area; % Intensity per unit area 

for i=(length(I_per_Area)+1):(length(I_per_Area)+Diff)
    I_per_Area(i)=0;
end






