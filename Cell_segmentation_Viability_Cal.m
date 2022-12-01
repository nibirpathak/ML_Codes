clear all
close all
clc
IB=imread('dapi_3.tif');
IR=imread('cal_3.tif'); %image input
% Adjust contrast
IB_a = imadjust(IB,[0 0.5]);
IR_a = imadjust(IR,[0 0.1]);
figure(1)
imshowpair(IB_a,IR_a,'montage')

% crop images 
figure(2)
[temp1,rect] = imcrop(IR_a);
[IB] = imcrop(IB,rect);
[IR] = imcrop(IR,rect);
[IB2] = imcrop(IB_a,rect);
[IR2] = imcrop(IR_a,rect);
% show croopped images
figure(3)
imshowpair(IB2,IR2,'montage')

% binarize image
%IR_b = imbinarize(IR);
IR_b=imbinarize(IR,'adaptive','ForegroundPolarity','bright','Sensitivity',0.55); % for 2e6
%Remove regions with less than 1000 pixels
imgR_b=bwareaopen(IR_b,800,8);

stats1 = regionprops(imgR_b,'PixelList');
L1=size(stats1); % return the size of 'stats' 
N1=L1(1,1); %number of cells alive

figure(4)
imshow(IB2,[])
cells =0;
locations = zeros(1000, 2);

   while cells <1000
   [x,y]=ginput(1);
   if isempty(x)
       break;
   end
   hold on;
   plot(x, y, 'r+', 'MarkerSize', 5, 'LineWidth', 1);
   text(x+5, y+5, num2str(cells+1),'Color','red','FontSize',14);
   locations(cells+1, 1) = x;
   locations(cells+1, 2) = y;
   cells =cells+1;
   end
   
figure(5)
imshow(IR2,[])
   
cells2 =0;
locations2 = zeros(1000, 2);

   while cells2 <1000
   [x,y]=ginput(1);
   if isempty(x)
       break;
   end
   hold on;
   plot(x, y, 'r+', 'MarkerSize', 5, 'LineWidth', 1);
   text(x+5, y+5, num2str(cells2+1),'Color','red','FontSize',14);
   locations2(cells2+1, 1) = x;
   locations2(cells2+1, 2) = y;
   cells2 =cells2+1;
   end 
figure(6)
subplot(1,2,1)
imshow(IR2,[])                      
title('mch raw image')
subplot(1,2,2)
imshow(IR_b,[])                       
title('Bin mch image')


% display the raw and binary images of DAPI and Mch after noise reduction
figure(7)
subplot(1,2,1)
imshow(IR2,[])                      
title('mch raw image')
subplot(1,2,2)
imshow(imgR_b,[])                       
title('Bin mch image')


IB_b=imbinarize(IB,'adaptive','ForegroundPolarity','bright','Sensitivity',0.50);
imgB_b=bwareaopen(IB_b,500,8);
temp=imgB_b;
 
% for i=1:20                          % performs a series of erosion and dilation
% % SE = strel('square',1);
% % I3=imdilate(temp,SE);
% SE = strel('square',1);
% I4=imerode(temp,SE);
% temp=I4;
% end
% imgB_b=temp;

stats2 = regionprops(imgB_b,'PixelList');
L2=size(stats2); % return the size of 'stats' 
N2=L2(1,1); %number of nuclei

figure(8)
subplot(1,2,1)
imshow(IB2,[])                      
title('mch raw image')
subplot(1,2,2)
imshow(imgB_b,[])                       
title('Bin mch image')


siz=size(imgB_b);
count=0;
for i=1:N2
   M=size(stats2(i).PixelList); % total number of pixels in one connected region
   m(i)= M(1,1);
   s1=0;
   s1=double(s1);
   s2=0;
   s2=double(s2);
   cc=0;
   for j=1:m(i)
       col=stats2(i).PixelList(j,1);    % collecting the row and col coordinates for each pixel in the connected region
       row=stats2(i).PixelList(j,2);   
       if row>siz(1,1)                 % ensuring that the coordinates do not exceed the original image size
           row=siz(1,1);
       end
       if col>siz(1,2)
           col=siz(1,2);
       end
       if(imgB_b(row,col)==1 && imgR_b(row,col)==1) % checking for colocalization
           cc=cc+1; 
       end
       
   end
   if (cc>0.4*m(i))  % setting a threshold of 80% colocalization
       count=count+1;
   end
                      
end
Viability=count/N2*100;
sprintf('Viablity= %f', Viability)
Viability_count=N1/N2*100;
sprintf('Viability_count= %f', Viability_count)
