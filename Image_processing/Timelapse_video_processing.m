clear all
close all
addpath('Z:\Research\LCAD Sampling\NFP-Sampling\Time lapse images\03.04.19\C1\Images') 
% Actual code to save individual image files from a video.

% vidfullfile ='C1_MMStack_Pos0.ome.avi';
% vidobj = VideoReader(vidfullfile);
% frames=vidobj.Numberofframes;
% 
% for f=1:frames
%   thisframe=read(vidobj,f);
%   figure(1);imagesc(thisframe);
%   thisfile=sprintf('%d.tif',f);
%   imwrite(thisframe,thisfile);
% end


nFrames = 250;
[filename, pathname] = uigetfile('*.bmp;*.tif', 'Pick an tif-file');
FileName = fullfile(pathname,filename);
info = imfinfo(FileName);                                                   % get info about the TIF
H = info(1).Height;             
W = info(1).Width;
Frame = zeros(H,W,nFrames);                                                 % creating an empty 3D array to store all the frames 
Images_after_median = zeros(H,W,nFrames);
for i=12:nFrames-1
if i<100
    imagename = sprintf('pic0%d.tif',i);
else
    imagename = sprintf('pic%d.tif',i);
end
ima=imread(imagename);
% ima=imadjust(ima,[0 0.6]);                                                % having a constant range of intensities for all the images  
GrayImage=ima;
% GrayImage = rgb2gray(ima);                                                % converting the RGB frame into grayscale 
img=imadjust(GrayImage,[0 1],[0 1]);
% Frame(:,:,i) = im2double(GrayImage);                                        % filling up each element of the 3D array with the respective frames. 
Frame(:,:,i) = img;                                                 % bypassing double concversion
% Images_after_median(:,:,i) = medfilt2(Frame(:,:,i),[2 2]);                % Applying a median filter for spatial smoothing  
Images_after_median(:,:,i)=Frame(:,:,i);                                    % No median filter
end
T=[-2 -1 0 1 2];
sigma = 1;                     
for i=1:5
    G(i)=(1/(sqrt(2*pi)*sigma))*exp(-(T(i)-T(3))^2/(2*sigma^2));            % Components of the Gaussian filter with sigma =1,spanning over five images
end
Images_after_median_gauss= zeros(H,W,nFrames);        
for i=14:nFrames-3
    Images_after_median_gauss(:,:,i)=G(1)*Images_after_median(:,:,i-2) + G(2)*Images_after_median(:,:,i-1) + G(3)*Images_after_median(:,:,i) + G(4)*Images_after_median(:,:,i+1) + G(5)*Images_after_median(:,:,i+2);  % Temporal smoothing using Gaussian Filter
    %Images_after_median_gauss(:,:,i)=Images_after_median(:,:,i);           % to by pass the guassian filter.
end

figure(1)
subplot(1,2,1)
imshow(Frame(:,:,12),[])
subplot(1,2,2)
imshow(Images_after_median(:,:,12),[])

figure(2)
for w=1:6
    subplot(2,3,w)
    if w<=3
        imshow(Frame(:,:,w+13),[])
    else
        imshow(Images_after_median_gauss(:,:,w+10),[])                        % displays the first 3 images of the sequence along with the gaussian filtered version of those
    end
end    
   
avg = zeros(nFrames-16,3);
sum = zeros(nFrames-16,3);                                       
for z=1:3
    if z==1
        th=1520;
    elseif z==2 
        th=1520;
    else 
        th=980;
    end
    imgbinary=imbinarize(Images_after_median_gauss(:,:,14),th);               % converts the first image into binary using a threshold of 0.265
    if z==3
        imgbinary=imcomplement(imgbinary);
    end
    temp=imgbinary;
    for i=1:3                                                               % performs a series of erosion and dilation
    SE = strel('square',2);
    I3=imdilate(temp,SE);
    SE = strel('square',3);
    Imgbinary_after_ero_dilo=imerode(I3,SE);
    temp=Imgbinary_after_ero_dilo;
    end
    
    figure(4)
    subplot(1,2,1);
    imshow(imgbinary,[])                                                    % displays the binary image and the image after a series of erosion and dilation
    title('Binary Image')
    subplot(1,2,2);
    imshow(Imgbinary_after_ero_dilo,[])

    imgcon = bwselect(Imgbinary_after_ero_dilo,4);                          % allows the user to select regions that the user thinks to be cells/ region of interest 
    imgcon_double = im2double(imgcon);                                      % converting the mask having the pixels of interest into double 
    final=times(imgcon_double,Images_after_median_gauss(:,:,14));            % element wise multiplication to convert the original frame to have only the ROI
    figure(5)
    subplot(1,2,1)
    imshow(imgcon,[])                                                       % shows the regions deemed to be cells with other unwanted regions removed  
    title('Connected regions')
    subplot(1,2,2)
    imshow(final,[])

    siz=size(Images_after_median_gauss(:,:,14));
    k=1;
    for i=1:siz(1,1)
        for j=1:siz(1,2)
            if(final(i,j)~=0)
                p(k)=i;q(k)=j;
                k=k+1;
            end
        end
    end
 
    n=k-1;
    
    for i=14:nFrames-3
        s=0;
        s=double(s);
        final=times(imgcon_double,Images_after_median_gauss(:,:,i));         % element wise multiplication to convert the original frame to have only the ROI
        for j=1:n
            s=s+final(p(j),q(j));
        end
        sum(i-13,z)=s;
        avg(i-13,z)=double(s/n);
    end
end
t=4.4:0.3:74.3;
avg_control=avg(:,1)./avg(1,1);                                             % Normalizing all the data points by the firsr data point 
sum_control=sum(:,1)./sum(1,1);

avg_cell=avg(:,2)./avg(1,2);
sum_cell=sum(:,2)./sum(1,2);

cell_control = avg_cell-avg_control;
avg_ref=avg(:,3)./avg(1,3);

figure(6)
subplot(2,2,1)
plot(t,avg_control,'b','Linewidth',2);

xlabel('Time');
ylabel('Normalized Mean Intensity(control)')

x0=1; y0=1; width=6.0; height=5.0;
set(gcf,'units','inches','position',[x0,y0,width,height])

%Font size
set(gca,'fontsize',8,'FontName','Arial')
subplot(2,2,2)
plot(t,avg_cell,'r','Linewidth',2);

xlabel('Time');
ylabel('Normalized Mean Intensity(cell)')

x0=1; y0=1; width=6.0; height=5.0;
set(gcf,'units','inches','position',[x0,y0,width,height])

%Font size
set(gca,'fontsize',8,'FontName','Arial')

subplot(2,2,3)
plot(t,cell_control,'k','Linewidth',2);

xlabel('Time');
ylabel('Cell-Control')

x0=1; y0=1; width=6.0; height=5.0;
set(gcf,'units','inches','position',[x0,y0,width,height])

%Font size
set(gca,'fontsize',8,'FontName','Arial')

subplot(2,2,4)
plot(t,avg_ref,'g','Linewidth',2);

xlabel('Time');
ylabel('No cell region')

x0=1; y0=1; width=6.0; height=5.0;
set(gcf,'units','inches','position',[x0,y0,width,height])

%Font size
set(gca,'fontsize',8,'FontName','Arial')

M=movmean(cell_control,6);
figure(7)
plot(t,M,'r','Linewidth',2);xlabel('Time(s)');
ylabel('Normalized Mean Intensity(I)');x0=1; y0=1; width=6; height=5;
set(gcf,'units','inches','position',[x0,y0,width,height]);set(gca,'fontsize',16,'FontName','Arial')
