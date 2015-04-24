function [patches,label] = testIMAGES( IMAGES,numpatches )
%TESTIMAGES Summary of this function goes here
%IMAGES =  IMAGES_DTest.mat (80*80*4)
% normalize each imgaes 
% pick some patches randomly from those normalized image and mark the label.
% retuen patches and label.
patchsize = 8;  % we'll use 8x8 patches 
num_images = size(IMAGES,3);
pixel = size(IMAGES,2); 

IMAGES_NORM = zeros(size(IMAGES));
patches = zeros(patchsize*patchsize, numpatches);
for i = 1 : num_images
    im = IMAGES(:,:,i);
    [n,m]= size(im);
    im_COPY = reshape(im,[],1);
    im_NORM = normalizeData (im_COPY);
    im_NORM = reshape(im_NORM,n,m);
    IMAGES_NORM(:,:,i)=im_NORM;
end
    
max_index = pixel - patchsize + 1;
label = zeros(1,numpatches); 
for sample= 1 : numpatches
    %randomly pick one of images
    image_index = randi (num_images);
    %randomly sample an 8x8 image patch from the selected image
    x_start = randi(max_index);
    y_start = randi(max_index); 
    %convert the image patch into a 64-dimensional vector  
    patches(:, sample) = reshape(IMAGES_NORM(x_start:(x_start+patchsize-1), y_start:(y_start+patchsize-1), image_index), patchsize*patchsize, 1);  
    label(sample) = image_index;
end;
end

%% ---------------------------------------------------------------
 function IMAGE = normalizeData(IMAGE)
% 
% % Squash data to [0.1, 0.9] since we use sigmoid as the activation
% % function in the output layer
% 
% % Remove DC (mean of images). 
% IMAGE = bsxfun(@minus, IMAGE, mean(IMAGE));
% 
% % Truncate to +/-3 standard deviations and scale to -1 to 1
% pstd = 3 * std(IMAGE(:));
% IMAGE = max(min(IMAGE, pstd), -pstd) / pstd;
% 
% % Rescale from [-1,1] to [0.1,0.9]
% IMAGE = (IMAGE + 1) * 0.4 + 0.1;
IMAGE = bsxfun(@rdivide,IMAGE-min(IMAGE),max(IMAGE)-min(IMAGE));
IMAGE = 0.1+IMAGE*(0.9-0.1);

end

