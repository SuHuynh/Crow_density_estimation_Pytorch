%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File to create training and validation set       %
% for ShanghaiTech Dataset Part A and B. 10% of    %
% the training set is set aside for validation     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear all;
seed = 95461354;
rng(seed)
N = 1;
dataset = 'A';
% dataset_name = ['shanghaitech_part_' dataset '_patches_' num2str(N)];
%path = ['../data/original/shanghaitech/part_' dataset '_final/train_data/images/'];
path = ['D:\Su\UAV_project_density\data_test\part_A\test_images\'];
gt_path = ['D:\Su\UAV_project_density\dataset\ShanghaiTech\part_' dataset '\test_data\ground-truth\'];
output_path = 'D:\Su\UAV_project_density\data_test\part_A\';

% train_path_img = strcat(output_path,'\test_images\');
train_path_den = strcat(output_path,'\test_den\');
% val_path_img = strcat(output_path, dataset_name,'\val\');
% val_path_den = strcat(output_path, dataset_name,'\val_den\');
% train_path_den_final = strcat(output_path, dataset_name,'\train_den_images\');

mkdir(output_path)
% mkdir(train_path_img);
mkdir(train_path_den);
% mkdir(val_path_img);
% mkdir(val_path_den);
% mkdir(train_path_den_final);
if (dataset == 'A')
    num_images = 182;
else
    num_images = 316;
end
% num_val = ceil(num_images*0.1);
indices = randperm(num_images);

for idx = 1:num_images
    i = indices(idx);
    if (mod(idx,1)==0)
        fprintf(1,'Processing %3d/%d files\n', i, num_images);
    end
    
    
    load(strcat(gt_path, 'GT_IMG_',num2str(i),'.mat')) ;
    input_img_name = strcat(path,'IMG_',num2str(i),'.jpg');
    im = imread(input_img_name);
    [h, w, c] = size(im);
    if (c == 3)
        im = rgb2gray(im);
    end
%     im=imresize(im,2,'bicubic'); %% resize image len 2 lan
    
%     wn2 = w/8; hn2 = h/8;
%     wn2 =8 * floor(wn2/8);
%     hn2 =8 * floor(hn2/8);

    annPoints =  image_info{1}.location;
%     annPoints=annPoints.*2; %% scale vi tri len 2 lan
    
    im_density = get_density_map_gaussian(im,annPoints);
    
%         x = floor((b_w - a_w) * rand + a_w);
%         y = floor((b_h - a_h) * rand + a_h);
%         x1 = x - wn2; y1 = y - hn2;
%         x2 = x + wn2-1; y2 = y + hn2-1;
%         
%         
%         im_sampled = im(y1:y2, x1:x2,:);
%         im_density_sampled = im_density(y1:y2,x1:x2);
%         
%         annPoints_sampled = annPoints(annPoints(:,1)>x1 & ...
%             annPoints(:,1) < x2 & ...
%             annPoints(:,2) > y1 & ...
%             annPoints(:,2) < y2,:);
%         annPoints_sampled(:,1) = annPoints_sampled(:,1) - x1;
%         annPoints_sampled(:,2) = annPoints_sampled(:,2) - y1;
        img_idx = strcat(num2str(i));        

%         if(idx < num_val)
%             imwrite(im_sampled, [val_path_img num2str(img_idx) '.jpg']);
%             csvwrite([val_path_den num2str(img_idx) '.csv'], im_density_sampled);
%         else
%             imwrite(im_sampled, [train_path_img num2str(img_idx) '.jpg']);
        csvwrite([train_path_den num2str(img_idx) '.csv'], im_density);
%         end
        
%             fprintf(1,'Processing %3d/%d files\n', i, num_images);

   
    
end

