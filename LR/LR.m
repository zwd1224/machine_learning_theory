clear;
reshaped_faces=[];
%ORL  AR   FERET   Yale
database_name = "Yale";

% ORL5646
if (database_name == "ORL")
  for i=1:40    
    for j=1:10       
       if(i<10)
           a=imread(strcat('C:\Users\74203\Desktop\lab3\ORL56_46\orl',num2str(i),'_',num2str(j),'.bmp'));     
        else
            a=imread(strcat('C:\Users\74203\Desktop\lab3\ORL56_46\orl',num2str(i),'_',num2str(j),'.bmp'));  
        end         
        b = reshape(a,2576,1);
        b=double(b);        
        reshaped_faces=[reshaped_faces, b];  
    end
  end
row = 56;
column = 46;
people_num = 40;
pic_num_of_each = 10;
train_num_each = 5;% 每类训练数量
test_num_each = 5; % 每类测试数量
test_sum = test_num_each * people_num; % 测试总数
end

%AR5040
if (database_name == "AR")
    for i=1:40    
      for j=1:10       
       if(i<10)
           a=imread(strcat('C:\Users\74203\Desktop\lab3\AR_Gray_50by40\AR00',num2str(i),'-',num2str(j),'.tif'));     
        else
            a=imread(strcat('C:\Users\74203\Desktop\lab3\AR_Gray_50by40\AR0',num2str(i),'-',num2str(j),'.tif'));  
        end              
        b = reshape(a,2000,1);
        b=double(b);        
        reshaped_faces=[reshaped_faces, b];  
      end
    end
row = 50;
column = 40;
people_num = 40;
pic_num_of_each = 10;
train_num_each = 4;% 每类训练数量
test_num_each = 6; % 每类测试数量
test_sum = test_num_each * people_num; % 测试总数
end

%FERET_80
if (database_name == "FERET")
    for i=1:80    
      for j=1:7       
        a=imread(strcat('C:\Users\74203\Desktop\lab3\FERET_80\ff',num2str(i),'_',num2str(j),'.tif'));            
        b = reshape(a,6400,1);
        b=double(b);        
        reshaped_faces=[reshaped_faces, b];  
      end
    end
row = 80;
column = 80;
people_num = 80;
pic_num_of_each = 7;
train_num_each = 4;% 每类训练数量
test_num_each = 3; % 每类测试数量
test_sum = test_num_each * people_num; % 测试总数
end

%Yale
if (database_name == "Yale")
    for i=1:15    
      for j=1:11
          if (i < 10)
            a=imread(strcat('C:\Users\74203\Desktop\lab3\face10080\subject0',num2str(i),'_',num2str(j),'.bmp')); 
          else
            a=imread(strcat('C:\Users\74203\Desktop\lab3\face10080\subject',num2str(i),'_',num2str(j),'.bmp'));  
          end
        b = reshape(a,8000,1);
        b=double(b);        
        reshaped_faces=[reshaped_faces, b];  
      end
    end
row = 100;
column = 80;
people_num = 15;
pic_num_of_each = 11;
train_num_each = 2;% 每类训练数量
test_num_each = 9; % 每类测试数量
test_sum = test_num_each * people_num; % 测试总数
end

%%PCA降维
mean_face = mean(reshaped_faces,2);
centered_face = (reshaped_faces - mean_face);
cov_matrix = centered_face * centered_face';
[eigen_vectors, dianogol_matrix] = eig(cov_matrix);
eigen_values = diag(dianogol_matrix);
[sorted_eigen_values, index] = sort(eigen_values, 'descend'); 
sorted_eigen_vectors = eigen_vectors(:, index);

 % 取出相应数量特征脸(降到n维)
  n = 200;
  eigen_faces = sorted_eigen_vectors(:,1:n);
    % 测试、训练数据降维
  projected_data = eigen_faces' * (reshaped_faces - mean_face);
  % 使用PCA降维
  reshaped_faces = projected_data;
    
% 回归过程
dimension = row * column;
count_right = 0;

for i = 0:1:people_num - 1
     %取出图片对应标签
    totest_index = i + 1;
    %对每一类进行一次线性回归
    for k = train_num_each + 1:1:pic_num_of_each
        %取出每一待识别（分类）人脸
        totest = reshaped_faces(:,i*pic_num_of_each + k);
        distest = []; %记录距离
     for j = 0:1:people_num - 1
       batch_faces = reshaped_faces(:,j * pic_num_of_each + 1 :j * pic_num_of_each + pic_num_of_each); 
       %取出每一类图片
       % 划分训练集与测试集
       %第一次  batch中的前train_num_each个数据作为训练集 后面数据作为测试集合
       train_data = batch_faces(:,1:train_num_each);
       test_data = batch_faces(:,train_num_each + 1:pic_num_of_each);
         % 1.求导法线性回归
         w = inv(train_data' * train_data) * train_data' * totest;
         img_predict = train_data * w; % 计算预测图片           

         % 2.岭回归
%        rr_data = (train_data' * train_data) + eye(train_num_each)*10^-6;
%        w = inv(rr_data) * train_data' * totest;
%        img_predict = train_data * w; % 计算预测图片

         % 3.lasso回归
%        [B,FitInfo] = lasso(train_data , totest);
%        img_predict = train_data * B + FitInfo.Intercept;

        

         % 5.新sheji 
          rr_data = (train_data' * train_data) + eye(train_num_each)*10^-6;
          eye(train_num_each)*10^-6; 
          w = inv(rr_data) * train_data' * test_data; % 改良w
          img_predict = train_data * w; % 计算预测图片
         

       dis = img_predict - totest; % 计算误差
       distest = [distest,norm(dis)]; % 计算欧氏距离
     end
            [min_dis,label_index] = min(distest); 
            % 找到最小欧氏距离下标（预测类）
            if label_index == totest_index
              count_right = count_right + 1;
            end
    end 
end
recognition_rate = count_right / test_sum; 
fprintf("准确度：%.2f\n" ,recognition_rate);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%二维可视化
% class_num_to_show = 3;
% pic_num_in_a_class = people_num;
% pic_to_show = class_num_to_show * pic_num_in_a_class;
% 
% 
%  取出相应数量特征向量
% project_matrix = eigen_vectors(:,1:2);
% 
% 投影
% projected_test_data = project_matrix' * (reshaped_faces - all_mean);
% projected_test_data = projected_test_data(:,1:pic_to_show);
% color = [];
% for j=1:pic_to_show
%     color = [color floor((j-1)/pic_num_in_a_class)*20];
% end
% 显示
% subplot(1, 7, [1, 2, 3, 4]);
% scatter(projected_test_data(1, :), projected_test_data(2, :), [], color, 'filled');
% for j=1:3
%     subplot(1, 7, j+4);
%     fig = show_face(test_data(:,floor((j - 1) * pic_num_in_a_class) + 1), row, column);
% end
% waitfor(fig);
   


% 输入向量，显示脸
function fig = show_face(vector, row, column)
    fig = imshow(mat2gray(reshape(vector, [row, column])));
end