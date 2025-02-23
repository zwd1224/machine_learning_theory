clc;
clear;
person=3;               %总共有三类人脸
dataset=2;
images=[];
if dataset==1
    %% ORL数据集读取数据，一共40个人脸，取3个人脸，一个人脸10张图片
    fprintf("%s\n","orl数据集");
    face=10;                %每个类的样本数
    imgNum=face*person;          %样本总数
    image=imread(strcat('C:\Users\74203\Desktop\data_face\ORL56_46\orl',num2str(1),'_',num2str(1),'.bmp'));
    image=im2double(image);
    [row,column]=size(image);   %图片的行数和列数
    images=[];%所有图片
    for i=1:person
        for j=1:face
            image=imread(strcat('C:\Users\74203\Desktop\data_face\ORL56_46\orl',num2str(i),'_',num2str(j),'.bmp'));
            image=im2double(image);
            image=reshape(image,row*column,1);
            images=[images,image];
        end
    end
elseif dataset==2
    %% AR数据集读取数据，一共120个人脸，取3个人脸，一个人脸26张图片
    fprintf("%s\n","ar数据集");
    face=26;                    %每个类的样本数
    imgNum=face*person;          %样本总数
    image=imread(strcat('AR_Gray_50by40\AR',num2str(1,'%03d'),'-',num2str(1),'.tif'));
    image=im2double(image);
    [row,column]=size(image);   %图片的行数和列数
    images=[];%所有图片
    for i=1:person
        for j=1:face
            image=imread(strcat('AR_Gray_50by40\AR',num2str(i,'%03d'),'-',num2str(j),'.tif'));
            image=im2double(image);
            image=reshape(image,row*column,1);
            images=[images,image];
        end
    end
elseif dataset==3
    %% coil-20数据集读取数据，一共20个旋转物体，取3个旋转物体，一个旋转物体72张图片
    fprintf("%s\n","coil-20数据集");
    face=72;                    %每个类的样本数
    imgNum=face*person;          %样本总数
    image=imread(strcat('C:\Users\74203\Desktop\data_face\coil-20-proc\obj',num2str(1),'__',num2str(1),'.png')); 
    image=im2double(image);
    [row,column]=size(image);   %图片的行数和列数
    images=[];%所有图片
    for i=1:person
        for j=0:face-1
            image=imread(strcat('C:\Users\74203\Desktop\data_face\coil-20-proc\obj',num2str(i),'__',num2str(j),'.png')); 
            image=im2double(image);
            image=reshape(image,row*column,1);
            images=[images,image];
        end
    end
end

eigen_matrix_sorted=pca(images,imgNum);
eigen_matrix_sorted=eigen_matrix_sorted(:,1:2);
images=eigen_matrix_sorted'*images;



for k=2:4                           %分成k类
    clusterCenter=zeros(2,k);           %各个类的聚类中心
    label=[];                           %用于存放每个样本的标签
    correct_label=ones(1,imgNum);           %保存正确的标签
    for i=1:imgNum
        label(i)=-1;            %初始化为-1
    end
    
    number=randperm(imgNum,k);              %随机选出1到pointNum中的k(3)个数
    for i=1:k
        clusterCenter(:,i)=images(:,number(i));   %随机选取所有样本中的k(3)个作为聚类中心
    end
    for i=1:imgNum
        correct_label(i)=floor((i+face-1)/face);
    end
    
    figure(1);
    scatter(images(1,:),images(2,:),50,correct_label,'filled');
    
    centerFlag=1;
    while centerFlag==1
        centerFlag=0;
        for i=1:imgNum
            minDistance=inf;                %将最小距离设置为无穷大
            centerIndex=-1;                 %先不给这个样本分类
            for j=1:k
                distance=norm(images(:,i)-clusterCenter(:,j));%求每个样本与聚类中心的距离，进行分类
                if distance<minDistance
                    minDistance=distance;
                    centerIndex=j;          %把这个样本分到第j个类
                end
            end
            if centerIndex~=label(i)        %如果被分到了别的类中，则更新，并继续循环
                label(i)=centerIndex;       %重新设置标签
                centerFlag=1;
            end
        end
        for i=1:k                   %找到所有相同标签的样本，更新聚类中心
            temp=zeros(2,1);        %temp保存所有相同聚类中心的样本的和
            numberOfLabel=0;        %记录样本的数量
            for j=1:imgNum
                if label(j)==i
                    temp=temp+images(:,j);
                    numberOfLabel=numberOfLabel+1;
                end
            end
            temp=temp/numberOfLabel;%得到新的聚类中心的值
            clusterCenter(:,i)=temp;
        end
    end
    figure(k);
    scatter(images(1,:),images(2,:),50,label,'filled');
    hold off;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     计算正确率
    %%%%%%%%%%%%%%%%        以每一类中最多的结果作为正确的结果来判断
    correctNumber=0;
    for i=1:k
        labelnumber=zeros(1,face);
        for j=1:imgNum
            if correct_label(j)==i
                labelnumber(label(j))=labelnumber(label(j))+1;
            end
        end
        correctNumber=correctNumber+max(labelnumber);
    end
    fprintf('k为%d时的正确率为%.2f\n',k,correctNumber/imgNum);
end

function [eigen_matrix_sorted] = pca(train_images,n)
    C=(train_images*train_images')/(n-1);
    [eigen_matrix,diagonal_matrix]=eig(C);%获得特征矩阵和对角矩阵
    eigen_values=diag(diagonal_matrix);%得到特征值
    [~, index] = sort(eigen_values, 'descend');%特征值从大到小排序
    eigen_matrix_sorted=eigen_matrix(:,index);%特征向量按特征值的顺序排序
end

