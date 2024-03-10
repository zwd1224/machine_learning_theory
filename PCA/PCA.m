clear;
% 特征提取
load('C:\Users\74203\Desktop\pca\ORL4646.mat')
wholematrix = reshape(ORL4646,46*46,400);
train_n = 5;
ldimension = 100;
train_m = zeros(46*46,0);
tic;
for i = 0:10:390
    for j = 1:train_n
     train_m = horzcat(train_m,wholematrix(:,i+j)); 
    end
end

%求平均脸，即每一列分别求均值
avcoloumn = mean(train_m,2);

for i = 1:train_n*40
    %中心化
     train_m(:,i) = train_m(:,i) - avcoloumn;
end

% 协方差矩阵
COV = train_m*train_m'/(40*train_n);

%计算COV的特征值对角阵D和特征向量V
[V,D] = eig(COV);

% 特征值排序
[d,ind] = sort(diag(D),"descend");

Ds = D(ind,ind);
Vs = V(:,ind);
eigenmatrix = Vs(:,1:ldimension);
%重构
rec_matrix = eigenmatrix*(eigenmatrix'*train_m);
show_matrix = reshape(rec_matrix(:,2),46,46);

%记录程序完成时间
toc;


% % 优化特征向量的计算
% cov1 = train_m'*train_m;
% [V,D] = eig(cov1);
% [d,ind] = sort(diag(D),"descend");
% Ds = D(ind,ind);
% Vs = V(:,ind);
% eigenmatrix1 = Vs(:,1:ldimension);
% eigenmatrix = train_m*eigenmatrix1;
% toc;
% 
% % 重构
% rec_matrix = eigenmatrix*(eigenmatrix'*train_m);
% show_matrix = reshape(rec_matrix(:,2),46,46);
% figure;
% imshow(show_matrix,[]);
% % 重构

% 识别
testmatrix = zeros(46*46,0);
for i = train_n:10:(390+train_n)
    for j = 1:(10-train_n)
        testmatrix = horzcat(testmatrix,wholematrix(:,i+j));
    end
end
for i = 1:(10-train_n)*40
     testmatrix(:,i) = testmatrix(:,i) - avcoloumn;
end
resultmatrix = eigenmatrix'*train_m;
testmatrix1 = eigenmatrix'*testmatrix;
correctnum = 0;
for i =1:40*(10-train_n)
    tmatrix = zeros(1,40*train_n);
    cmatrix = zeros(2,ldimension);
    for j = 1:40*train_n
        cmatrix(1,:) = testmatrix1(:,i)';
        cmatrix(2,:) = resultmatrix(:,j)';
        tmatrix(j) = pdist(cmatrix);
    end
    [M,I] = min(tmatrix);
    a1 = ceil(i/(10-train_n));
    a2 = ceil(I/(train_n));
    if a1==a2
        correctnum = correctnum + 1;
    end
end
rateofcorrect = correctnum/(40*(10-train_n))
% 识别