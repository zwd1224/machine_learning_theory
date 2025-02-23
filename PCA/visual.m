clear;
% 特征提取
load('C:\Users\74203\Desktop\pca\ORL4646.mat')
wholematrix = reshape(ORL4646,46*46,400);
t_n = 5;
ldimension = 2;
t_m = zeros(46*46,0);
tic;
for i = 0:10:390
    for j = 1:t_n
     t_m = horzcat(t_m,wholematrix(:,i+j)); 
    end
end
avcoloumn = mean(t_m,2);
for i = 1:t_n*40
     t_m(:,i) = t_m(:,i) - avcoloumn;
end
fcmatrix = t_m*t_m'/(40*t_n);
[V,D] = eig(fcmatrix);
[d,ind] = sort(diag(D),"descend");
Ds = D(ind,ind);
Vs = V(:,ind);
eigenmatrix = Vs(:,1:ldimension);
r_m = zeros(46*46,0);
for i = 1:50
    r_m = horzcat(r_m,wholematrix(:,i));
end
v_m = eigenmatrix'*r_m;

% % 二维
for i = 1:10
    scatter(v_m(1,i),v_m(2,i),[],[1,0,0],"filled");
     hold on;
end
for i = 1:10
    scatter(v_m(1,i+10),v_m(2,i+10),[],[0,1,0],"filled");
     hold on;
end
for i = 1:10
    scatter(v_m(1,i+20),v_m(2,i+20),[],[0,0,1],"filled");
     hold on;
end
for i = 1:10
    scatter(v_m(1,i+30),v_m(2,i+30),[],[1,0,1],"filled");
     hold on;
end
for i = 1:10
    scatter(v_m(1,i+40),r_m(2,i+40),[],[0,1,1],"filled");
     hold on;
end


% %三维
for i = 1:10
    scatter3(v_m(1,i),v_m(2,i),r_m(3,i),[],[1,0,0],"filled");
    hold on;
end
for i = 1:10
    scatter3(v_m(1,i+10),v_m(2,i+10),r_m(3,i+10),[],[0,1,0],"filled");
    hold on;
end
for i = 1:10
    scatter3(v_m(1,i+20),v_m(2,i+20),r_m(3,i+20),[],[0,0,1],"filled");
    hold on;
end
for i = 1:10
    scatter3(v_m(1,i+30),v_m(2,i+30),r_m(3,i+30),[],[1,0,1],"filled");
    hold on;
end
for i = 1:10
    scatter3(v_m(1,i+40),v_m(2,i+40),r_m(3,i+40),[],[0,1,1],"filled");
    hold on;
end
