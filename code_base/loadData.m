function [Data_ori, gt, c] = loadData(data)

     switch data
       case 'YALE-B'
             load('Data/YALE-B/X.mat')
             Data_ori = [];
             gt = [];
             X=fea;
             c = numel(X);
             for k=1:c
                 clu = X{1,k};
                 nn = 59;%size(clu,3);
                 A = reshape(clu, 32*32, []);
                 Data_ori = [Data_ori, A(:, 1:nn)];
                 ind = k*ones(nn,1);
                 gt = [gt; ind];
             end

         case 'UMIST'
             load('Data/UMIST/UMIST')
             Data_ori = [];
             gt = [];
             c = numel(X);
             for k=1:c
                 clu = X{1,k};
                 nn = 19;%size(clu,3);
                 A = reshape(clu, 112*92, []);
                 Data_ori = [Data_ori, A(:, 1:19)];
                 ind = k*ones(nn,1);
                 gt = [gt; ind];
             end

         case 'wine';
             load 'Data/wine/X.mat';
             load 'Data/wine/Y.mat';
             c = 3;
             gt = wine1;
             Data_ori = wine2;

         case 'ionosphere';
             load 'Data/ionosphere/X.mat';
             load 'Data/ionosphere/Y.mat';
             c = 2;
             gt = Y;
             Data_ori = X;

     end

end