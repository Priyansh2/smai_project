
%   Balanced Clustering with Leas Square Regression
%   Main function
close all;
clear all;
% Directory setting
base_dir = [pwd,'/'];
tmp_dir = [base_dir 'tmp/'];
% Parameter setting
gamma = 10^(-5); % the value of Gamma should be between 0 and 10^(-10)
lam = 10^(2);
mu = 0.1;
infRes = 0.90;    % the percentage of information reserved of the data during PCA dimension reduction
data = 'UMIST';
save([tmp_dir 'param.mat']);

%% initialization
display('Initializing data...');
initialization(data, tmp_dir, infRes);
load([tmp_dir 'init.mat']);
[d,n] = size(X);

StartInd = randsrc(n,1,1:c); Y0 = TransformL(StartInd, c); save([tmp_dir 'Y0'], 'Y0');
load([tmp_dir 'Y0']);


%% Optimization
display('Optimizing...');
[ID, Y, Obj] = BCLS_ALM(X, Y0, gamma, lam, mu);

%% Evaluation
ys = sum(Y);
result = ClusteringMeasure(gt, ID);
ACC = result(1);
NMI = result(2);
[entropy,~,~] = BalanceEvl(c, ys);

%% Show the results
% visualization(data, ID, la, X, n, c);
figure; plot(Obj);
figure; stem(ys);

function  [X_reduce, k, share] = pcaInit(X, thresh)
%PCAINIT Applying dimentional reduction for the original data using PCA.
%   Detailed explanation goes here

X = zscore(X');
[~, SCORE, latent]=princomp(X);
contr = cumsum(latent)./sum(latent);
k = find(contr>=thresh,1);
share = contr(k);
X_reduce = (SCORE(:,1:k))';
end

function initialization(data, tmp_dir, infRes)
%INITIALIZATION Initialize the original data and other variates
%   Detailed explanation goes here

% load data
[Data_ori, gt, c] = loadData(data);
[~,n]=size(Data_ori);
[X, k, share] = pcaInit(Data_ori, infRes);
% X = Data_ori;

% centralization
H = eye(n) - 1/n*ones(n);
X = X*H;

save([tmp_dir 'init.mat'], 'X', 'gt', 'c');
end


function Y = TransformL(y, nclass, type)

n =length(y);
if nargin <= 2
    type = '01';
end;

if nargin > 1
    c = nclass;
    class_set = 1:c;
else
    class_set = unique(y);
    c = length(class_set);
end;

if strcmp(type, '01')
    Y = zeros(n, c);
    for cn = 1:c
        Y((y==class_set(cn)),cn) = 1;
    end;
else
    Y = -1*ones(n, c);
    for cn = 1:c
        Y((y==class_set(cn)),cn) = 1;
    end;
end;
end
%
function [ID, Y, Obj] = BCLS_ALM(X, Y, gamma, lam, mu)
% BCLS_ALM
% min_Y,W,b ||X'W+1b'-Y||^2 + gamma*||W||^2 + lam*Tr(Z'11'Z) + mu/2*||Y-Z + 1/mu*Lambda||^2
% INPUT:
% X: data matrix (d by n), already processed by PCA with 80%~90% information preserved
% Y: randomly initialized label matrix (n by c)
% Parameters: gamma and lam are the parameters respectively corresponding to Eq.(13) in the paper
% OUTPUT:
% ID: indicator vector (n by 1)
% Y: generated label matrix (b by c)


ITER = 1200;
[dim, n] = size(X);

H = eye(n) - 1/n*ones(n);
X = X*H;

c = size(Y,2);   % number of clusters
Lambda = zeros(n,c);
rho = 1.005;
P = eye(dim)/(X*X'+gamma*eye(dim));

for iter = 1:ITER

    display(['Solving alternatively...',num2str(iter)]);

    % Solve W and b
    W = P*(X*Y);
    b = mean(Y)';
    E = X'*W + ones(n,1)*b' - Y;

    % Solve Z
%     Z = (mu*eye(n)+2*lam*ones(n))\(mu*Y + Lambda);   % original solution - O(n^3)
    Z = (-2*lam*ones(n)+(mu+2*n*lam)*eye(n))/(mu^2+2*n*lam*mu)*(mu*Y+Lambda);  % new solution - O(n^2)

    % Solve Y
    V = 1/(2+mu)*(2*X'*W + 2*ones(n,1)*b' + mu*Z - Lambda);
    [~, ind] = max(V,[],2);
    Y = zeros(n,c);
    Y((1:n)' + n*(ind-1)) = 1;

    % Update Lambda and mu according to ALM
    Lambda = Lambda + mu*(Y-Z);
    mu = min(mu*rho, 10^5);

    % Objective value
    Obj(iter) = trace(E'*E) + gamma*trace(W'*W) + lam*trace(Y'*ones(n)*Y);

end;

[~,ID] = max(Y,[],2);

end
%% Normalized Entropy

% Evaluate the balance of the distribution of the clustering

function [entro, stDev, RME] = BalanceEvl(k, N_cluster)

     aa = [];
     bb = [];
     for i=1:k
         N = sum(N_cluster);
         Ni = N_cluster(i)+eps;
         a = Ni/N * log(Ni/N);
         aa(i) = a;
         b = (Ni-N/k)^2;
         bb(i) = b;
     end
     entro = -1/(log(k)) * sum(aa);    % Entropy of the cluster distribution; (0,1)
     stDev = (1/(k-1)*sum(bb))^(1/2);  % Standard deviation in cluster size (SDCS)

     RME = (min(N_cluster))/(N/k);     % ratio of minimum to expected (RME); (0,1)

end
