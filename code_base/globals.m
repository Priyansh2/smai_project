
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
