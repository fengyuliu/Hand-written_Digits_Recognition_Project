% #1 SVM: Implement linear and kernel SVM on MNIST dataset. Three kernels (linear, polynomial, RBF) are considered.
%         A online toolbox: LIBSVM is used. PCA and LDA is applied first for the dimensionality reduction.
clear; clc; close all;

s1 = input('Please input a number: #1 PCA, #2 LDA: ');
if ismember(s1,[1,2])~=1
    disp('error.')
    return
end

s2 = input('Please input a number: #1 linear Kernel, #2 Polynomial Kernel, #3 RBF Kernel: ');
if ismember(s2,[1,2,3])~=1
    disp('error.')
    return
end

Dtrain = loadMNISTImages('MNIST/train-images-idx3-ubyte'); %training dataset
Ltrain = loadMNISTLabels('MNIST/train-labels-idx1-ubyte'); %training data label
Ntrain = length(Ltrain);
Dtest = loadMNISTImages('MNIST/t10k-images-idx3-ubyte'); %testing dataset
Ltest = loadMNISTLabels('MNIST/t10k-labels-idx1-ubyte'); %testing data label
Ntest = length(Ltest);
Nc = length(unique(Ltrain)); %number of the class

%% dimensionality reduction
Nd = 10; %expected number of dimension
if s1 == 1
    [U,Dmu] = PCAfun(Dtrain,Nd);
    Dtrain = U'*(Dtrain-Dmu*ones(1,Ntrain));
    Dtest = U'*(Dtest-Dmu*ones(1,Ntest));
elseif s1 == 2
    U = LDAfun(Dtrain,Ltrain,Nc,Nd);
    Dtrain = U'*Dtrain;
    Dtest = U'*Dtest;
end

model = cell(1,Nc);
for i = 1:Nc
    model{i} = svmtrain(double(Ltrain==(i-1)), Dtrain', ['-s 0 -t ',num2str(s2-1),' -b 1']);
    disp(['finish training ',num2str(i)]);
end

%% using P to prevent being classified into more than one class or not belonging to any class
P = zeros(Ntest,Nc);
for i = 1:Nc
    [l,a,p] = svmpredict(double(Ltest==(i-1)), Dtest', model{i}, '-b 1');
    P(:,i) = p(:, model{i}.Label==1);
    disp(['finish predicting ', num2str(i)]);
end
[M,solution] = max(P,[],2);
accuracy = sum((solution-1) == Ltest)/Ntest;






