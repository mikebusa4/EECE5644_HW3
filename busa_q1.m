%Author: Michael Busa
%ML HW 3 - Question 1
%3/19/20

clear
close all, 
clc

mult = 10;
%Generate datasets
C = 4;
N_train = 100;
N_test = 10000;

train_f1 = strcat('d_train_',num2str(N_train),'.mat');
train_f2 = strcat('d_train_labels_',num2str(N_train),'.mat');

load('d_test.mat');
load('d_test_labels.mat');
load(train_f1);
load(train_f2);

% [d_train,d_train_labels] = generateMultiringDataset(C, N_train);
% [d_test,d_test_labels] = generateMultiringDataset(C, N_test);

%Activation Functions
%Sigmoid: 1/(1+e^-x)
%Soft ReLu: ln(1+e^x)

%% 10-fold cross-validation
%Split d_train into 10 equal parts
for i=1:N_train
    dtl_grid(d_train_labels(i),i) = 1;
end
folds = 10;
samples_per_fold = N_train/folds;
k_points =  N_train - samples_per_fold;
num_in = 2;
num_out = C;
lowest_cost = 0;

k_folds = zeros(2,samples_per_fold,folds);
k_folds_labels = zeros(1,samples_per_fold,folds);
for k=1:folds
    k_folds(:,:,k) = d_train(:,((k-1)*samples_per_fold)+1:samples_per_fold*k);
    k_folds_labels(:,:,k) = d_train_labels(:,((k-1)*samples_per_fold)+1:samples_per_fold*k);
end
L = 5;
P = 6;
F = 1;
k_train = zeros(2,N_train-samples_per_fold);
k_train_labels = zeros(1,N_train-samples_per_fold);
error_results = zeros(1,P);
%total_vecParams = zeros(46,L*P*F*folds);
counter = 0;
for func = 1:1
    for p = 1:P %perceptrons
        counter=counter+1;
        for l=1:L
            correct = 0;
            all_params = 0;
            clear all_params
            for k=1:folds
                %Separate data into one train, one val
                unused_folds = [1:10];
                unused_folds(unused_folds==k) = [];
                k_val = k_folds(:,:,k);
                k_val_labels = k_folds_labels(:,:,k);
                kvl_grid = zeros(4,samples_per_fold);
                for i=1:samples_per_fold
                    kvl_grid(k_val_labels(i),i) = 1;
                end
                for i=1:9
                    k_train(:,(i-1)*samples_per_fold+1:samples_per_fold*i) = k_folds(:,:,i);
                    k_train_labels(:,(i-1)*samples_per_fold+1:samples_per_fold*i) = k_folds_labels(:,:,i);
                end
                ktl_grid = zeros(4,k_points);
                for i=1:k_points
                    ktl_grid(k_train_labels(i),i) = 1;
                end
                % Determine/specify sizes of parameter matrices/vectors
                nX = 2;%size(X,1); 
                nPerceptrons = p; 
                nY = 4;%size(Y,1);
                sizeParams = [nX;nPerceptrons;nY];       
           
            
                %fprintf('F = %d\tp = %d\tk = %d\tl = %d\n',func,p,k,l);
                % Initialize model parameters
                params.A = 10*rand(nPerceptrons,nX);
                params.b = 10*rand(nPerceptrons,1);
                params.C = 10*rand(nY,nPerceptrons);
                params.d = 10*randn(nY,1);

                vecParamsInit = [params.A(:);params.b;params.C(:);params.d];

                % Optimize model
                options = optimset('MaxFunEvals',200000, 'MaxIter', 200000);
                vecParams = fminsearch(@(vecParams)(objectiveFunction(k_train,ktl_grid,sizeParams,vecParams,func)),vecParamsInit,options);
                final_cost = objectiveFunction(k_train,ktl_grid,sizeParams,vecParams,func);
                all_params(1:length(vecParams),k) = vecParams;
                
                %vecParams = total_vecParams(:,min_l);
                params.A = reshape(vecParams(1:nX*nPerceptrons),nPerceptrons,nX);
                params.b = vecParams(nX*nPerceptrons+1:(nX+1)*nPerceptrons);
                params.C = reshape(vecParams((nX+1)*nPerceptrons+1:(nX+1+nY)*nPerceptrons),nY,nPerceptrons);
                params.d = vecParams((nX+1+nY)*nPerceptrons+1:(nX+1+nY)*nPerceptrons+nY);

                %Test on validator set
                Y_found(:,samples_per_fold*(k-1)+1:samples_per_fold*k) = mlpModel(k_val,params,func);
                tester_labels(samples_per_fold*(k-1)+1:samples_per_fold*k) = k_val_labels;
                fprintf('%d\t%d\t%d\n',p,l,k);
            end
            [~,labels] = max(Y_found);
            correct = correct + length(find(labels==tester_labels));
            wrong = N_train-correct;
            p_error = 100.*wrong./N_train;
            if l==1
                lowest_p_error = p_error;
                best_p_params = all_params;
            elseif p_error<lowest_p_error
                lowest_p_error = p_error;
                best_p_params = all_params;
            end 
        end
        error_results(counter) = lowest_p_error;
    end
end


m = min(error_results)
m_locs = find(error_results==min(error_results))
num_min = length(m_locs)
best_p = zeros(1,num_min);
best_f = zeros(1,num_min);
l_res = length(error_results);
for i=1:num_min
    if m_locs(i)<=l_res/2
        best_f(i)=1;
    else
        %m_locs(i) = m_locs(i)-l_res/2;
        best_f(i)=2;
    end
    if m_locs(i)<=l_res/12
        best_p(i) = 1;
    elseif m_locs(i)<=l_res/6
        best_p(i) = 2;
    elseif m_locs(i)<=l_res/4
        best_p(i) = 3;
    elseif m_locs(i)<=l_res/3
        best_p(i) = 4;
    elseif m_locs(i)<=5*l_res/12
        best_p(i) = 5;
    else
        best_p(i) = 6;
    end
end
best_f = 1;
best_p = mode(best_p)



%best_params = total_vecParams(:,m_locs);
%best_params = vecParamsTrue;


p_correct = zeros(1,15);
X = d_train;
best_params = zeros(46,5);
for l=1:L 
    sizeParams = [2;best_p;4];
    vecParamsInit_final = best_p_params(:,l); %Using the parameters found during k-fold as initial values
    % Optimize model
    best_params(1:length(vecParamsInit_final),l) = fminsearch(@(best_params)(objectiveFunction(X,dtl_grid,sizeParams,best_params,best_f)),vecParamsInit_final,options);
    clear final_params
    final_params.A = [best_params(1:best_p,l) best_params(best_p+1:2*best_p,l)];
    final_params.b = best_params(2*best_p+1:3*best_p,l);
    for r = 1:best_p
        final_params.C(:,r) = best_params(3*best_p+(nY*(r-1))+1:3*best_p+(nY*r),l);
    end
    final_params.d = best_params(3*best_p+1+nY*r:3*best_p+nY*(r+1),l);

    %Apply MLP to Dtest;
    clear Y_end cor wr
    cor = 0;
    Y_end = mlpModel(d_test,final_params,best_f);  
    [~,final_test_labels] = max(Y_end);
    cor = cor + length(find(final_test_labels==d_test_labels));
    wr = N_test-cor;
    final_p_error(l) = 100.*wr./N_test
end


    

%%




