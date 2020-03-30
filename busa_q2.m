%Author: Michael Busa
%ML HW 3 - Question 1
%3/19/20

clear
close all, 
clc

delta = 1e-2; %tolerance for EM stopping criterion
regWeight = 1e-10; % regularization parameter for covariance estimates
%Generate datasets
C = 4;
N_train = 1000;
N_test = 10000;

train_f1 = strcat('d_train_',num2str(N_train),'.mat');
train_f2 = strcat('d_train_labels_',num2str(N_train),'.mat');

load('d_test.mat');
load('d_test_labels.mat');
load(train_f1);
load(train_f2);

for i=1:C
    train_cp(i) = length(find(d_train_labels==i))/N_train;
    test_cp(i) = length(find(d_test_labels==i))/N_test;
end

c1_train_data = d_train(:,d_train_labels==1);
c2_train_data = d_train(:,d_train_labels==2);
c3_train_data = d_train(:,d_train_labels==3);
c4_train_data = d_train(:,d_train_labels==4);

%Activation Functions
%Sigmoid: 1/(1+e^-x)
%Soft ReLu: ln(1+e^x)

%% 10-fold cross-validation
%Split d_train into 10 equal parts
folds = 10;
samples_per_fold = N_train/folds;
k_points =  N_train - samples_per_fold;

k_folds = zeros(2,samples_per_fold,folds);
k_folds_labels = zeros(1,samples_per_fold,folds);
for k=1:folds
    k_folds(:,:,k) = d_train(:,((k-1)*samples_per_fold)+1:samples_per_fold*k);
    k_folds_labels(:,:,k) = d_train_labels(:,((k-1)*samples_per_fold)+1:samples_per_fold*k);
end

GMM_C = 6;
k_train = zeros(2,N_train-samples_per_fold);
k_train_labels = zeros(1,N_train-samples_per_fold);
%%
hoods=zeros(GMM_C,folds);
for class = 1:C
    for gmm_c = 1:GMM_C %GMM components
        for Runs = 1:10
            fprintf('%d\t%d\t%d\n',class,gmm_c,Runs);
            for k=1:folds
                %Separate data into one train, one val
                unused_folds = [1:10];
                unused_folds(unused_folds==k) = [];
                k_val = k_folds(:,:,k);
                k_val_labels = k_folds_labels(:,:,k);
                for i=1:9
                    k_train(:,(i-1)*samples_per_fold+1:samples_per_fold*i) = k_folds(:,:,i);
                    k_train_labels(:,(i-1)*samples_per_fold+1:samples_per_fold*i) = k_folds_labels(:,:,i);
                end
                
                %Only use points in the current class
                k_train_class = k_train(:,k_train_labels==class);
                k_val_class = k_val(:,k_val_labels==class);
                clear k_train k_val
                k_train = k_train_class;
                k_val = k_val_class;

                N = length(k_train);
                clearvars -except delta hoods comp_choice components regWeight d_train d_train_labels d_test d_test_labels B N Runs train_cp test_cp k_val_class k_train_class k_train_labels k_train k_val_labels k_val unused_folds class gmm_c k folds GMM_C C k_folds k_folds_labels samples_per_fold k_points N_train N_test c1_train_data c2_train_data c3_train_data c4_train_data
                % Initialize the GMM to randomly selected samples
                alpha = ones(1,gmm_c)/gmm_c;
                shuffledIndices = randperm(N);
                mu = k_train(:,shuffledIndices(1:gmm_c)); % pick M random samples as initial mean estimates
                [~,assignedCentroidLabels] = min(pdist2(mu',k_train'),[],1); % assign each sample to the nearest mean
                for m = 1:gmm_c % use sample covariances of initial assignments as initial covariance estimates
                    Sigma(:,:,m) = cov(k_train(:,find(assignedCentroidLabels==m))') + regWeight*eye(2,2);
                    if(isnan(Sigma(:,:,m)))
                        Sigma(:,:,m) = eye(2,2);
                    end
                end
                t = 0; %displayProgress(t,x,alpha,mu,Sigma);

                Converged = 0; % Not converged at the beginning
                while ~Converged
                    for l = 1:gmm_c
                        temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(k_train,mu(:,l),Sigma(:,:,l));
                    end
                    plgivenx = temp./sum(temp,1);
                    alphaNew = mean(plgivenx,2);
                    w = plgivenx./repmat(sum(plgivenx,2),1,N);
                    muNew = k_train*w';
                    for l = 1:gmm_c
                        v = k_train-repmat(muNew(:,l),1,N);
                        u = repmat(w(l,:),2,1).*v;
                        SigmaNew(:,:,l) = u*v' + regWeight*eye(2,2); % adding a small regularization term
                    end
                    Dalpha = sum(abs(alphaNew-alpha));
                    Dmu = sum(sum(abs(muNew-mu)));
                    DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
                    Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
                    alpha = alphaNew; 
                    mu = muNew; 
                    Sigma = SigmaNew;
                    t = t+1;
                    %displayProgress(t,k_train,alpha,mu,Sigma,gmm_c);
                end
                %pause(1)
                alpha_total(gmm_c,1:gmm_c) = alpha';
                mu_total(:,gmm_c,1:gmm_c) = mu;
                Sigma_total(:,:,gmm_c,1:gmm_c) = Sigma;

                logLikelihood = sum(log(evalGMM(k_val,alpha_total(gmm_c,1:gmm_c),mu_total(:,gmm_c,1:gmm_c),Sigma_total(:,:,gmm_c,1:gmm_c)))); 
                hoods(gmm_c,k) = logLikelihood;
            end
            [~,components] = min(abs(hoods));
            comp_choice(class,Runs) = max(mode(components));
        end
    end
end

clear k_train
class_choice = mode(comp_choice,2)';
total_alphas = zeros(6,4);
total_mus = zeros(2,6,4);
total_sigs = zeros(2,2,6,4);
for i=1:C
    clear Sigma SigmaNew
    if i==1
        k_train = c1_train_data;
    elseif i==2
        k_train = c2_train_data;
    elseif i==3
        k_train = c3_train_data;
    else
        k_train = c4_train_data;
    end
    
    gmm_c = class_choice(i);
    N = length(k_train);
    
    alpha = ones(1,gmm_c)/gmm_c;
    shuffledIndices = randperm(N);
    mu = k_train(:,shuffledIndices(1:gmm_c)); % pick M random samples as initial mean estimates
    [~,assignedCentroidLabels] = min(pdist2(mu',k_train'),[],1); % assign each sample to the nearest mean
    for m = 1:gmm_c % use sample covariances of initial assignments as initial covariance estimates
        Sigma(:,:,m) = cov(k_train(:,find(assignedCentroidLabels==m))') + regWeight*eye(2,2);
        if(isnan(Sigma(:,:,m)))
            Sigma(:,:,m) = eye(2,2);
        end
    end
    t = 0; %displayProgress(t,x,alpha,mu,Sigma);

    Converged = 0; % Not converged at the beginning
    clear temp
    while ~Converged
        for l = 1:gmm_c
            temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(k_train,mu(:,l),Sigma(:,:,l));
        end
        plgivenx = temp./sum(temp,1);
        alphaNew = mean(plgivenx,2);
        w = plgivenx./repmat(sum(plgivenx,2),1,N);
        muNew = k_train*w';
        for l = 1:gmm_c
            v = k_train-repmat(muNew(:,l),1,N);
            u = repmat(w(l,:),2,1).*v;
            SigmaNew(:,:,l) = u*v' + regWeight*eye(2,2); % adding a small regularization term
        end
        Dalpha = sum(abs(alphaNew-alpha));
        Dmu = sum(sum(abs(muNew-mu)));
        DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
        Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
        alpha = alphaNew; 
        mu = muNew; 
        Sigma = SigmaNew;
        t = t+1;
        %displayProgress(t,k_train,alpha,mu,Sigma,gmm_c);
    end
    total_alphas(1:gmm_c,i) = alpha;
    total_mus(:,1:gmm_c,i) = mu;
    total_sigs(:,:,1:gmm_c,i) = Sigma;
    
    test_results(i,:) = evalGMM(d_test,alpha,mu,Sigma)*test_cp(i);
end
[~,class_labels] = max(test_results);
p_error = 100-100*length(find(class_labels==d_test_labels))/N_test;
fprintf('Probability of Error = %2.2f\n',p_error)

%%
function displayProgress(t,x,alpha,mu,Sigma,M)
figure(M),
if size(x,1)==2
    cla
    plot(x(1,:),x(2,:),'k.'); 
    txt = strcat('Data and Estimated GMM Contours:  ', int2str(M));
    xlabel('x_1'), ylabel('x_2'), title(txt),
    axis equal, hold on;
    rangex1 = [min(x(1,:)),max(x(1,:))];
    rangex2 = [min(x(2,:)),max(x(2,:))];
    [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2);
    contour(x1Grid,x2Grid,zGMM); axis equal, 
end
%logLikelihood = sum(log(evalGMM(x,alpha,mu,Sigma)));
%xlabel('Iteration Index'), ylabel('Log-Likelihood of Data'),
drawnow;
end

function [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2)
x1Grid = linspace(floor(rangex1(1)),ceil(rangex1(2)),101);
x2Grid = linspace(floor(rangex2(1)),ceil(rangex2(2)),91);
[h,v] = meshgrid(x1Grid,x2Grid);
GMM = evalGMM([h(:)';v(:)'],alpha, mu, Sigma);
zGMM = reshape(GMM,91,101);
end
