function H = mlpModel(X,params,func)
N = size(X,2);                          
nY = length(params.d);                  
U = params.A*X + repmat(params.b,1,N);  
Z = activationFunction(U,func);              
V = params.C*Z + repmat(params.d,1,N); 
H = exp(V)./repmat(sum(exp(V),1),nY,1); % softmax nonlinearity for second/last layer
% Add softmax layer to make this a model for class posteriors
end

