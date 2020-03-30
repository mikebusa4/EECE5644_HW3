function out = activationFunction(in,func)
if func==1
    out = 1./(1+exp(-in)); % sigmoid function
else
    out = in./sqrt(1+in.^2); % ISRU
end
end
