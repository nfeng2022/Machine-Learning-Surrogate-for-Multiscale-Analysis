function y = fc_layer(x,w,b,activation_flag)
% FC layer with RE_LU activation
% n = input 
% x -> input vector volume [n x 1]  
% b -> bias vector    [1xnout]
% y -> output vector volume [nout x 1]
% w -> weight matrix [nxnout]
%%
y = w'*x + b';
%% RE_LU Activation
if activation_flag==1
    y(y<=0)=0;
end
end


