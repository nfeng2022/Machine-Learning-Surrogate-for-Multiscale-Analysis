function [y,dy_dx] = conv_1d(x,w,b,activation_flag)
% Convolution layer: equal padding, stride=1
% x -> input vector volume [n x nci]  
% w -> filter weights [fs x nci x nco]
% b -> bias vector    [n x nco]
% y -> output vector volume [n x nco]
% dy_dx -> derivatives for this layer [n*nco] x [n*nci]
% activation_flag -> if activation is used
%% Check dimensions
% ...
%% Extract sizes
n = size(x,1); % number of rows in x or y <== equal padding
nci = size(x,2); % # of input channels
nco = size(w,3); % # of filter == # of output channels
fs = size(w,1); % 1D filter size
%% Padded volume xp
npad = fs-1; % number of zeros to be padded
xp = zeros(n+npad,nci);
if ~mod(npad,2) % even number
    npad2 = npad/2;
else % odd number
    npad2 = (npad-1)/2;
end
xp(npad2+1:npad2+n,:) = x;
i_loc = npad2*nci+1:npad2*nci+n*nci;
j_loc = 1:n*nci;
dxp_dx = sparse(i_loc,j_loc,ones(n*nci,1),(n+npad)*nci,n*nci);
%%
y = zeros(n,nco);
dy_dxp = zeros(n*nco,(n+npad)*nci); % y flattened row-wise  && xp flattened row-wise
for k=1:n
    t = xp(k:k+fs-1,:).*w;
    y(k,:)= sum(t,[1,2]);
    y(k,:)= y(k,:)+ b;% Add Bias
    idx2 = nco*(k-1)+1:nco*k;
    idx3 = nci*(k-1)+1:nci*(k-1)+fs*nci;
    dy_dxp(idx2,idx3) = reshape(permute(w,[2 1 3]),fs*nci,nco)'; % filter size: fs * nci
end
dy_dx = dy_dxp*dxp_dx;
%% RE_LU Activation
if activation_flag==1    
    idx = (y<=0)';
    y(idx')=0; % ReLU activation
    dy_dx(idx(:),:) = 0;    
end
end
