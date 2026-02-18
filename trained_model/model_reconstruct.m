function [S, CM] = model_reconstruct(x,model_weights, norm_stat)
% reconstruct the neural netwrok model for FEA
% input: x - C tesor somponents with 3x1 vector [C11; C21; C22];
% output: S - S tensor components with 3x1 vector [S11; S21; S22]
% output: CM - CM tensor components with 9x1 vector
% [CM1111; CM2111; CM2211; CM1112; CM2112; CM2212; 
% CM1122; CM2122; CM2222]
%% Normalizing
x0 = (x-norm_stat.Cmac_mu')./norm_stat.Cmac_std';
%% Forward and backpropagation pass
%%%%%%%%%%%%%1st network block%%%%%%%%%%%%%%%%%%%%
activation_flag = 1;
y1 = fc_layer(x0,model_weights.block0.dense_trainable0,model_weights.block0.dense_trainable1,activation_flag);
dy1_dx0 = zeros(size(y1,1), size(x0,1));
dy1_dx0(y1~=0,:) = model_weights.block0.dense_trainable0(:, y1~=0)';

[y11,dy11_dy1] = conv_1d(y1,model_weights.block0.conv1d_trainable0,model_weights.block0.conv1d_trainable1,activation_flag);

activation_flag = 0;
[y12, dy12_dy11] = conv_1d(y11,model_weights.block0.conv1d_1_trainable0,model_weights.block0.conv1d_1_trainable1,activation_flag);

y13 = y12+y1; % Resnet connection
dy13_dx0 = dy12_dy11*dy11_dy1*dy1_dx0+dy1_dx0;
%%%%%%%%%%%%%2nd network block%%%%%%%%%%%%%%%%%%%%
activation_flag = 1;
y2 = fc_layer(y13,model_weights.block1.dense_1_trainable0,model_weights.block1.dense_1_trainable1,activation_flag);
dy2_dy13 = zeros(size(y2,1), size(y13,1));
dy2_dy13(y2~=0,:) = model_weights.block1.dense_1_trainable0(:, y2~=0)';

[y21,dy21_dy2] = conv_1d(y2,model_weights.block1.conv1d_2_trainable0,model_weights.block1.conv1d_2_trainable1,activation_flag);

activation_flag = 0;
[y22,dy22_dy21] = conv_1d(y21,model_weights.block1.conv1d_3_trainable0,model_weights.block1.conv1d_3_trainable1,activation_flag);

y23 = y22+y2;
dy23_dy13 = dy22_dy21*dy21_dy2*dy2_dy13+dy2_dy13;
%%%%%%%%%%%%%3rd network block%%%%%%%%%%%%%%%%%%%%
activation_flag = 1;
y3 = fc_layer(y23,model_weights.block2.dense_2_trainable0,model_weights.block2.dense_2_trainable1,activation_flag);
dy3_dy23 = zeros(size(y3,1), size(y23,1));
dy3_dy23(y3~=0,:) = model_weights.block2.dense_2_trainable0(:, y3~=0)';

[y31,dy31_dy3] = conv_1d(y3,model_weights.block2.conv1d_4_trainable0,model_weights.block2.conv1d_4_trainable1,activation_flag);

activation_flag = 0;
[y32,dy32_dy31] = conv_1d(y31,model_weights.block2.conv1d_5_trainable0,model_weights.block2.conv1d_5_trainable1,activation_flag);

y33 = y32+y3;
dy33_dy23 = dy32_dy31*dy31_dy3*dy3_dy23+dy3_dy23;
%%%%%%%% output layer %%%%%%%%%%%%%%%%%%%
activation_flag = 0;
S0 = fc_layer(y33,model_weights.last_layer_w,model_weights.last_layer_b,activation_flag);
dS0_dy33 = model_weights.last_layer_w';
CM0 = (dS0_dy33*dy33_dy23)*(dy23_dy13*dy13_dx0);
%% Recover P and A
S = S0.*norm_stat.Smac_std'+norm_stat.Smac_mu';
CM = CM0.*kron(norm_stat.Smac_std',1./norm_stat.Cmac_std);
CM = CM(:);
end