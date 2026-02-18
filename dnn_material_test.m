%%
clc
clear all
%% Load data
load('testing_data.mat');  % testing data
load('final_constrained_model_weights.mat');     % neural network weights
load('normalization_statistics.mat'); % normalizing statistics
%% Use reconstructed model
idex = randi(size(Smac_test,1));                    % sample index for testing
% use the trained neural network model
[Smac_vec, CMmac_vec] = model_reconstruct(Cmac_test(idex,:)',model_weights, norm_stat);
%% plot reconstructed and tensorflow target output data
figure()
plot(Smac_vec,'-b');
hold on
plot(Smac_test(idex,:),'--r');
legend('Reconstruct NN prediction','Target data');
figure()
plot(CMmac_vec,'-b');
hold on
plot(CMmac_test(idex,:),'--r');
legend('Reconstruct NN prediction','Target data');