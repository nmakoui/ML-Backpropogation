%% ===== Load your normalised CSV file =====
% Skip the first row (header)
data = csvread("C:\Users\amaku\OneDrive\Desktop\cslab\Assignment Code\Assignment Code\ce889_dataCollection.csv", 1, 0);

%% ===== Extract input and output columns =====
input_full  = data(:, 1:2);   % x_target, y_target
output_full = data(:, 3:4);   % vel_y, vel_x (recorded from gameplay)

%% ===== Perform 70% / 30% split as required =====
N = size(input_full, 1);
idx = randperm(N);            % random shuffle of row indices

Ntrain = round(0.70 * N);     % 70% for training

train_in  = input_full(idx(1:Ntrain), :);
train_out = output_full(idx(1:Ntrain), :);

%% ===== Transpose AFTER splitting (MATLAB requires this format) =====
train_in  = train_in';
train_out = train_out';

%% ===== Create and configure neural network =====
% feedforwardnet(hidden_neurons, training_function)
net = feedforwardnet(3, 'traingdm');   

% Set learning parameters (from provided script)
net.trainParam.lr = 0.5;        % learning rate (lambda)
net.trainParam.mc = 0.9;        % momentum constant
net.trainParam.min_grad = 1e-5; % eta threshold

%% ===== Train the network =====
net = train(net, train_in, train_out);

%% ===== Get predicted outputs for the training data =====
y = net(train_in);
