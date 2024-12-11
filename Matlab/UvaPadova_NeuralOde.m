%% Load and Prepare Data
load('Bergmann_data.mat');
A = data;
xInit = A(:,1);
xTrain = A;
T = 1440; % Total time in minutes
numTimeSteps = 1441;
timesteps = linspace(0, T, numTimeSteps);
u = xTrain(4,:); % Control input
d = xTrain(5,:); % Disturbance
% Extract States for Training
xTrain = xTrain(1:3, :);
X0 = dlarray(xTrain(:,1));

% Plot Training Data
figure;
plot(timesteps, xTrain);
xlabel('Time (minutes)');
ylabel('States');
title('Training Data');
legend('State 1', 'State 2', 'State 3', 'Control Input', 'Disturbance');

%% Neural ODE Setup
neuralOdeTimesteps = 850;
dt = timesteps(3) - timesteps(2);
stateSize = 3;
hiddenSize = 256;
externalInputSize = 2;

% Interpolation Functions for Control and Disturbance
global U D
U = @(ti) dlarray(interp1(timesteps, u, ti), "CB");
D = @(ti) dlarray(interp1(timesteps, d, ti), "CB");



% Initialize Neural ODE Parameters
neuralOdeParameters = initializeNeuralOdeParameters(stateSize, hiddenSize, externalInputSize);

%% Training Setup
gradDecay = 0.9;
sqGradDecay = 0.999;
learnRate = 0.01;
numIter = 1000;
miniBatchSize = 20;
plotFrequency = 20;
weights = 1 ./ (max(xTrain(1:3,:), [], 2) + 0.01);

averageGrad = [];
averageSqGrad = [];
monitor = trainingProgressMonitor(Metrics="Loss", Info=["Iteration", "LearnRate"], XLabel="Iteration");
yscale(monitor,"Loss","log")
numTrainingTimesteps = numTimeSteps;
trainingTimesteps = 1:numTrainingTimesteps;
plottingTimesteps = 2:numTimeSteps;

iteration = 0;

while iteration < numIter && ~monitor.Stop
    iteration = iteration + 1;

    % Create Mini-Batch
    disp("Creating Mini-Batch")
    [X, targets,T] = createMiniBatch(numTrainingTimesteps, neuralOdeTimesteps, miniBatchSize, xTrain,timesteps);

    % Evaluate Model and Compute Loss and Gradients
    disp("Evaluating Model")
    [loss, gradients] = dlfeval(@modelLoss, T, X, neuralOdeParameters, targets, weights);

    % Update Neural ODE Parameters
    disp("Updating Parameters")
    [neuralOdeParameters, averageGrad, averageSqGrad] = adamupdate(neuralOdeParameters, gradients, averageGrad, averageSqGrad, iteration, ...
        learnRate, gradDecay, sqGradDecay);

    % Record Metrics
    recordMetrics(monitor, iteration, Loss=loss);

    % Plot Predicted vs. Real Dynamics
    if mod(iteration, plotFrequency) == 0 || iteration == 1
        % Use ode45 to Compute the Solution
        y = dlode45(@odeModel, timesteps, X0, neuralOdeParameters, DataFormat="CB");
        updateTrainingPlots(plottingTimesteps, xTrain, y);
    end

    if mod(iteration, plotFrequency/4) == 0
        % Update Training Progress Plot
        updateInfo(monitor, Iteration=iteration, LearnRate=learnRate);
        monitor.Progress = 100 * iteration / numIter;
    end
end

%% Functions

function X = model(tspan, X0, neuralOdeParameters)
    % Solve ODE using dlode45
    % d=dlarray(zeros(1,550))
    % u=dlarray(zeros(1,550))
    X = dlode45(@odeModel, tspan, X0, neuralOdeParameters,DataFormat="CB");
    X = cat(3,X0,X);
end

function y = odeModel(t, y, theta)
    % Neural ODE Model Definition
    global U D;
    u = U(t);
    d = D(t);
    input = stripdims(dlarray([y;u;d]));
    y = tanh(theta.fc1.Weights * input + theta.fc1.Bias);
    y = theta.fc2.Weights * y + theta.fc2.Bias;
end

function [loss, gradients] = modelLoss(tspan, X0, neuralOdeParameters, targets, weights)
    % Compute Predictions
    X=zeros();
    for btc=1:size(targets,2)
        Xb = model(tspan(:,btc), X0(:,btc), neuralOdeParameters);
        if btc==1
            X=Xb;
        else
            X=cat(2,X,Xb);
        end
    end

    % Compute L2 Loss
    loss = l2loss(X, targets, weights, 'WeightsFormat', 'C', 'NormalizationFactor', 'batch-size', DataFormat="CBT");

    % Compute Gradients
    gradients = dlgradient(loss, neuralOdeParameters);
end

function [x0, targets,T] = createMiniBatch(numTimesteps, numTimesPerObs, miniBatchSize, X,timesteps)
    % Create Mini-Batches of Trajectories
    s = randperm(numTimesteps - numTimesPerObs, miniBatchSize);
    x0 = dlarray(X(:, s));
    targets = zeros([size(X, 1), miniBatchSize, numTimesPerObs]);

    for i = 1:miniBatchSize
        targets(:, i, 1:numTimesPerObs) = X(:, s(i) + 1:(s(i) + numTimesPerObs));
        T(:,i)=timesteps(s(i) + 1:(s(i) + numTimesPerObs));
    end
end

function updateTrainingPlots(plottingTimesteps, xTrain, y)
    % Update Training Plots
    figure;
    subplot(3, 1, 1);
    plot(plottingTimesteps, xTrain(1, plottingTimesteps), 'r--');
    hold on;
    plot(plottingTimesteps, y(1, :), 'b-');
    hold off;
    xlabel('Time');
    ylabel('x(1)');
    title('Glucose');
    legend('Training Ground Truth', 'Predicted');

    subplot(3, 1, 2);
    plot(plottingTimesteps, xTrain(2, plottingTimesteps), 'r--');
    hold on;
    plot(plottingTimesteps, y(2, :), 'b-');
    hold off;
    xlabel('Time');
    ylabel('x(2)');
    title('Acting Insulin');
    legend('Training Ground Truth', 'Predicted');

    subplot(3, 1, 3);
    plot(plottingTimesteps, xTrain(3, plottingTimesteps), 'r--');
    hold on;
    plot(plottingTimesteps, y(3, :), 'b-');
    hold off;
    xlabel('Time');
    ylabel('x(3)');
    title('Insulin');
    legend('Training Ground Truth', 'Predicted');

    drawnow;
end

function neuralOdeParameters = initializeNeuralOdeParameters(stateSize, hiddenSize, externalInputSize)
% Initialize Parameters for Neural ODE
neuralOdeParameters = struct;

neuralOdeParameters.fc1 = struct;
% use the activation function from odeModel(~,y,theta) (tanh for now)
sz = [hiddenSize stateSize+externalInputSize];
neuralOdeParameters.fc1.Weights = initializeGlorot(sz, hiddenSize, stateSize);
neuralOdeParameters.fc1.Bias = initializeZeros([hiddenSize 1]);

neuralOdeParameters.fc2 = struct;
sz = [stateSize hiddenSize];
neuralOdeParameters.fc2.Weights = initializeGlorot(sz, stateSize, hiddenSize);
neuralOdeParameters.fc2.Bias = initializeZeros([stateSize 1]);

neuralOdeParameters.fc1
neuralOdeParameters.fc2

end

