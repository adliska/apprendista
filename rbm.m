classdef rbm < handle
    properties
        LayerSizes = [] % 1.visible, 2.hidden
        ActFuncts = []
        Weights = []
        Biases = {}
        Activations = []
    end %properties

    methods
        function rbm = rbm(LayerSizes, ActFuncts)
            rbm.LayerSizes = LayerSizes;
            rbm.ActFuncts = ActFuncts;
        end %constructor

        function randomInitialize(rbm)
            % Initializing symmetric weights and biases. 
            rbm.Weights = 0.1*randn(rbm.LayerSizes(1), ...
                rbm.LayerSizes(2));
            rbm.Biases{1} = zeros(1, rbm.LayerSizes(1));
            rbm.Biases{2} = zeros(1, rbm.LayerSizes(2));
        end %initialize

        function activations = train(rbm, input, opts)
            opts = parseOptions(opts);

            numcases = size(input, 1);
            numbatches = size(input, 3);
            
            fprintf('Training RBM %d-%d, %s-%s\n', ...
               rbm.LayerSizes(1), rbm.LayerSizes(2), ...
               rbm.ActFuncts(1).desc, rbm.ActFuncts(2).desc);
            fprintf(['learningRate: %s\n', ...
               'weightCost: %.4f\nmomentum:%s\n'], ...
               sprintf('%.4f ', opts.learningRate), opts.weightCost, ...
               sprintf('%.4f ', opts.momentum));
            fprintf(1, 'size(input): %d x %d x %d', ...
                size(input));
            
            h0probs = zeros(numcases,rbm.LayerSizes(2));
            h1probs = zeros(numcases,rbm.LayerSizes(2));
            vh0 = zeros(size(rbm.Weights));
            vh1 = zeros(size(rbm.Weights));
            dWeights = zeros(size(rbm.Weights));
            dVisBiases = zeros(size(rbm.Biases{1}));
            dHidBiases = zeros(size(rbm.Biases{2}));
            activations = zeros(numcases, rbm.LayerSizes(2), numbatches);

            for epoch = 1:opts.numepochs
                fprintf(1,'\nepoch %d ************************\n',epoch);
                errsum = 0;
                for batch = 1:numbatches
                    fprintf(1,'epoch %d batch %d ',epoch,batch);
                    
                    % Positive phrase - visible units
                    v0 = input(:,:,batch);
                    % Positive phase - hidden units
                    [h0probs, h0] = rbm.updateHidden(v0);
                    % Saving probabilities of hidden units (used in dae's)
                    activations(:, :, batch) = h0probs;
                    % Learning statistics for positive phase
                    vh0 = v0' * h0probs;
                                        
                    % Negative phase - visible units reconstruction
                    v1 = rbm.updateVisible(h0);
                    % Negative phase - hidden units
                    [h1probs, ~] = rbm.updateHidden(v1);
                    % Learning statistics for negative phase
                    vh1  = v1'*h1probs;

                    % Reconstruction error
                    err = sum(sum((v0-v1).^2 ));
                    errsum = errsum + err;
                    fprintf(1, 'error %f\n', err);
                    
                    % Update weights
                    dWeights = opts.momentum(epoch)*dWeights + ...
                        opts.learningRate(epoch)* ...
                        ((vh0-vh1)/numcases - opts.weightCost*rbm.Weights);
                    rbm.Weights = rbm.Weights + dWeights;

                    % Update biases of visible units
                    dVisBiases = opts.momentum(epoch)*dVisBiases + ...
                        (opts.learningRate(epoch)/numcases)*(sum(v0)-sum(v1));
                    rbm.Biases{1} = rbm.Biases{1} + dVisBiases;

                    % Update biases of hidden units
                    dHidBiases = opts.momentum(epoch)*dHidBiases + ...
                        (opts.learningRate(epoch)/numcases) * ...
                        (sum(h0probs)-sum(h1probs));
                    rbm.Biases{2} = rbm.Biases{2} + dHidBiases;
                end % batch
                
                fprintf(1, '\nepoch %d error %.1f\n', epoch, errsum);
            end; % epoch

            function popts = parseOptions(opts)
                popts = opts;
                if ~isfield(opts, 'numepochs') 
                    popts.numepochs = 100; 
                end
                if ~isfield(opts, 'learningRate')
                    popts.learningRate = 0.1*ones(popts.numepochs,1);
                end
                if ~isfield(opts, 'weightCost')
                    popts.weightCost = 0;
                end
                if ~isfield(opts, 'momentum')
                    popts.momentum = zeros(popts.numepochs, 1);
                end
            end; % parseOptions
    
        end %train

        function [hprobs, h] = updateHidden(rbm, v)
            switch rbm.ActFuncts(2)
                case AF.Sigmoid
                    hprobs = 1./(1 + exp(bsxfun(@minus, ...
                        -v*rbm.Weights, rbm.Biases{2})));
                    % Binarising hidden states
                    h = hprobs > rand(size(hprobs, 1), ...
                        rbm.LayerSizes(2));
                case AF.Linear
                    hprobs = bsxfun(@plus, ...
                        v*rbm.Weights, rbm.Biases{2});
                    % Adding Gaussian noise
                    h = hprobs + randn(size(hprobs,1), ...
                        rbm.LayerSizes(2));
            end
        end % updateHidden
            
        function v = updateVisible(rbm, h)
            switch rbm.ActFuncts(1)
                case AF.Sigmoid
                    v = 1./(1 + exp(bsxfun(@minus, ...
                        -h*rbm.Weights', rbm.Biases{1})));
                case AF.Linear
                    v = bsxfun(@plus, h*rbm.Weights', rbm.Biases{1});
            end
        end % updateVisible


    end %methods

end %class
