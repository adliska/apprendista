classdef rbm < handle
    properties
        LayerSizes = [] % 1.visible, 2.hidden
        ActivationFunctions = []
        Weights = []
        Biases = {}
        Activations = {}
    end %properties

    methods
        function rbm = rbm(LayerSizes, ActivationFunctions)
            rbm.LayerSizes = LayerSizes;
            rbm.ActivationFunctions = ActivationFunctions;
        end %constructor

        function randomInitialize(rbm)
            % Initializing symmetric weights and biases. 
            rbm.Weights = 0.1*randn(rbm.LayerSizes(1), ...
                rbm.LayerSizes(2));
            rbm.Biases{1} = zeros(1, rbm.LayerSizes(1));
            rbm.Biases{2} = zeros(1, rbm.LayerSizes(2));
        end %initialize

        function train(rbm, input, opts)
            opts = parseOptions(opts);

            numcases = size(input, 1);
            numbatches = size(input, 3);
            
            h0probs = zeros(numcases,rbm.LayerSizes(2));
            h1probs = zeros(numcases,rbm.LayerSizes(2));
            vh0 = zeros(size(rbm.Weights));
            vh1 = zeros(size(rbm.Weights));
            dWeights = zeros(size(rbm.Weights));
            dVisBiases = zeros(size(rbm.Biases{1}));
            dHidBiases = zeros(size(rbm.Biases{2}));

            for epoch = 1:opts.numepochs
                fprintf(1,'epoch %d\r',epoch); 
                errsum = 0;
                for batch = 1:numbatches
                    fprintf(1,'epoch %d batch %d ',epoch,batch);
                    
                    % Positive phrase
                    v0 = input(:,:,batch);
                    h0probs = 1./(1 + exp(bsxfun(@minus, -v0*rbm.Weights, rbm.Biases{2})));
                    vh0 = v0' * h0probs;
                    h0 = h0probs > rand(numcases,rbm.LayerSizes(2));
                    
                    % Negative phase
                    switch(rbm.ActivationFunctions(1))
                        case AF.Sigmoid
                            v1 = 1./(1 + exp(bsxfun(@minus,-h0*rbm.Weights', rbm.Biases{1})));
                        case AF.Linear
                            v1 = bsxfun(@minus, h0*rbm.Weights', rbm.Biases{1});
                    end
                    h1probs = 1./(1 + exp(bsxfun(@minus, -v1*rbm.Weights, rbm.Biases{2})));
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
                        (opts.learningRate(epoch)/numcases)*(sum(h0probs)-sum(h1probs));
                    rbm.Biases{2} = rbm.Biases{2} + dHidBiases;
                end % batch
                
                fprintf(1, '\nepoch %4i error %6.1f  \n', epoch, errsum);
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
    end %methods

end %class
