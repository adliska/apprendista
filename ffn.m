classdef ffn < handle
    properties
        Architecture
        ActFuncts
        Weights = {}
        Biases = {}
        Activations = {}
    end %properties

    methods
        function nn = ffn(Architecture, ActFuncts)
            nn.Architecture = Architecture;
            nn.ActFuncts = ActFuncts;
        end % constructor

        function randomInitialize(nn)
            for layer = 2:numel(nn.Architecture)
                nn.Weights{layer} = 0.1*rand(nn.Architecture(layer-1), ...
                    nn.Architecture(layer));
                nn.Biases{layer} = 0.1*rand(1, nn.Architecture(layer));
            end
        end % randomInitialize

        function train(nn, input, target, opts)
            for epoch = 1:opts.numepochs
                output = nn.forwardpass(input);
                
                dEdy = - (target - output);
                error = sum(sum((target-output).^2)) / size(input, 1);
                
                fprintf('Before epoch %d, total error %f \n', epoch, error);
                numcases = size(input, 1);
                
                % Derivatives of final-layer units
                layer = numel(nn.Architecture);
                dEdz = {};
                switch nn.ActFuncts(layer)
                    case AF.Sigmoid
                        dEdz{layer} = dEdy .* ...
                            nn.Activations{layer} .* (1 - nn.Activations{layer});
                    case AF.Linear
                        dEdz{layer} = dEdy;
                end
                
                % Backpropagation of derivatives
                for layer = numel(nn.Architecture)-1:-1:2
                    dEdz{layer} = dEdz{layer+1} * nn.Weights{layer+1}';
                    switch(nn.ActFuncts(layer))
                        case AF.Sigmoid
                            dEdz{layer} = dEdz{layer}.* nn.Activations{layer} .* ...
                                (1 - nn.Activations{layer});
                    end
                end

                % Gradient descent
                for layer = 2:numel(nn.Weights)
                    % Adjust weights                    
                    dEdw = nn.Activations{layer-1}' * dEdz{layer} / numcases;
                    nn.Weights{layer} = nn.Weights{layer} - ...
                        opts.learningRate*dEdw;
                    
                    % Adjust biases
                    dB = sum(dEdz{layer}) / numcases;
                    nn.Biases{layer} = nn.Biases{layer} - ...
                        opts.learningRate*dB;
                end % gradient descent

            end % epoch loop
            
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
        end % train

        function output = forwardpass(nn, input)      
            nn.Activations{1} = input;
            for layer = 2:numel(nn.Weights)
                switch nn.ActFuncts(layer)
                    case AF.Sigmoid
                        nn.Activations{layer} = 1./(1 + exp(bsxfun(@minus, ...
                            -nn.Activations{layer-1}*nn.Weights{layer}, ...
                            nn.Biases{layer})));
                    case AF.Linear
                        nn.Activations{layer} = bsxfun(@plus, ...
                            nn.Activations{layer-1}*nn.Weights{layer}, ...
                            nn.Biases{layer});
                end
            end
            output = nn.Activations{end};
        end % forwardpass 
    end %methods
end %classdef
