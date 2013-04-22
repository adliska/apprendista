classdef dae < handle
    properties
        FFN
    end % properties

    methods
        function dae = dae(LayerSizes, ActFuncts)
            dae.FFN = ffn([LayerSizes LayerSizes(end-1:-1:1)], ...
                [ActFuncts ActFuncts(end-1:-1:1)]);
        end %constructor

        function activations = pretrainlayer(dae, layer, input, opts)
            % Initialize an RBM with an identical pair of layers
            lrbm = rbm(dae.FFN.LayerSizes(layer-1:layer), ...
                dae.FFN.ActFuncts(layer-1:layer));
            lrbm.randomInitialize();
            % Pretrain and return the activations
            activations = lrbm.train(input, opts);
         
            % Copy pretrained weights and biases to the FFN
            numlayers = numel(dae.FFN.LayerSizes);
            dae.FFN.Weights{layer} = lrbm.Weights;
            dae.FFN.Weights{numlayers-layer+2} = lrbm.Weights';
            dae.FFN.Biases{layer} = lrbm.Biases{2};
            dae.FFN.Biases{numlayers-layer+2} = lrbm.Biases{1};
        end % pretrainlayer
    end % methods
end %classef
