from Featurization.CompressionFunctions import create_autoencoder
create_autoencoder(input_shape = 1000, layers_structure=[2000,500], loss = 'mse',
                       lr = 0.001,  type_architecture = 'convoluted',
                       conv_reduction = 2, ## only if type_architecture is convoluted
                       )
