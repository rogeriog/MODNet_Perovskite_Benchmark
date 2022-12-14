from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import pickle
from matplotlib import pyplot
import matplotlib
from modnet.preprocessing import MODData
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import BatchNormalization
import os, sys

def get_encoder_decoder(model, central_layer_name):
    layer = model.get_layer(central_layer_name)
    encoder_input = Input(model.input_shape[1:])
    encoder_output = encoder_input
    decoder_input = Input(layer.output_shape[1:])
    decoder_output = decoder_input

    encoder = True
    for layer in model.layers:
        if encoder:
            encoder_output = layer(encoder_output)
        else:
            decoder_output = layer(decoder_output)
        if layer.name == central_layer_name:
            encoder = False

    encoder_model = Model(encoder_input, encoder_output)
    decoder_model = Model(decoder_input, decoder_output)

    return encoder_model, decoder_model

def get_results_model(model,Xtest):
    from scipy.spatial.distance import correlation, cosine
    import numpy as np
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    y=Xtest
    ypred=model.predict(Xtest)
    resultsmodel=[]
    for metric in [correlation, cosine]:
        metric_result=np.array([metric(y[i],ypred[i]) for i in range(len(y))]).mean()
        print(metric.__name__,metric_result)
        resultsmodel.append(metric_result)
    results_mae=mean_absolute_error(y, ypred)
    resultsmodel.append(results_mae)
    print('MAE:',results_mae)
    rmse_result=np.sqrt(mean_squared_error(y, ypred))
    resultsmodel.append(rmse_result)
    print('RMSE:',rmse_result)
    r2_result=r2_score(y, ypred, multioutput='variance_weighted')
    resultsmodel.append(r2_result)
    print('r2:',r2_result)
    yzeros=np.zeros(y.shape)
    results_rmse0=np.sqrt(mean_squared_error(y, yzeros))
    resultsmodel.append(results_rmse0)
    print('RMSE zero-vector:',results_rmse0)
    return resultsmodel

def main():
    import numpy as np
    from matplotlib import pyplot
    import matplotlib
    matplotlib.use("Agg")
    ##preparing data
    from modnet.preprocessing import MODData
    data=MODData.load('../../DATAFILES/matbench_perovskites_moddata.pkl.gz')
    ## partial data to test
    # Xtoencode=data.df_featurized.filter(regex="Atomic*").sample(200,random_state=1)
    # y = data.df_targets['e_form'].sample(200,random_state=1)
    ## full data for script
    Xtoencode=data.df_featurized
    y = data.df_targets['e_form']
    ## eliminate all zero columns but keep track
    columns = Xtoencode.columns[(Xtoencode == 0).all()]
    with open("dropped_columns.txt", 'w') as f:
        text=""
        for column in columns:
            text+=str(column)+"\n"
        f.write(text)
    print(Xtoencode.shape)
    Xtoencode = Xtoencode.loc[:, Xtoencode.any()]
    print(Xtoencode.shape)
    
    # split into train test sets
    #X_train, X_test, y_train, y_test = train_test_split(Xtoencode, y, test_size=0.1, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(Xtoencode, y, test_size=0.1, random_state=2)
    # scale data
    t = MinMaxScaler()
    t.fit(X_train)
    X_train = t.transform(X_train)
    X_test = t.transform(X_test)
    # number of input columns
    n_inputs = Xtoencode.shape[1]
    name_encoder="PerovskitesMODNet"
    results_filename=f'EncoderResults_{name_encoder}.txt'
    if not os.path.exists(results_filename):
        with open(results_filename, 'w') as f:
            f.write(f"# Training {name_encoder} # Initial Number of Features: {n_inputs}\n")
            entries=['n_bottleneck_ratio','n_bottleneck', 'train_loss', 'val_loss', 'correlation',
                    'cosine dist', 'MAE', 'RMSE', 'R2', 'RMSE zero-vector']
            text=""
            for i in range(len(entries)):
                text+=f"{entries[i]:>18}|"
            text+='\n'
            f.write(text)
    for n_bottleneck_ratio in np.concatenate([np.arange(0.3,0.2,-0.01),np.arange(0.097,0.07,-0.003)]):
        results=[]
        # define encoder
        visible = Input(shape=(n_inputs,))
        e = Dense(n_inputs*2)(visible)
        e = BatchNormalization()(e)
        e = ReLU()(e)
        # define bottleneck
        n_bottleneck = int(n_inputs*n_bottleneck_ratio)
        print(f"Compressed layer size: {n_bottleneck}")
        bottleneck = Dense(n_bottleneck,name="bottleneck")(e)
        # define decoder
        d = Dense(n_inputs*2)(bottleneck)
        d = BatchNormalization()(d)
        d = ReLU()(d)
        # output layer
        output = Dense(n_inputs, activation='linear')(d)
        # define autoencoder model
        model = Model(inputs=visible, outputs=output)
        # compile autoencoder model
        model.compile(optimizer='adam', loss='mse')
        # model.summary()
        # fit the autoencoder model to reconstruct input
        history = model.fit(X_train, X_train, epochs=400, batch_size=16, verbose=2, validation_data=(X_test,X_test))
        # plot loss
        print(f"COMPRESSED VECTOR SIZE: {n_bottleneck}")    
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        print(f"Loss in the autoencoder: {history.history['val_loss'][-1]}")
        mse_error_train=history.history['loss'][-1]
        mse_error_val=history.history['val_loss'][-1]
        results=[np.round(n_bottleneck_ratio,4), n_bottleneck, 
                 mse_error_train, mse_error_val]
        results_model=get_results_model(model,X_test)
        results+=results_model
        with open(results_filename, 'a') as f:
            text=""
            for i in range(len(results)):
                text+=f"{np.round(results[i],8):18} "
            f.write(text)
            f.write('\n')
        pyplot.legend()
        pyplot.savefig(f"{name_encoder}_{np.round(n_bottleneck_ratio,4)}.png")
        pyplot.clf()
        # save full autoencoder model (without the decoder)
        model.save(f'{name_encoder}_AutoEncoder_compressratio_{np.round(n_bottleneck_ratio,4)}.h5')
        # define and save an encoder model (without the decoder)
        encoder,decoder = get_encoder_decoder(model, "bottleneck")
        encoder.save(f'{name_encoder}_encoder_compressratio_{np.round(n_bottleneck_ratio,4)}.h5')
        # define and save a decoder model (without the decoder)
        decoder.save(f'{name_encoder}_decoder_compressratio_{np.round(n_bottleneck_ratio,4)}.h5')
        print("Full AutoEncoder")
        model.summary()
        print("Encoder")
        encoder.summary()
        print("Decoder")
        decoder.summary() 
if __name__ == '__main__':
    main()
