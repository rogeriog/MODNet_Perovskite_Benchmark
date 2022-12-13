from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import os, pickle
from matplotlib import pyplot
import matplotlib
from modnet.preprocessing import MODData
def main():
    matplotlib.use("Agg")
    ##preparing data
    data=MODData.load('../../DATAFILES/matbench_perovskites_moddata.pkl.gz')
    OFMfeaturized=pickle.load(open('../OFM_featurization/OFM_featurizedDF.pkl','rb'))
    ## subset data to test script
    #Xtoencode=data.df_featurized.filter(regex="Atomic*").sample(200,random_state=1)
    #y = data.df_targets['e_form'].sample(200,random_state=1)
    ## full data for script
    Xtoencode=OFMfeaturized
    y = data.df_targets['e_form']
    # split into train test sets
    X_train, X_test, y_train, y_test = train_test_split(Xtoencode, y, test_size=0.1, random_state=1)
    # scale data
    t = MinMaxScaler()
    t.fit(X_train)
    X_train = t.transform(X_train)
    X_test = t.transform(X_test)
    # number of input columns
    n_inputs = Xtoencode.shape[1]
    results=[]
    name_encoder="PerovskitesMODNet"
    record_filename=f'EncoderResults_{name_encoder}.txt'
    if not os.path.exists(record_filename): 
        with open(record_filename, 'w') as f:
            f.write(f'''# Training {name_encoder} # Initial Number of Features: {n_inputs}
n_bottleneck_ratio n_bottleneck train_loss val_loss\n''')
    for n_bottleneck_ratio in list(np.arange(0.2,0,-0.025)):
        # define encoder
        visible = Input(shape=(n_inputs,))
        e = Dense(n_inputs*2)(visible)
        e = BatchNormalization()(e)
        e = ReLU()(e)
        # define bottleneck
        n_bottleneck = int(n_inputs*n_bottleneck_ratio)
        print(f"Compressed layer size: {n_bottleneck}")
        bottleneck = Dense(n_bottleneck)(e)
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
        results=(np.round(n_bottleneck_ratio,3), n_bottleneck, 
                 history.history['loss'][-1], history.history['val_loss'][-1])
        with open(record_filename, 'a') as f:
            f.write(' '.join(map(str,results)))
            f.write('\n')
        pyplot.legend()
        pyplot.savefig(f"{name_encoder}_{np.round(n_bottleneck_ratio,2)}.png")
        pyplot.clf()
        # define an encoder model (without the decoder)
        encoder = Model(inputs=visible, outputs=bottleneck)
        # plot_model(encoder, 'encoder.png', show_shapes=True)
        # save the encoder to file
        encoder.save(f'encoder_compressionratio_{np.round(n_bottleneck_ratio,2)}.h5')

if __name__ == '__main__':
    main()
