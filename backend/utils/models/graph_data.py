from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from spektral.layers import GCNConv, GlobalAvgPool
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

def build_gcn_model(input_shape, hidden, out_channels, dropout=0.0, lr=0.001):
    x_in = Input(shape=input_shape)             # Node features
    a_in = Input(shape=(None,), sparse=True)    # Adjacency matrix
    i_in = Input(shape=(), dtype='int32')       # Batch index

    x = GCNConv(hidden, activation='relu')([x_in, a_in])
    x = Dropout(dropout)(x)
    x = GCNConv(hidden, activation='relu')([x, a_in])
    x = GlobalAvgPool()([x, i_in])
    output = Dense(out_channels, activation='sigmoid')(x)

    model = Model(inputs=[x_in, a_in, i_in], outputs=output)
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss=BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model
