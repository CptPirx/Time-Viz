__doc__ = """
Defines the bilinear and TABL models. 
Slightly modified version of the original code.
"""

from TABL.tabl_layers_tf import BL, TABL
from tensorflow import keras


class BL_class(keras.Model):
    def __init__(self, template, dropout=0.1, regularizer=None, constraint=None):
        super(BL_class, self).__init__()
        self.bl1 = BL(template[1], regularizer, constraint)
        self.activation = keras.layers.Activation('relu')
        self.dropout = keras.layers.Dropout(dropout)
        self.bl2 = BL(template[2], regularizer, constraint)
        self.out = keras.layers.Activation('softmax')

    def call(self, input):
        x = self.bl1(input)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.bl2(x)
        x = self.out(x)


def BL_model(template, dropout=0.1, regularizer=None, constraint=None):
    """
    Bilinear Layer network, refer to the paper https://arxiv.org/abs/1712.00975

    inputs
    ----
    template: a list of network dimensions including input and output, e.g., [[40,10], [120,5], [3,1]]
    dropout: dropout percentage
    regularizer: keras regularizer object
    constraint: keras constraint object

    outputs
    ------
    keras model object
    """
    inputs = keras.layers.Input(template[0])

    x = inputs
    for k in range(1, len(template) - 1):
        x = BL(template[k], regularizer, constraint)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(dropout)(x)

    x = BL(template[-1], regularizer, constraint)(x)
    outputs = keras.layers.Activation('softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    optimizer = keras.optimizers.Adam(0.01)

    model.compile(optimizer, 'categorical_crossentropy', ['acc'])

    return model


def TABL_model(template, dropout=0.1, projection_regularizer=None, projection_constraint=None,
               attention_regularizer=None, attention_constraint=None, loss=keras.losses.categorical_crossentropy):
    """
    Temporal Attention augmented Bilinear Layer network, refer to the paper https://arxiv.org/abs/1712.00975

    inputs
    ----
    template: a list of network dimensions including input and output, e.g., [[40,10], [120,5], [3,1]]
    dropout: dropout percentage
    projection_regularizer: keras regularizer object for projection matrices
    projection_constraint: keras constraint object for projection matrices
    attention_regularizer: keras regularizer object for attention matrices
    attention_constraint: keras constraint object for attention matrices

    outputs
    ------
    keras model object
    """
    inputs = keras.layers.Input(template[0])

    x = inputs
    for k in range(1, len(template) - 1):
        x = BL(template[k], projection_regularizer, projection_constraint)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(dropout)(x)

    x = TABL(template[-1], projection_regularizer, projection_constraint, attention_regularizer, attention_constraint)(x)
    outputs = keras.layers.Activation('softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    optimizer = keras.optimizers.Adam(0.01)
    # optimizer = keras.optimizers.SGD(learning_rate=meta.learning_rate, momentum=0.9, nesterov=True)

    model.compile(optimizer, loss, ['acc'])

    return model
