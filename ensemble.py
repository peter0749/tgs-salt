import sys
import glob
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, concatenate
from keras.losses import mean_squared_error
import tensorflow as tf

def ensemble(models, random_ensemble_weight=True):
    input_shapes = models[0].input_shape
    if not isinstance(input_shapes, list):
        input_shapes = [input_shapes]
    model_inputs  = [Input(shape=shape[1:]) for shape in input_shapes]
    input_branchs = [model_inputs]*len(models)
    output_ = []
    for n, (model, inp_b) in enumerate(zip(models, input_branchs)):
        model.name = 'ensemble_t_output_%d'%n
        out = model(inp_b)
        output_.append(out)
    ensemble_m_output = concatenate(output_, axis=-1)
    limit = np.sqrt(6/len(models))
    weights = np.random.uniform(-limit, limit, size=(1,1,len(models),1)) if random_ensemble_weight else np.ones((1,1,len(models),1),dtype=np.float32) / len(models)
    ensemble_m_output = Conv2D(1, (1,1), padding='valid', use_bias=False, weights=[weights]) (ensemble_m_output)
    return Model(model_inputs, ensemble_m_output, name='ensemble_model')

if __name__ == '__main__':
    models = [load_model(model_name, custom_objects={'bce': mean_squared_error, 'tf': tf, 'lovasz_loss': mean_squared_error, 'lb': mean_squared_error}) for model_name in glob.glob(str(sys.argv[1])+'/*.model')]
    model = ensemble(models)
    model.compile(optimizer='sgd', loss='mean_squared_error') # need to compile
    model.summary()
    model.save('ensemble.model')

