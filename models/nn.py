import tensorflow as tf


custom_objects = {}


def get_activation(activation):
    try:
        activation = tf.keras.activations.get(activation)
    except ValueError:
        activation = eval(activation)
    return activation


def fully_connected_block(
    units, activations, kernel_init='glorot_uniform', input_shape=None, output_shape=None, dropouts=None, name=None
):
    assert len(units) == len(activations)
    if dropouts:
        assert len(dropouts) == len(units)

    activations = [get_activation(a) for a in activations]

    layers = []
    for i, (size, act) in enumerate(zip(units, activations)):
        args = dict(units=size, activation=act, kernel_initializer=kernel_init)
        if i == 0 and input_shape:
            args['input_shape'] = input_shape

        layers.append(tf.keras.layers.Dense(**args))

        if dropouts and dropouts[i]:
            layers.append(tf.keras.layers.Dropout(dropouts[i]))

    if output_shape:
        layers.append(tf.keras.layers.Reshape(output_shape))

    args = {}
    if name:
        args['name'] = name

    return tf.keras.Sequential(layers, **args)


def fully_connected_residual_block(
    units,
    activations,
    input_shape,
    kernel_init='glorot_uniform',
    batchnorm=True,
    output_shape=None,
    dropouts=None,
    name=None,
):
    assert isinstance(units, int)
    if dropouts:
        assert len(dropouts) == len(activations)
    else:
        dropouts = [None] * len(activations)

    activations = [get_activation(a) for a in activations]

    def single_block(xx, units, activation, kernel_init, batchnorm, dropout):
        xx = tf.keras.layers.Dense(units=units, kernel_initializer=kernel_init)(xx)
        if batchnorm:
            xx = tf.keras.layers.BatchNormalization()(xx)
        xx = activation(xx)
        if dropout:
            xx = tf.keras.layers.Dropout(dropout)(xx)
        return xx

    input_tensor = tf.keras.Input(shape=input_shape)
    xx = input_tensor
    for i, (act, dropout) in enumerate(zip(activations, dropouts)):
        args = dict(units=units, activation=act, kernel_init=kernel_init, batchnorm=batchnorm, dropout=dropout)
        if len(xx.shape) == 2 and xx.shape[1] == units:
            xx = xx + single_block(xx, **args)
        else:
            assert i == 0
            xx = single_block(xx, **args)

    if output_shape:
        xx = tf.keras.layers.Reshape(output_shape)(xx)

    args = dict(inputs=input_tensor, outputs=xx)
    if name:
        args['name'] = name
    return tf.keras.Model(**args)


def concat_block(input1_shape, input2_shape, reshape_input1=None, reshape_input2=None, axis=-1, name=None):
    in1 = tf.keras.Input(shape=input1_shape)
    in2 = tf.keras.Input(shape=input2_shape)
    concat1, concat2 = in1, in2
    if reshape_input1:
        concat1 = tf.keras.layers.Reshape(reshape_input1)(concat1)
    if reshape_input2:
        concat2 = tf.keras.layers.Reshape(reshape_input2)(concat2)
    out = tf.keras.layers.Concatenate(axis=axis)([concat1, concat2])
    args = dict(inputs=[in1, in2], outputs=out)
    if name:
        args['name'] = name
    return tf.keras.Model(**args)


def conv_block(
    filters,
    kernel_sizes,
    paddings,
    activations,
    poolings,
    kernel_init='glorot_uniform',
    input_shape=None,
    output_shape=None,
    dropouts=None,
    name=None,
):
    assert len(filters) == len(kernel_sizes) == len(paddings) == len(activations) == len(poolings)
    if dropouts:
        assert len(dropouts) == len(filters)

    activations = [get_activation(a) for a in activations]

    layers = []
    for i, (nfilt, ksize, padding, act, pool) in enumerate(zip(filters, kernel_sizes, paddings, activations, poolings)):
        args = dict(filters=nfilt, kernel_size=ksize, padding=padding, activation=act, kernel_initializer=kernel_init)
        if i == 0 and input_shape:
            args['input_shape'] = input_shape

        layers.append(tf.keras.layers.Conv2D(**args))

        if dropouts and dropouts[i]:
            layers.append(tf.keras.layers.Dropout(dropouts[i]))

        if pool:
            layers.append(tf.keras.layers.MaxPool2D(pool))

    if output_shape:
        layers.append(tf.keras.layers.Reshape(output_shape))

    args = {}
    if name:
        args['name'] = name

    return tf.keras.Sequential(layers, **args)


def vector_img_connect_block(vector_shape, img_shape, block, vector_bypass=False, concat_outputs=True, name=None):
    vector_shape = tuple(vector_shape)
    img_shape = tuple(img_shape)

    assert len(vector_shape) == 1
    assert 2 <= len(img_shape) <= 3

    input_vec = tf.keras.Input(shape=vector_shape)
    input_img = tf.keras.Input(shape=img_shape)

    block_input = input_img
    if len(img_shape) == 2:
        block_input = tf.keras.layers.Reshape(img_shape + (1,))(block_input)
    if not vector_bypass:
        reshaped_vec = tf.tile(tf.keras.layers.Reshape((1, 1) + vector_shape)(input_vec), (1, *img_shape[:2], 1))
        block_input = tf.keras.layers.Concatenate(axis=-1)([block_input, reshaped_vec])

    block_output = block(block_input)

    outputs = [input_vec, block_output]
    if concat_outputs:
        outputs = tf.keras.layers.Concatenate(axis=-1)(outputs)

    args = dict(inputs=[input_vec, input_img], outputs=outputs,)

    if name:
        args['name'] = name

    return tf.keras.Model(**args)


def build_block(block_type, arguments):
    if block_type == 'fully_connected':
        block = fully_connected_block(**arguments)
    elif block_type == 'conv':
        block = conv_block(**arguments)
    elif block_type == 'connect':
        inner_block = build_block(**arguments['block'])
        arguments['block'] = inner_block
        block = vector_img_connect_block(**arguments)
    elif block_type == 'concat':
        block = concat_block(**arguments)
    elif block_type == 'fully_connected_residual':
        block = fully_connected_residual_block(**arguments)
    else:
        raise (NotImplementedError(block_type))

    return block


def build_architecture(block_descriptions, name=None, custom_objects_code=None):
    if custom_objects_code:
        print("build_architecture(): got custom objects code, executing:")
        print(custom_objects_code)
        exec(custom_objects_code, globals(), custom_objects)

    blocks = [build_block(**descr) for descr in block_descriptions]

    inputs = [tf.keras.Input(shape=i.shape[1:]) for i in blocks[0].inputs]
    outputs = inputs
    for block in blocks:
        outputs = block(outputs)

    args = dict(inputs=inputs, outputs=outputs)
    if name:
        args['name'] = name
    return tf.keras.Model(**args)
