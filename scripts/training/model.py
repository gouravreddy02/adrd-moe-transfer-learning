import numpy as np
import tensorflow as tf

from tensorflow.keras import Input, Model, regularizers
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Lambda, Softmax,
    Activation, Concatenate, Multiply, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.initializers import HeNormal, GlorotUniform


## Autoencoder

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu',
                     kernel_initializer=HeNormal(seed=42), kernel_regularizer=l1_l2(l1=1e-4, l2=1e-4))(input_layer)
    encoded = Dense(64, activation='relu',
                     kernel_initializer=HeNormal(seed=42), kernel_regularizer=l1_l2(l1=1e-4, l2=1e-4))(encoded)
    encoded = Dense(32, activation='relu',
                     kernel_initializer=HeNormal(seed=42), kernel_regularizer=l1_l2(l1=1e-4, l2=1e-4))(encoded)

    decoded = Dense(64, activation='relu',
                    kernel_initializer=HeNormal(seed=42))(encoded)
    decoded = Dense(128, activation='relu',
                    kernel_initializer=HeNormal(seed=42))(decoded)
    decoded = Dense(input_dim, activation='linear',
                    kernel_initializer=GlorotUniform(seed=42))(decoded)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    autoencoder.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='mse')

    return autoencoder, encoder

def add_noise(x, corruption_level=0.3, seed=None):
    rng = np.random.default_rng(seed)
    noise_x = x.copy()
    noise = rng.binomial(n=1, p=corruption_level, size=x.shape)
    noise_indices = np.where(noise == 1)
    noise_x[noise_indices] = 0
    return noise_x


## Pretrain Autoencoder

def pretrain_autoencoders(X_train_blocks_source, X_train_blocks_target=None, max_iter=500, batch_size=32, corruption_level=0.3):
    """
    Pretrains one autoencoder per expert block using optionally combined source and target data.

    Parameters:
        X_train_blocks_source (dict): Source domain feature blocks.
        X_train_blocks_target (dict or None): Optional target domain feature blocks.
        max_iter (int): Number of training iterations per autoencoder.
        batch_size (int): Batch size for training.
        corruption_level (float): Proportion of noise to add for denoising.

    Returns:
        dict: Pretrained encoder models keyed by expert name.
    """
    pretrained_encoders = {}
    expert_names = list(X_train_blocks_source.keys())

    for name in expert_names:
        print(f"Training autoencoder for block: {name}")

        X_train_source = X_train_blocks_source[name]

        if X_train_blocks_target is not None:
            X_train_target = X_train_blocks_target[name]
            X_combined = np.vstack([X_train_source, X_train_target])
        else:
            X_combined = np.vstack([X_train_source])

        input_dim = X_combined.shape[1]
        autoencoder, encoder = build_autoencoder(input_dim)
        losses = []

        for iteration in range(max_iter):
            batch_indices = np.random.choice(len(X_combined), size=batch_size, replace=False)
            batch_data = X_combined[batch_indices]
            noisy_batch = add_noise(batch_data, corruption_level=corruption_level, seed=iteration)
            loss = autoencoder.train_on_batch(noisy_batch, batch_data)
            losses.append(loss)

        pretrained_encoders[name] = encoder

    return pretrained_encoders


## Build MoE Model

def build_moe_model_adaptive_gates(X_train_blocks, pretrained_encoders, loss_fn='binary_crossentropy', learning_rate=1e-4, seed=42):
    """
    Builds a Mixture of Experts model with learnable gating.

    Parameters:
        X_train_blocks (dict): Dictionary mapping expert names to their input arrays.
        pretrained_encoders (dict): Pretrained encoder models keyed by expert name.
        loss_fn: Loss function for compilation.
        seed (int): Random seed for initializer and dropout.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        Model: Compiled Keras model.
    """
    expert_names = list(X_train_blocks.keys())
    expert_inputs = []
    expert_outputs = []

    for name in expert_names:
        n_features = X_train_blocks[name].shape[1]

        input_layer = Input(shape=(n_features,), name=f"{name}_input")
        expert_inputs.append(input_layer)

        # Use pretrained encoder
        encoder = pretrained_encoders[name]

        for layer in encoder.layers:
            layer.trainable = True

        encoded = encoder(input_layer)

        # Add dense layers after the encoder
        x = Dense(16,
                  kernel_initializer=HeNormal(seed=seed),
                  kernel_regularizer=regularizers.l1_l2(1e-4, 1e-4), name=f"{name}_dense_1")(encoded)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5, seed=seed)(x)

        out = Dense(1, activation='sigmoid',
                    kernel_regularizer=l1_l2(l1=1e-4, l2=1e-4), kernel_initializer=GlorotUniform(seed=42)
                    , name=f"{name}_output")(x)
        expert_outputs.append(out)

    # ====== Gating Network ======
    concat_inputs = Concatenate(name="concat_inputs")(expert_inputs)

    g = Dense(64, kernel_initializer=HeNormal(seed=seed),
            kernel_regularizer=regularizers.l1_l2(1e-4, 1e-4), name='gate_dense1')((concat_inputs))
    g = BatchNormalization(name='gate_bn1')(g)
    g = Activation('relu', name='gate_act1')(g)
    g = Dropout(0.5, name='gate_drop1', seed=seed)(g)

    gate_logits = Dense(len(expert_names), activation=None, name="gate_logits")(g)
    gate_probs = Softmax(name="gate_probs")(gate_logits)

    # ====== Weighted sum of expert outputs ======
    weighted_outputs = []
    for i in range(len(expert_names)):
        expert_out = expert_outputs[i]
        gate_prob = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[:, i], axis=-1))(gate_probs)
        weighted = Multiply()([expert_out, gate_prob])
        weighted_outputs.append(weighted)

    final_output = Add(name="final_output")(weighted_outputs)

    moe_model = Model(inputs=expert_inputs, outputs=final_output, name="MoE_AE")

    moe_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return moe_model


## Train MoE AE

def train_moe_model(model_or_tuple, X_train_blocks, y_train, X_val_blocks, y_val, epochs=40, batch_size=256, verbose=1, callbacks=[], class_weight=None):
    """
    Trains the MoE model with optional validation, callbacks, and class weighting.
    Accepts either the model directly or (model, extra_info) tuple.
    """
    if isinstance(model_or_tuple, tuple):
        model = model_or_tuple[0]
    else:
        model = model_or_tuple

    input_names = [input_tensor.name.split(":")[0] for input_tensor in model.inputs]

    input_list = [X_train_blocks[name.replace("_input", "")] for name in input_names]

    if X_val_blocks is not None and y_val is not None:
        val_input_list = [X_val_blocks[name.replace("_input", "")] for name in input_names]
        val_data = (val_input_list, y_val)
    else:
        val_data = None

    history = model.fit(
        x=input_list,
        y=y_train,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=verbose
    )

    return history


## Wrapper model for explainability

def build_moe_wrapper_model(moe_model, X_train_blocks, expert_names, wrapper_name="wrapped_moe"):
    """
    Wraps a multi-input MoE model into a single-input model using Lambda slicing.

    Parameters:
        moe_model (Model): The original trained MoE model.
        X_train_blocks (dict): Dict of expert-wise inputs (e.g. output of extract_blocks).
        expert_names (list): Ordered list of expert names (keys in X_train_blocks).
        wrapper_name (str): Optional name prefix for the wrapper model.

    Returns:
        wrapper_model (Model): A single-input wrapper model compatible with SHAP.
    """
    total_features = sum([X_train_blocks[expert].shape[1] for expert in expert_names])
    combined_input = Input(shape=(total_features,))

    expert_inputs = []
    start = 0
    for expert in expert_names:
        n_feat = X_train_blocks[expert].shape[1]
        slice_layer = Lambda(
            lambda x, s=start, e=start+n_feat: x[:, s:e],
            name=f"{expert}_slice"
        )
        expert_input = slice_layer(combined_input)
        expert_inputs.append(expert_input)
        start += n_feat

    moe_output = moe_model(expert_inputs)

    wrapper_model = Model(inputs=[combined_input], outputs=moe_output, name=wrapper_name)
    return wrapper_model
