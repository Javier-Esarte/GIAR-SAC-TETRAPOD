import numpy as np

data_type = np.float64  # Data type used in internal arrays


class NNLayer:
    def __init__(self, layer_type, output_shape, name="", initializer=("Normal", 0.0, 0.001)):
        if not isinstance(output_shape, tuple):
            if isinstance(output_shape, list): output_shape = tuple(output_shape)
            elif isinstance(output_shape, int): output_shape = (output_shape,)
            else: print("Error: Parameter output shape must be an integer or a list/tuple of integers")

        # Initialize internal variables
        self.name, self.in_shape, self.out_shape = name, None, output_shape         # layer info
        self.x, self.w, self.b, self.z, self.a = None, None, None, None, None       # Forward pass
        self.dl_dx, self.dl_dw, self.dl_db, self.dl_da = None, None, None, None     # Backward pass
        self.w_aux, self.b_aux, self.o_aux, self.a_aux = None, None, None, None     # Optimizer
        self.input_layers, self.input_offset, self.shared_layer = None, None, None  # Related layers
        self.frozen = False  # Indicates whether the layer weights should be trained (False) or not (True)
        self.critical = False  # Indicates whether the layer weights should be trained (False) or not (True)

        # Assign the corresponding layer functions for forward and backward propagation
        if not isinstance(layer_type, str): print("Error: Parameter layer_type must be of type string"); return
        layer_type = layer_type.lower()
        if layer_type == 'input':
            self.forward, self.backward = self.__forward_input, self.__backward_input
            self.in_shape = output_shape
        elif layer_type == 'linear':
            self.forward, self.backward = self.__forward_linear, self.__backward_linear
        elif layer_type == 'relu':
            self.forward, self.backward = self.__forward_relu, self.__backward_relu
        elif layer_type == 'tanh':
            self.forward, self.backward = self.__forward_tanh, self.__backward_tanh
        elif layer_type == 'exponential':
            self.forward, self.backward = self.__forward_exp, self.__backward_exp
        elif layer_type == 'sigmoid':
            self.forward, self.backward = self.__forward_sigmoid, self.__backward_sigmoid
        elif layer_type == 'minimum':
            self.forward, self.backward = self.__forward_minimum, self.__backward_minimum
            self.frozen = True
        elif layer_type == 'gaussian':
            self.forward, self.backward = self.__forward_gaussian, self.__backward_gaussian
            self.frozen = True
        elif layer_type == 'entropy':
            self.forward, self.backward = self.__forward_entropy, self.__backward_entropy
            self.frozen = True
        else:
            print(
                "Error: Unsupported layer type: " + layer_type + ". Use: 'input', 'linear', 'tanh', 'sigmoid', 'minimum' or 'gaussian'"); return
        self.layer_type = layer_type

        # Initializer
        if (not isinstance(layer_type, str)) or (len(initializer) != 3):
            print("Error: 3 parameters required for initializer (distribution (string), mean, standard deviation)")
        self.i_name, self.i_mean, self.i_stdv = initializer[0].lower(), initializer[1], initializer[2]
        if self.i_name == 'normal' or self.i_name == 'gaussian':
            self.initializer = self.__initializer_gaussian
        elif self.i_name == 'uniform':
            self.initializer = self.__initializer_uniform
        else:
            print("Error: Unsupported initializer:" + initializer[0] + ". Use: 'uniform' or 'normal'")

    def __call__(self, x):
        if self.in_shape is not None: print("Warning: Layer's input connection was overwritten")
        if isinstance(x, (list, tuple)):
            in_shape = list(x[0].out_shape)
            last_dim = 0
            for layer in x:
                if layer.out_shape[0:-1] != in_shape[0:-1] and len(in_shape) > 1:
                    print("Error: Input layers dimensions do not match:", layer.out_shape[0:-1], "!=", in_shape[0:-1])
                    return None
                last_dim += layer.out_shape[-1]
            in_shape[-1] = last_dim
            self.in_shape = tuple(in_shape)
            self.input_layers = list(x)
            self.input_offset = np.cumsum([input_layer.out_shape[-1] for input_layer in self.input_layers])
        else:
            self.in_shape = x.out_shape
            self.input_layers = [x]
            self.input_offset = np.cumsum([input_layer.out_shape[-1] for input_layer in self.input_layers])

        # Verify compatibility with the shared layer
        if self.shared_layer:
            if self.shared_layer.in_shape[-1] != self.in_shape[-1]:
                print("Error: New layer's input shape is not compatible with its shared layer's input shape")
                return None
        # Initialize the layers weights
        else:
            w_shape, b_shape = (self.in_shape[-1], self.out_shape[-1]), (1, self.out_shape[-1])
            self.w, self.dl_dw = np.zeros(w_shape, dtype=data_type), np.zeros(w_shape, dtype=data_type)
            self.b, self.dl_db = np.zeros(b_shape, dtype=data_type), np.zeros(b_shape, dtype=data_type)

        return self

    @classmethod
    def copy(cls, name, source_layer, share_weights=False):
        initializer = (source_layer.i_name, source_layer.i_mean, source_layer.i_stdv)
        layer = cls(source_layer.layer_type, source_layer.out_shape, name=name, initializer=initializer)
        layer.frozen, layer.critical = source_layer.frozen, source_layer.critical

        # Shared layer's variables
        if share_weights:
            layer.in_shape = source_layer.in_shape
            layer.w, layer.dl_dw = source_layer.w, source_layer.dl_dw
            layer.b, layer.dl_db = source_layer.b, source_layer.dl_db
            layer.shared_layer = source_layer.shared_layer if source_layer.shared_layer else source_layer

        return layer

    # Activation Functions and Derivatives
    def __forward_input(self):
        return

    def __backward_input(self):
        return

    def __forward_linear(self):
        self.z = np.add(np.matmul(self.x, self.w), self.b)
        self.a = self.z

    def __backward_linear(self):
        dl_dz = self.dl_da
        self.dl_dx = np.matmul(dl_dz, self.w.T)
        if not self.frozen:
            np.divide(np.matmul(self.x.T, dl_dz, out=self.dl_dw), np.prod(self.x.shape[0:-1]), out=self.dl_dw)
            np.mean(dl_dz, axis=0, keepdims=True, out=self.dl_db)

    def __forward_relu(self):
        self.z = np.add(np.matmul(self.x, self.w), self.b)
        self.a = np.multiply(self.z, self.z > 0)

    def __backward_relu(self):
        #dl_dz = np.multiply(self.dl_da, np.multiply(0.99, self.z > 0) + 0.01)
        dl_dz = np.multiply(self.dl_da, np.multiply(1, self.z > 0))
        self.dl_dx = np.matmul(dl_dz, self.w.T)
        if not self.frozen:
            np.divide(np.matmul(self.x.T, dl_dz, out=self.dl_dw), np.prod(self.x.shape[0:-1]), out=self.dl_dw)
            np.mean(dl_dz, axis=0, keepdims=True, out=self.dl_db)

    def __forward_tanh(self):
        self.z = np.add(np.matmul(self.x, self.w), self.b)
        self.a = np.tanh(self.z)

    def __backward_tanh(self):
        dl_dz = np.multiply(self.dl_da, 1 - np.square(np.tanh(self.z)))
        self.dl_dx = np.matmul(dl_dz, self.w.T)
        if not self.frozen:
            np.divide(np.matmul(self.x.T, dl_dz, out=self.dl_dw), np.prod(self.x.shape[0:-1]), out=self.dl_dw)
            np.mean(dl_dz, axis=0, keepdims=True, out=self.dl_db)

    def __forward_exp(self):
        self.z = np.add(np.matmul(self.x, self.w), self.b)
        self.a = np.exp(self.z)
        self.a = np.clip(self.a, 1E-9, 10)

    def __backward_exp(self):
        dl_dz = np.multiply(self.dl_da, self.a)
        self.dl_dx = np.matmul(dl_dz, self.w.T)
        if not self.frozen:
            np.divide(np.matmul(self.x.T, dl_dz, out=self.dl_dw), np.prod(self.x.shape[0:-1]), out=self.dl_dw)
            np.mean(dl_dz, axis=0, keepdims=True, out=self.dl_db)

    def __forward_sigmoid(self):
        self.z = np.add(np.matmul(self.x, self.w), self.b)
        self.a = np.divide(1, 1 + np.exp(-self.z))

    def __backward_sigmoid(self):
        dl_dz = np.multiply(self.dl_da, np.divide(np.exp(-self.z), np.square(1 + np.exp(-self.z))))
        self.dl_dx = np.matmul(dl_dz, self.w.T)
        if not self.frozen:
            np.divide(np.matmul(self.x.T, dl_dz, out=self.dl_dw), np.prod(self.x.shape[0:-1]), out=self.dl_dw)
            np.mean(dl_dz, axis=0, keepdims=True, out=self.dl_db)

    def __forward_minimum(self):
        self.a = np.min(self.x, axis=1, keepdims=True)
#        print("\n\nForward\n",self.x[0:4],self.a[0:4])

    def __backward_minimum(self):
        self.dl_dx = np.multiply(np.array(self.x == self.a), self.dl_da)
#       print("\n\nBackward\n",self.a[0:4], "\n", self.dl_dx[0:4])

    def __forward_gaussian(self):
        u, s = self.input_layers[0].a, self.input_layers[1].a
        self.z = np.random.normal(size=s.shape)
        self.a = np.tanh(u + np.multiply(s, self.z))

    def __backward_gaussian(self):
        da_dz = [1-np.square(self.a), np.multiply(1-np.square(self.a), self.z)]
        self.dl_dx = np.concatenate([np.multiply(da_dz[0], self.dl_da), np.multiply(da_dz[1], self.dl_da)], axis=-1)

    def __forward_entropy(self):
        # Get the mean, standard deviation and the random normal arrays and compute the action
        u, s, r = self.input_layers[1].input_layers[0].a, self.input_layers[1].input_layers[1].a, self.input_layers[1].z
        a = u + r * s
        alpha = 0.0 if self.z is None else self.z
        # Compute the entropy
        h = np.sum(0.5 * np.square(r) + np.log(np.sqrt(2 * np.pi) * s) - 2 * (a + np.log(0.5 + 0.5 * np.exp(-2 * a))),
                   axis=-1, keepdims=True)
        self.a = np.add(self.input_layers[0].a, alpha * h)

    def __backward_entropy(self):
        # Get the mean, standard deviation and the random normal arrays and compute the derivatives
        u, s, r = self.input_layers[1].input_layers[0].a, self.input_layers[1].input_layers[1].a, self.input_layers[1].z
        da_du = -2 + np.divide(4, 1 + np.exp(2 * (u + r * s)))
        alpha = 0.0 if self.z is None else self.z
        dl_du = np.multiply(alpha * self.dl_da, da_du)
        dl_ds = np.multiply(alpha * self.dl_da, np.divide(1, s) + np.multiply(r, da_du))
        np.add(dl_du, self.input_layers[1].input_layers[0].dl_da, out=self.input_layers[1].input_layers[0].dl_da)
        np.add(dl_ds, self.input_layers[1].input_layers[1].dl_da, out=self.input_layers[1].input_layers[1].dl_da)
        self.dl_dx = np.concatenate([self.dl_da, np.zeros(self.input_layers[1].a.shape)], axis=1)


    # Initializers
    def __initializer_gaussian(self, shape):
        return np.random.normal(self.i_mean, self.i_stdv, size=shape)

    def __initializer_uniform(self, shape):
        return np.random.uniform(self.i_mean-self.i_stdv*self.i_stdv, self.i_mean+self.i_stdv*self.i_stdv, size=shape)


class NeuralNetwork:
    def __init__(self, name, input_layers, output_layers):
        # Adjust inputs
        input_layers = input_layers if isinstance(input_layers, (list, tuple)) else [input_layers]
        output_layers = output_layers if isinstance(output_layers, (list, tuple)) else [output_layers]

        # Initialize internal variables
        self.name = name  # Network name
        self.input_layers = input_layers  # List of input layers
        self.layers = []  # List of layer objects, ordered by computation priority
        self.output_layers = output_layers  # List of output layers

        # Return if called by a class method
        if (input_layers[0] is None) or (output_layers[0] is None): return

        # Determine the computation order of the layers in order to fulfill all dependencies
        layers = output_layers.copy()
        idx, idx_last, circ_dep_count = 0, 0, 0
        while idx <= idx_last:
            # Get the next layer and increment the index counter
            layer, idx = layers[idx], idx + 1

            # If the layer is an input layer, skip it
            if layer in input_layers: continue

            # Check if the layer does not have dependencies an input layer is missing, throw an error
            if layer.input_layers is None: print("Error: Layer", layer.name, "is not connected to any input"); return

            # Evaluate the layer dependencies
            for input_layer in layer.input_layers:
                # If the dependency was not added to the list of layers add it and increase the last index
                if input_layer not in layers:
                    layers.append(input_layer)
                    idx_last += 1
                # If the layer depends on itself, throw a self dependency error
                elif input_layer == layer:
                    print("Error: Self dependency detected for layer", layer.name)
                    return
                # If already present move it to the bottom of the list and adjust the index if necessary
                else:
                    # If the layer was evaluated, decrease the index and increase the circular dependency counter
                    if idx > layers.index(input_layer):
                        idx, circ_dep_count = idx - 1, circ_dep_count + 1
                        if circ_dep_count > 1000: print("Error: Circular dependency detected"); return
                    layers.remove(input_layer)
                    layers.append(input_layer)

        # Reverse layer list order and copy all layers to preserve the connections even if the originals are modified
        layer_idx = dict(zip(reversed(layers), range(len(layers))))
        for layer in reversed(layers):
            net_layer = NNLayer.copy(layer.name, source_layer=layer, share_weights=True)
            if layer.input_layers:
                net_layer.input_layers = [self.layers[layer_idx[input_layer]] for input_layer in layer.input_layers]
                net_layer.input_offset = layer.input_offset
            self.layers.append(net_layer)
        self.input_layers = [self.layers[layer_idx[layer]] for layer in self.input_layers]
        self.output_layers = [self.layers[layer_idx[layer]] for layer in self.output_layers]

        # Remove all input layers from the layer list
        for layer in self.input_layers:
            if layer not in self.layers: print("Error: Input layer", layer.name, "is not connected to any output layer")
            self.layers.remove(layer)

        # Initialize all layers
        self.reset()

    @classmethod
    def connect(cls, name, input_network, output_network, keep_outputs=False):
        #Make a copy of the input and output networks
        input_network = NeuralNetwork.copy(name, input_network, share_weights=True)
        output_network = NeuralNetwork.copy(name, output_network, share_weights=True)

        # Create a new network object
        network = cls(name, None, None)

        # Assign the combined layers of both networks, and their dependencies
        network.input_layers = input_network.input_layers + output_network.input_layers
        network.output_layers = input_network.output_layers + output_network.output_layers
        network.layers = input_network.layers + output_network.layers

        # Find the inputs of the output network that should be replaced by the inputs or outputs of the input network
        replace_list = []
        for input_out in output_network.input_layers:
            # Duplicate input layer
            for input_in in input_network.input_layers:
                if (input_out.name == input_in.name) and (input_out.in_shape == input_in.in_shape):
                    network.input_layers.remove(input_out)
                    replace_list.append((input_out, input_in))
            # Connected input and output layers
            for output_in in input_network.output_layers:
                if (input_out.name == output_in.name) and (input_out.in_shape == output_in.out_shape):
                    network.input_layers.remove(input_out)
                    if not keep_outputs: network.output_layers.remove(output_in)
                    replace_list.append((input_out, output_in))
        if not replace_list: print("Error: No matching layers to connect (name or shape do not match)."); return

        # Replace all references to the found layers
        for layer in network.layers:
            for idx in range(len(layer.input_layers)):
                for replaced_layer, replacing_layer in replace_list:
                    if layer.input_layers[idx] == replaced_layer:
                        layer.input_layers[idx] = replacing_layer

        return network

    @classmethod
    def copy(cls, name, src_network, share_weights=False):
        # Create a new network object
        network = cls(name, None, None)

        # Join all layers in a single list and index it
        src_layers = src_network.input_layers + src_network.layers
        src_layer_idx = dict(zip(src_layers, range(len(src_layers))))
        layers = []

        # Create a copy of each layer and update their connections
        for layer in src_layers:
            net_layer = NNLayer.copy(layer.name, source_layer=layer, share_weights=share_weights)
            net_layer.in_shape = layer.in_shape
            if layer.input_layers:
                net_layer.input_layers = [layers[src_layer_idx[input_layer]] for input_layer in layer.input_layers]
                net_layer.input_offset = layer.input_offset
            if not share_weights:
                w_shape, b_shape = (net_layer.in_shape[-1], net_layer.out_shape[-1]), (1, net_layer.out_shape[-1])
                net_layer.w, net_layer.dl_dw = np.zeros(w_shape, dtype=data_type), np.zeros(w_shape, dtype=data_type)
                net_layer.b, net_layer.dl_db = np.zeros(b_shape, dtype=data_type), np.zeros(b_shape, dtype=data_type)
            layers.append(net_layer)
        network.input_layers = [layers[src_layer_idx[layer]] for layer in src_network.input_layers]
        network.output_layers = [layers[src_layer_idx[layer]] for layer in src_network.output_layers]
        network.layers = [layers[src_layer_idx[layer]] for layer in src_network.layers]

        network.reset()

        return network

    def reset(self, seed=None):
        if seed is not None: np.random.seed(seed)
        for layer in self.layers:
            np.copyto(layer.w, layer.initializer((layer.in_shape[-1], layer.out_shape[-1])), casting='same_kind')
            np.copyto(layer.b, np.zeros((1,layer.out_shape[-1]), dtype=data_type), casting='same_kind')
            layer.w_aux, layer.b_aux, layer.a_aux = None, None, None

    def update(self, src_network, update_coefficient):
        for layer, src_layer in zip(self.layers, src_network.layers):
            np.add((1 - update_coefficient) * layer.w, update_coefficient * src_layer.w, out=layer.w)
            np.add((1 - update_coefficient) * layer.b, update_coefficient * src_layer.b, out=layer.b)

    def freeze_layers(self, layers_names, freeze=True):
        for layer in self.layers:
            if layer.name in layers_names:
                layer.frozen = freeze

    def compute(self, x):
        # Adjust function's arguments
        net_inputs = x if isinstance(x, (list, tuple)) else [x]
        m = np.array(net_inputs[0]).reshape(-1, *self.input_layers[0].in_shape).shape[0]

        #print(self.name, self.input_layers, net_inputs)
        # Validate and load the inputs to each input layer
        for idx, layer, x in zip(range(len(self.input_layers)), self.input_layers, net_inputs):
            if not isinstance(x, np.ndarray): x = np.array(x, dtype=data_type)
            if (x.shape == layer.in_shape) or (len(x.shape) == 1): x = x.reshape(-1, *layer.in_shape)
            if x.shape[0] != m: print("Error: Input X[" + str(idx) + "] does not match the number of samples of X[0]")
            if x.shape[1:] != layer.in_shape: print("Error: Input X[" + str(idx) + "] does not match the shape of input layer", layer.name)
            layer.a = x

        # Compute the forward pass for each layer
        for layer in self.layers:
            layer.x = np.concatenate([input_layer.a for input_layer in layer.input_layers], axis=-1)
            layer.forward()

        # Return the output layers' output
        return self.output_layers[0].a if len(self.output_layers) == 1 else [layer.a for layer in self.output_layers]

    def train(self, x, y, loss_function, optimizer, regularization, epochs, n_batches=1):
        # Adjust function's arguments
        net_inputs = x if isinstance(x, (list, tuple)) else [x]
        net_outputs = y if isinstance(y, (list, tuple)) else [y]
        m = np.array(net_inputs[0]).reshape(-1, *self.input_layers[0].in_shape).shape[0]
        n_batches = m if n_batches > m else n_batches

        # Validate the network's inputs and outputs
        for idx, layer, x in zip(range(len(self.input_layers)), self.input_layers, net_inputs):
            if not isinstance(x, np.ndarray): x = np.array(x, dtype=data_type)
            if (x.shape == layer.in_shape) or (len(x.shape) == 1): x = x.reshape(-1, *layer.in_shape)
            if x.shape[0] != m: print("Error: Input X[" + str(idx) + "] does not match the number of samples of X[0]")
            if x.shape[1:] != layer.in_shape: print("Error: Input X[" + str(idx) + "] does not match the shape of input layer", layer.name)
        for idx, layer, y in zip(range(len(self.output_layers)), self.output_layers, net_outputs):
            if not isinstance(y, np.ndarray): y = np.array(y, dtype=data_type)
            if (y.shape == layer.out_shape) or (len(y.shape) == 1): y = y.reshape(-1, *layer.out_shape)
            if y.shape[0] != m: print("Error: Expected output Y[" + str(idx) + "] does not match the number of samples of X[0]")
            if y.shape[1:] != layer.out_shape: print( "Error: Expected output Y[" + str(idx) + "] does not match the shape of output layer", layer.name)

        # Initialize required variables
        epoch_x, epoch_y = net_inputs, net_outputs
        loss_evolution = np.zeros((epochs, n_batches))
        batch_idxs = np.floor(np.linspace(0, m, n_batches+1)).astype(int)
        forward_layers, backward_layers = self.layers, tuple(reversed(self.layers))
        loss_function, loss_derivative = self.__get_loss_fuction(loss_function)
        optimizer = self.__get_optimizer(optimizer)
        reg_L1, reg_L2, reg_DO = self.__get_regularization(regularization)

        # For each epoch
        for epoch in range(epochs):
            # Shuffle the inputs and form batches when appropriate
            if n_batches > 1:
                idx_shuffle = np.random.permutation(m)
                epoch_x = [x[idx_shuffle] for x in net_inputs]
                epoch_y = [y[idx_shuffle] for y in net_outputs]

            # Train n batches
            for batch in range(n_batches):
                # Get the batch input and output
                batch_x = [x[batch_idxs[batch]:batch_idxs[batch+1]] for x in epoch_x]
                batch_y = [y[batch_idxs[batch]:batch_idxs[batch+1]] for y in epoch_y]

                # Assign the inputs to the input layers and prepare dl_da for the backward pass
                for layer, x in zip(self.input_layers, batch_x):
                    layer.a = x
                    layer.dl_da = np.zeros(layer.a.shape)

                # Compute the forward pass
                for layer in forward_layers:
                    # Concatenate inputs, perform the layer's forward pass and prepare dl_da for the backward pass
                    layer.x = np.concatenate([input_layer.a for input_layer in layer.input_layers], axis=-1)
                    layer.forward()
                    layer.dl_da = np.zeros(layer.a.shape)

                    # Compute dropout if it is enabled and the layer is not critical
                    if (not reg_DO) or layer.critical: continue
                    layer.a_aux = np.array(np.random.random_sample(layer.out_shape) < reg_DO, dtype=data_type)/reg_DO
                    np.multiply(layer.a, layer.a_aux, out=layer.a)

                # Compute the loss function
                a = [layer.a for layer in self.output_layers]
                loss_evolution[epoch,batch] = np.mean(loss_function(batch_y, a))

                # Compute and assign the derivative dL/dA for the output layers
                for layer, dl_da in zip(self.output_layers, loss_derivative(batch_y, a)):
                    np.add(layer.dl_da, dl_da, out=layer.dl_da)

                # Compute the backward pass (Backpropagation)
                for layer in backward_layers:
                    # Compute dropout if it is enabled and the layer is not critical
                    if reg_DO and not layer.critical: np.multiply(layer.dl_da, layer.a_aux, out=layer.dl_da)

                    # Perform the layer's backward pass and add the current layer's dL/dX to its input layers' dL/dA
                    layer.backward()
                    for input_layer, dl_dx in zip(layer.input_layers, np.split(layer.dl_dx, layer.input_offset, axis=-1)):
                        np.add(input_layer.dl_da, dl_dx, out=input_layer.dl_da)

                    # If the layers weights can be modified, update them with the optimizer
                    if not layer.frozen:
                        if reg_L1:
                            np.add(layer.dl_dw, (reg_L1/batch_x[0].shape[0]) * np.sign(layer.dl_dw), out=layer.dl_dw)
                            np.add(layer.dl_db, (reg_L1/batch_x[0].shape[0]) * np.sign(layer.dl_db), out=layer.dl_db)
                        if reg_L2:
                            np.add(layer.dl_dw, (reg_L2/batch_x[0].shape[0]) * layer.w, out=layer.dl_dw)
                            np.add(layer.dl_db, (reg_L2/batch_x[0].shape[0]) * layer.b, out=layer.dl_db)
                        optimizer(layer)

        return loss_evolution

    def __get_loss_fuction(self, loss_function):
        if isinstance(loss_function, str):
            loss_function = loss_function.lower()
            if loss_function == 'meansquarederror' or loss_function == 'mse':
                return self.__loss_mean_squared_error, self.__loss_mean_squared_error_derivative
            elif loss_function == 'maximum' or loss_function == 'max':
                return self.__loss_maximize, self.__loss_maximize_derivative
            else:
                print("Error: Requested loss function", loss_function, "unavailable. Use 'MSE', 'MAX'.")
        elif isinstance(loss_function, (tuple, list)):
            return loss_function[0], loss_function[1]
        else:
            print("Error: Parameter loss_function should be either a string with the name of a loss function, or a "
                  "tuple of a function and its derivative (function(y_true, ypred), derivative(y_true, ypred)).")

    def __get_optimizer(self, optimizer):
        if not isinstance(optimizer, (list, tuple)):
            print("Error: Parameter optimizer should be a tuple of (optimizer type, list of parameters)")
            return self.__opt_gradient_descent
        if not isinstance(optimizer[0], str):
            print("Error: Parameter optimizer's first element must be a string")
            return self.__opt_gradient_descent

        optimizer = list(optimizer)
        if not isinstance(optimizer[1], (list, tuple)): optimizer[1] = [optimizer[1]]

        optimizer[0] = optimizer[0].lower()
        if optimizer[0] == 'gd':
            if len(optimizer[1]) > 1: print("Error: Optimizer Gradient Descent accepts 1 parameter: learning rate")
            self.opt_vars = optimizer[1]
            return self.__opt_gradient_descent
        elif optimizer[0] == 'adam':
            if len(optimizer[1]) > 3: print("Error: Optimizer Adam accepts 3 parameters: learning rate, beta1, beta2")
            self.opt_vars = [
                optimizer[1][0],
                optimizer[1][1] if len(optimizer[1]) >= 2 else 0.9,
                optimizer[1][2] if len(optimizer[1]) >= 3 else 0.999
            ]
            for layer in self.layers:
                if (layer.w_aux is None) or (layer.b_aux is None) or (layer.o_aux is None):
                    layer.w_aux = [np.zeros(layer.w.shape, dtype=data_type), np.zeros(layer.w.shape, dtype=data_type)]
                    layer.b_aux = [np.zeros(layer.b.shape, dtype=data_type), np.zeros(layer.b.shape, dtype=data_type)]
                    layer.o_aux = [1.0, 1.0]
            return self.__opt_adaptive_moment_gradient_descent
        else:
            print("Error: Requested optimizer", optimizer[0], "unavailable. Use 'GD' or 'ADAM'.")
        return self.__opt_gradient_descent

    def __get_regularization(self, regularization):
        reg_L1, reg_L2, reg_DO = False, False, False
        if regularization is None:
            return reg_L1, reg_L2, reg_DO
        if not isinstance(regularization, (list, tuple)):
            print("Error: Parameter regularization should be a list of lists of (regularization type, weight)")
            return reg_L1, reg_L2, reg_DO
        if not isinstance(regularization[0], (list, tuple)): regularization = [regularization]
        for reg, reg_w in regularization:
            if not isinstance(reg, str):
                print("Error: Parameter regularization's regularization type must be a string")
                return reg_L1, reg_L2, reg_DO
            reg = reg.lower()
            if reg == 'l1':                             reg_L1 = float(reg_w)
            elif reg == 'l2':                           reg_L2 = float(reg_w)
            elif (reg == 'do') or (reg == 'dropout'):   reg_DO = float(1-reg_w)
            else: print("Error: Requested regularization", reg, "unavailable. Use 'L1', 'L2', 'Dropout'.")
        return reg_L1, reg_L2, reg_DO


    @staticmethod
    def __loss_mean_squared_error(y_true, y_pred):
        return [np.square(y_true[0] - y_pred[0])]

    @staticmethod
    def __loss_mean_squared_error_derivative(y_true, y_pred):
        return [2*(y_pred[0] - y_true[0])]

    @staticmethod
    def __loss_maximize(y_true, y_pred):
        return [- y_pred[0]]

    @staticmethod
    def __loss_maximize_derivative(y_true, y_pred):
        return [- np.ones(y_pred[0].shape)]

    def __opt_gradient_descent(self, layer):
        alpha = self.opt_vars[0]    # Learning rate
        np.subtract(layer.w, alpha * layer.dl_dw, out=layer.w)
        np.subtract(layer.b, alpha * layer.dl_db, out=layer.b)

    def __opt_adaptive_moment_gradient_descent(self, layer):
        alpha = self.opt_vars[0]    # Learning rate
        beta1 = self.opt_vars[1]    # First moment decay rate
        beta2 = self.opt_vars[2]    # Second moment decay rate
        layer.o_aux[0] *= beta1     # Power of the first moment decay to the number of optimization passes
        layer.o_aux[1] *= beta2     # Power of the second moment decay to the number of optimization passes

        np.add(beta1 * layer.w_aux[0], (1-beta1) * layer.dl_dw, out=layer.w_aux[0])
        np.add(beta2 * layer.w_aux[1], (1-beta2) * np.square(layer.dl_dw), out=layer.w_aux[1])
        np.divide(layer.w_aux[0], np.sqrt(layer.w_aux[1]/(1-layer.o_aux[1]))+1E-8, out=layer.dl_dw)
        np.subtract(layer.w, (alpha/(1-layer.o_aux[0])) * layer.dl_dw, out=layer.w)

        np.add(beta1 * layer.b_aux[0], (1-beta1) * layer.dl_db, out=layer.b_aux[0])
        np.add(beta2 * layer.b_aux[1], (1-beta2) * np.square(layer.dl_db), out=layer.b_aux[1])
        np.divide(layer.b_aux[0], np.sqrt(layer.b_aux[1]/(1-layer.o_aux[1]))+1E-8, out=layer.dl_db)
        np.subtract(layer.b, (alpha/(1-layer.o_aux[0])) * layer.dl_db, out=layer.b)

    def save(self, file):
        if isinstance(file, str):
            with open(file, 'wb') as f:
                self.__save(f)
        else:
            self.__save(file)

    def __save(self, file):
        # Join all layers
        layers = self.input_layers + self.layers
        layer_idx = dict(zip(layers, range(len(layers))))

        # Save the neural network's name
        np.save(file, self.name)

        # Save layers' data
        np.save(file, len(layers))
        for layer in layers:
            np.save(file, layer.name)
            np.save(file, layer.layer_type)
            np.save(file, str(layer.in_shape))
            np.save(file, str(layer.out_shape))
            np.save(file, str((layer.i_name, layer.i_mean, layer.i_stdv)))
            np.save(file, layer.frozen)
            np.save(file, layer.critical)
            if (layer.w is not None) and (layer.b is not None):
                np.save(file, True)
                np.save(file, layer.w)
                np.save(file, layer.b)
            else: np.save(file, False)
            if (layer.w_aux is not None) and (layer.b_aux is not None) and (layer.o_aux is not None):
                np.save(file, True)
                np.save(file, layer.w_aux)
                np.save(file, layer.b_aux)
                np.save(file, layer.o_aux)
            else: np.save(file, False)
            if layer.a_aux is not None:
                np.save(file, True)
                np.save(file, layer.a_aux)
            else: np.save(file, False)

        # Save layers' connections
        for layer in layers:
            if (layer.input_layers is not None) and (layer.input_offset is not None):
                np.save(file, True)
                np.save(file, [layer_idx[input_layer] for input_layer in layer.input_layers])
                np.save(file, layer.input_offset)
            else: np.save(file, False)
            if layer.shared_layer in layers:
                np.save(file, True)
                np.save(file, layer_idx[layer.shared_layer])
            else: np.save(file, False)

        # Save network's layers lists
        np.save(file, [layer_idx[layer] for layer in self.input_layers])
        np.save(file, [layer_idx[layer] for layer in self.layers])
        np.save(file, [layer_idx[layer] for layer in self.output_layers])


    @classmethod
    def load(cls, file):
        if isinstance(file, str):
            with open(file, 'rb') as f:
                network = cls(str(np.load(f)), None, None)
                network.__load(f)
        else:
            network = cls(str(np.load(file)), None, None)
            network.__load(file)

        return network

    def __load(self, file):
        layers = []

        # Load layers' data
        for idx in range(np.load(file)):
            name = str(np.load(file))
            layer_type = str(np.load(file))
            in_shape, out_shape = eval(str(np.load(file))), eval(str(np.load(file)))
            initializer = eval(str(np.load(file)))
            layer = NNLayer(name=name, layer_type=layer_type, output_shape=out_shape, initializer=initializer)
            layer.in_shape, layer.frozen, layer. critical = in_shape, bool(np.load(file)), bool(np.load(file))
            if np.load(file):
                layer.w, layer.b = (np.array(np.load(file)), np.array(np.load(file)))
                layer.dl_dw, layer.dl_db = np.zeros(layer.w.shape, dtype=data_type), np.zeros(layer.b.shape, dtype=data_type)

            layer.w_aux, layer.b_aux, layer.o_aux = (np.load(file), np.load(file), np.load(file)) if np.load(file) else (None, None, None)
            layer.a_aux = np.load(file) if np.load(file) else None
            layers.append(layer)

        # Load layers' connections
        for layer in layers:
            layer.input_layers, layer.input_offset = ([layers[int(idx)] for idx in np.load(file)], np.load(file)) if np.load(file) else (None, None)
            layer.shared_layer = layers[int(np.load(file))] if np.load(file) else None

        # Load network's layers lists
        self.input_layers = [layers[int(idx)] for idx in list(np.load(file))]
        self.layers = [layers[int(idx)] for idx in list(np.load(file))]
        self.output_layers = [layers[int(idx)] for idx in list(np.load(file))]

    def print(self, print_weights=False):
        layers = self.input_layers + self.layers
        layer_idx = dict(zip(layers, range(len(layers))))

        print()
        print("|"+("-"*130)+"|")
        print("| {0:128s} |".format("Network: " + self.name))
        print("| {0:128s} |".format("Inputs:  "+str([layer.name for layer in self.input_layers])))
        print("| {0:128s} |".format("Outputs: "+str([layer.name for layer in self.output_layers])))
        print("|"+("-"*130)+"|")
        print("| {0:128s} |".format("Layers: "))
        print("|"+("-"*130)+"|")
        print("| {0:<3s} | {1:<20s} | {2:<9s} | {3:<9s} | {4:<7s} | {5:<65s} |".format(
            "N°",
            "Name",
            "In",
            "Out",
            "Frozen",
            "Dependencies"
        ))
        for layer in self.input_layers:
            print("|"+("-"*5)+"|"+("-"*22)+"|"+("-"*11)+"|"+("-"*11)+"|"+("-"*9)+"|"+("-"*67)+"|")
            print("| {0:>3d} | {1:<20s} | {2:^9s} | {3:^9s} | {4:^7s} | {5:<65s} |".format(
                layer_idx[layer]+1,
                layer.name,
                '-',
                str(('M',)+layer.out_shape),
                "-",
                "None"
            ))
        for layer in self.layers:
            print("|"+("-"*5)+"|"+("-"*22)+"|"+("-"*11)+"|"+("-"*11)+"|"+("-"*9)+"|"+("-"*67)+"|")
            print("| {0:>3d} | {1:<20s} | {2:^9s} | {3:^9s} | {4:^7s} | {5:<65s} |".format(
                layer_idx[layer]+1,
                layer.name,
                str(('M',)+layer.in_shape),
                str(('M',)+layer.out_shape),
                str(layer.frozen),
                str([layers[layer_idx[input_layer]].name for input_layer in layer.input_layers])
            ))
        print("|"+("-"*130)+"|")

        if not print_weights: print(); return
        for layer in self.layers:
            print("| {0:128s} |".format("Layers Weights: {0:<30}".format(layer.name)))
            print("|" + ("-" * 130) + "|")
            print("| {0:128s} |".format("Weights"))
            print(layer.w[:])
            print("| {0:128s} |".format("Bias"))
            print(layer.b[:])
            print("|" + ("-" * 130) + "|")

        print()


if __name__ == '__main__':
    # Create the neural network
    X1 = NNLayer("Input", 1, "X1")
    X2 = NNLayer("Input", 1, "X2")
    X3 = NNLayer("Tanh", 5, "X3")([X1, X2])
    X4 = NNLayer("Tanh", 1, "X4")(X3)
    Net1 = NeuralNetwork("TestNet", [X1, X2], X4)
    Net1.reset(seed=12345)
    Net1.print()

    # Copy and save the neural network for future tests
    Net2 = NeuralNetwork.copy("TestNetCopy", Net1, share_weights=False)
    Net2.reset(seed=12345)
    Net1.save("Test")

    # Train the neural network
    result1 = Net1.compute([np.array([[-1], [1], [3], [5], [7]]), np.array([[0], [0], [0], [0], [0]])])
    Net1.train(x=[np.array([[1],[2],[3],[4],[5]]), np.array([[0],[0],[0],[0],[0]])],
               y=np.array([[0.1],[0.2],[0.3],[0.4],[0.5]]),
               loss_function='MeanSquaredError', optimizer=('Adam',0.1), regularization=(("L2",0.0),), epochs=100)
    result1_tr = Net1.compute([np.array([[-1], [1], [3], [5], [7]]), np.array([[0], [0], [0], [0], [0]])])
    print("\nUntrained result:\n", result1, "\n\nTrained result:\n", result1_tr, "\n")

    if False:
        for layer in Net.layers:
            print("\n\nLayer name: " + layer.name)
            print("x", layer.x)
            print("w", layer.w)
            print("b", layer.b)
            print("z", layer.z)
            print("a", layer.a)

        for layer in reversed(Net.layers):
            print("\n\nLayer name: " + layer.name)
            print("da", layer.dl_da)
            print("dw", layer.dl_dw)
            print("db", layer.dl_db)
            print("dx", layer.dl_dx)

        for layer in reversed(Net.input_layers):
            print("\n\nLayer name: " + layer.name)
            print("da", layer.dl_da)
        while True: pass

    # Copy test
    result2 = Net2.compute([np.array([[-1], [1], [3], [5], [7]]), np.array([[0], [0], [0], [0], [0]])])
    Net2.train(x=[np.array([[1],[2],[3],[4],[5]]), np.array([[0],[0],[0],[0],[0]])],
               y=np.array([[0.1],[0.2],[0.3],[0.4],[0.5]]),
               loss_function='MeanSquaredError', optimizer=('Adam',0.1), regularization=(("L2",0.0),), epochs=100)
    result2_tr = Net2.compute([np.array([[-1], [1], [3], [5], [7]]), np.array([[0], [0], [0], [0], [0]])])
    print("\nCopy:\nUntrained result match:", np.all(result1 == result2),
          "\nTrained result match:", np.all(result1_tr == result2_tr), "\n")

    # Save and load test
    Net3 = NeuralNetwork.load("Test")
    result2 = Net3.compute([np.array([[-1], [1], [3], [5], [7]]), np.array([[0], [0], [0], [0], [0]])])
    Net3.train(x=[np.array([[1],[2],[3],[4],[5]]), np.array([[0],[0],[0],[0],[0]])],
               y=np.array([[0.1],[0.2],[0.3],[0.4],[0.5]]),
               loss_function='MeanSquaredError', optimizer=('Adam',0.1), regularization=(("L2",0.0),), epochs=100)
    result2_tr = Net3.compute([np.array([[-1], [1], [3], [5], [7]]), np.array([[0], [0], [0], [0], [0]])])
    print("\nSave & Load:\nUntrained result match:", np.all(result1 == result2),
          "\nTrained result match:", np.all(result1_tr == result2_tr), "\n")

    # Connect test
    X1 = NNLayer("Input", 1, "X1")
    X2 = NNLayer("Input", 1, "X2")
    X3 = NNLayer("Tanh", 5, "X3")([X1, X2])
    I3 = NNLayer("Input", 5, "X3")
    X4 = NNLayer("Tanh", 1, "X4")(I3)
    Net4 = NeuralNetwork.connect("TestNetConnect", NeuralNetwork("", [X1, X2], X3), NeuralNetwork("", I3, X4))
    Net4.reset(seed=12345)
    result2 = Net4.compute([np.array([[-1], [1], [3], [5], [7]]), np.array([[0], [0], [0], [0], [0]])])
    Net4.train(x=[np.array([[1],[2],[3],[4],[5]]), np.array([[0],[0],[0],[0],[0]])],
               y=np.array([[0.1],[0.2],[0.3],[0.4],[0.5]]),
               loss_function='MeanSquaredError', optimizer=('Adam',0.1), regularization=(("L2",0.0),), epochs=100)
    result2_tr = Net4.compute([np.array([[-1], [1], [3], [5], [7]]), np.array([[0], [0], [0], [0], [0]])])
    print("\nConnect:\nUntrained result match:", np.all(result1 == result2),
          "\nTrained result match:", np.all(result1_tr == result2_tr), "\n")

    # Freeze and shared test
    Net5 = NeuralNetwork.copy("TestNetFreeze", Net1)
    Net5.freeze_layers(["X4"])
    Net6 = NeuralNetwork.copy("TestNetShared", Net5, share_weights=True)
    Net5.reset(seed=12345)

    print("\nShared:\nUntrained weights match:", np.all([(i.w, i.b) for i in Net5.layers] == [(i.w, i.b) for i in Net6.layers]))
    result1 = [Net5.output_layers[0].w.copy(), Net5.output_layers[0].b.copy()]
    Net5.train(x=[np.array([[1], [2], [3], [4], [5]]), np.array([[0], [0], [0], [0], [0]])],
               y=np.array([[0.1], [0.2], [0.3], [0.4], [0.5]]), loss_function='MeanSquaredError',
               optimizer=('Adam', 0.1), regularization=(("L2", 0.0),), epochs=100)
    result2 = [Net5.output_layers[0].w, Net5.output_layers[0].b]
    print("Trained weights match:", np.all([(i.w, i.b) for i in Net5.layers] == [(i.w, i.b) for i in Net6.layers]), "\n")
    print("\nFrozen:\nUntrained and trained weights match:", np.all(result1[0] == result2[0]) and np.all(result1[1] == result2[1]), "\n")
