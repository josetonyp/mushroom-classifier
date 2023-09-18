from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input


class VGG19Base:
    def __init__(self, target_file_size, base_layer_train=0):
        ## Build the base model
        self.base_model = VGG19(
            weights="imagenet",
            include_top=False,
            classifier_activation=None,
            input_shape=target_file_size + (3,),
        )

        for layer in self.base_model.layers:
            layer.trainable = False

        if base_layer_train < 0:
            for layer in self.base_model.layers[base_layer_train:]:
                layer.trainable = True

    def model(self):
        return self.base_model

    def preprocess_input_method(self):
        return preprocess_input
