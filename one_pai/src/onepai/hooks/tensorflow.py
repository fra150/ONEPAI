"""
Gancio (Callback) per modelli TensorFlow/Keras.
"""
import tensorflow as tf
from onepai.core.observer import Observer

class OnePaiTensorflowCallback(tf.keras.callbacks.Callback):
    """
    Un callback Keras per catturare il potenziale da un modello TensorFlow.
    """
    def __init__(self, observer: Observer, layer_to_watch: str):
        super().__init__()
        self.observer = observer
        self.layer_to_watch = layer_to_watch
        self.intermediate_model = None

    def on_train_begin(self, logs=None):
        # Creiamo un sotto-modello che restituisce l'output del layer che ci interessa
        layer_output = self.model.get_layer(self.layer_to_watch).output
        self.intermediate_model = tf.keras.Model(inputs=self.model.inputs, outputs=layer_output)
        print(f"INFO: Callback ONEPAI per TF/Keras inizializzato per il layer: {self.layer_to_watch}")

    def on_predict_batch_end(self, batch, logs=None):
        # Durante l'inferenza, catturiamo l'output intermedio
        if 'inputs' in logs:
            intermediate_output = self.intermediate_model.predict(logs['inputs'])
            self.observer.capture_potentiality(intermediate_output)