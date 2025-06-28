import logging
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
import selfies as sf

class SELFormerLayer(tf.keras.layers.Layer):
    def __init__(self, model_name="HUBioDataLab/SELFormer", max_length=128, num_trainable_encoder_layers=-1, **kwargs):
        super(SELFormerLayer, self).__init__(**kwargs)
        self.model_name = model_name
        self.max_length = max_length
        self.num_trainable_encoder_layers = num_trainable_encoder_layers
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.transformer = TFAutoModel.from_pretrained(self.model_name, from_pt=True)
        self._configure_transformer_trainability()

    def _configure_transformer_trainability(self):
        if self.num_trainable_encoder_layers == 0:
            self.transformer.trainable = False
        elif self.num_trainable_encoder_layers == -1:
            self.transformer.trainable = True
        elif self.num_trainable_encoder_layers > 0:
            self.transformer.trainable = True
            for layer in self.transformer.roberta.encoder.layer[:-self.num_trainable_encoder_layers]:
                layer.trainable = False

    def call(self, inputs, training=None):
        
        def py_call(smiles_tensor):
            smiles_list = [s.decode('utf-8') for s in smiles_tensor.numpy()]
            selfies_list = [sf.encoder(s) for s in smiles_list]
            
            tokens = self.tokenizer(selfies_list, return_tensors='tf', padding='max_length', truncation=True, max_length=self.max_length)
            return tokens['input_ids'], tokens['attention_mask']

        input_ids, attention_mask = tf.py_function(
            py_call,
            inp=[inputs],
            Tout=[tf.int32, tf.int32]
        )
        
        input_ids.set_shape([None, self.max_length])
        attention_mask.set_shape([None, self.max_length])
        
        outputs = self.transformer({'input_ids': input_ids, 'attention_mask': attention_mask}, training=training)
        return outputs.pooler_output

    def get_config(self):
        config = super(SELFormerLayer, self).get_config()
        config.update({
            "model_name": self.model_name,
            "max_length": self.max_length,
            "num_trainable_encoder_layers": self.num_trainable_encoder_layers
        })
        return config