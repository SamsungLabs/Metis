# Copyright 2024 Samsung Electronics Co., Ltd. All Rights Reserved
from utils import ModelConfig


class GPTActivationAndParam:
    def __init__(self, model_config: ModelConfig, model_params):
        """
        Initialize the GPTActivationAndParam

        Parameters:
             hidden_size (int): The hidden size of the model.
             sequence_length (int): The sequence length of the input.
             num_layers (int): The total number of layers in the GPT model.
             vocab_size (int): The size of the vocabulary.
             attention_head_size (int): The size of each attention head.
        """
        self.hidden_size = model_config.hidden_size
        self.sequence_length = model_config.sequence_length
        self.num_layers = model_config.num_layers
        self.vocab_size = model_config.vocab_size
        self.attention_head_size = model_config.attention_head_size
        self.input_params = float(model_params[0])
        self.output_params = float(model_params[-1])
        self.transformer_params = float(model_params[1])

    def get_num_layers(self):
        return self.num_layers

    def get_activation_size(self, layer_id, batch_size, tp_deg):
        if layer_id == (self.num_layers - 1):
            return batch_size * self.sequence_length * self.vocab_size / tp_deg
        return batch_size * self.sequence_length * self.hidden_size

    def get_parameter_size(self, tp_deg):
        parameters = [self.input_params/tp_deg]
        parameters += [self.transformer_params/tp_deg for i in range(self.num_layers-2)]
        parameters.append(self.output_params/tp_deg)
        return parameters

    def get_parameter_size_by_stage(self, tp_deg, start_layer_id, end_layer_id):
        num_transformer_layer = end_layer_id - start_layer_id
        parameters = 0
        if start_layer_id == 0:
            parameters += (self.input_params / tp_deg)
            num_transformer_layer -= 1
        if end_layer_id == self.num_layers:
            parameters += (self.output_params / tp_deg)
            num_transformer_layer -= 1

        parameters += (self.transformer_params / tp_deg * num_transformer_layer)
        return parameters
