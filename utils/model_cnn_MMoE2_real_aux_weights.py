from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf

class DatasetWeightLayer(Layer):
    """为dataset_input生成全局一致的权重"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        # 定义可训练权重（旧数据和新数据各一个标量）
        self.weight_old = self.add_weight(name='weight_old', shape=(1,), initializer='ones')
        self.weight_new = self.add_weight(name='weight_new', shape=(1,), initializer='ones')
        super().build(input_shape)
    
    def call(self, inputs):
        dataset_flag = inputs  # shape: (batch, 1)
        # 使用sigmoid约束权重到(0,1)范围
        return tf.nn.sigmoid(self.weight_old) * (1 - dataset_flag) + \
               tf.nn.sigmoid(self.weight_new) * dataset_flag  # shape: (batch, 1)

def get_model_mmoe(max_len_en=30, max_dep=4, n_variants=17,
                  tower_units=[80, 60], aux_feature_dim=4):
    # 输入层
    seq_input = Input(shape=(max_len_en, max_dep), name="input_seq")
    aux_input = Input(shape=(aux_feature_dim,), name="aux_features")
    dataset_input = Input(shape=(1,), name="dataset_flag")  # 0=旧数据, 1=新数据

    # --- 专家网络 ---
    expert_configs = [(3, 100), (5, 70), (7, 40)]
    expert_outputs = []
    for i, (kernel_size, filters) in enumerate(expert_configs):
        x = Conv1D(filters, kernel_size, activation='relu', padding='same',name=f'expert_cnn_{i}')(seq_input)
        x = Dropout(0.3)(x)
        x = AveragePooling1D(2, padding='same')(x)
        x = Flatten()(x)
        expert_outputs.append(Dense(256, activation='relu',name=f'expert_{i}_fc')(x))
    expert_stack = tf.stack(expert_outputs, axis=1)  # (batch, n_experts, 256)

    # --- 数据集权重（全局一致）---
    dataset_weight = DatasetWeightLayer(name="dataset_weights")(dataset_input)  # shape: (batch, 1)

    # --- 门控网络（整合数据集权重）---
    seq_flat = Flatten()(seq_input)
    aux_flat = Flatten()(aux_input)
    dataset_weight_flat = Flatten()(dataset_weight)
    # 合并原始输入特征作为 gate_input
    gate_input = Concatenate()([seq_flat, aux_flat,dataset_weight_flat*10])

    # --- 多任务输出 ---
    task_outputs = []
    for task_idx in range(n_variants):
        # 动态门控（受dataset_weight影响）
        gate = Dense(len(expert_outputs), activation='softmax')(gate_input)
        gate = tf.expand_dims(gate, -1)
        mixed = tf.reduce_sum(expert_stack * gate, axis=1)

        # Task-specific tower
        x = mixed
        for units in tower_units:
            x = Dense(units, activation='relu')(x)
            x = Dropout(0.3)(x)
        task_outputs.append(Dense(1)(x))

    outputs = Concatenate(name="output_all_variants", axis=1)(task_outputs)
    model = Model(inputs=[seq_input, aux_input, dataset_input], outputs=outputs)
    return model
