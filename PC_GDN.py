import os
import random
import numpy as np
import scipy.sparse as sp
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
seed = 2025
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)


def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


class PC_GDN:
    def __init__(self, n_users, n_items_A, n_items_B, graph_matrix_A, graph_matrix_B, cross_domain_graph):
        self.embedding_size = 64
        self.num_heads = 4
        self.num_layers = 2  # GAT layers
        self.transformer_layers = 2  # Transformer layers
        self.temperature = 0.4  # tau
        self.alpha = 0.7   # alpha
        self.dropout_rate = 0.1
        self.learning_rate = 0.001

        self.n_users = n_users
        self.n_items_A = n_items_A
        self.n_items_B = n_items_B
        self.graph_matrix_A = graph_matrix_A
        self.graph_matrix_B = graph_matrix_B
        self.cross_domain_graph = cross_domain_graph

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.uid, self.seq_A, self.seq_B, self.len_A, self.len_B, \
            self.pos_A, self.pos_B, self.target_A, self.target_B = self.get_inputs()

            self.build_network()

    def get_inputs(self):
        uid = tf.placeholder(dtype=tf.int32, shape=[None, ], name='uid')
        seq_A = tf.placeholder(dtype=tf.int32, shape=[None, None], name='seq_A')
        seq_B = tf.placeholder(dtype=tf.int32, shape=[None, None], name='seq_B')
        len_A = tf.placeholder(dtype=tf.int32, shape=[None, ], name='len_A')
        len_B = tf.placeholder(dtype=tf.int32, shape=[None, ], name='len_B')
        pos_A = tf.placeholder(dtype=tf.int32, shape=[None, None], name="pos_A")
        pos_B = tf.placeholder(dtype=tf.int32, shape=[None, None], name="pos_B")
        target_A = tf.placeholder(dtype=tf.int32, shape=[None, ], name='target_A')
        target_B = tf.placeholder(dtype=tf.int32, shape=[None, ], name='target_B')

        return uid, seq_A, seq_B, len_A, len_B, pos_A, pos_B, target_A, target_B

    def _init_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()

        all_weights['user_embedding'] = tf.Variable(
            initializer([self.n_users, self.embedding_size]), name='user_emb')
        all_weights['item_embedding_A'] = tf.Variable(
            initializer([self.n_items_A, self.embedding_size]), name='item_emb_A')
        all_weights['item_embedding_B'] = tf.Variable(
            initializer([self.n_items_B, self.embedding_size]), name='item_emb_B')

        all_weights['pos_embedding_A'] = tf.Variable(
            initializer([self.n_items_A, self.embedding_size]), name='pos_emb_A')
        all_weights['pos_embedding_B'] = tf.Variable(
            initializer([self.n_items_B, self.embedding_size]), name='pos_emb_B')

        for l in range(self.num_layers):
            for h in range(self.num_heads):
                all_weights[f'W_gat_A_l{l}_h{h}'] = tf.Variable(
                    initializer([self.embedding_size, self.embedding_size // self.num_heads]),
                    name=f'W_gat_A_l{l}_h{h}')
                all_weights[f'a_gat_A_l{l}_h{h}'] = tf.Variable(
                    initializer([2 * (self.embedding_size // self.num_heads), 1]),
                    name=f'a_gat_A_l{l}_h{h}')

                all_weights[f'W_gat_B_l{l}_h{h}'] = tf.Variable(
                    initializer([self.embedding_size, self.embedding_size // self.num_heads]),
                    name=f'W_gat_B_l{l}_h{h}')
                all_weights[f'a_gat_B_l{l}_h{h}'] = tf.Variable(
                    initializer([2 * (self.embedding_size // self.num_heads), 1]),
                    name=f'a_gat_B_l{l}_h{h}')

        all_weights['W_prompt'] = tf.Variable(
            initializer([self.embedding_size, self.embedding_size]), name='W_prompt')

        for l in range(self.transformer_layers):
            all_weights[f'W_attn_A_l{l}'] = tf.Variable(
                initializer([self.embedding_size, self.embedding_size]), name=f'W_attn_A_l{l}')
            all_weights[f'W_ffn_A_l{l}'] = tf.Variable(
                initializer([self.embedding_size, self.embedding_size]), name=f'W_ffn_A_l{l}')
            all_weights[f'W_attn_B_l{l}'] = tf.Variable(
                initializer([self.embedding_size, self.embedding_size]), name=f'W_attn_B_l{l}')
            all_weights[f'W_ffn_B_l{l}'] = tf.Variable(
                initializer([self.embedding_size, self.embedding_size]), name=f'W_ffn_B_l{l}')

        all_weights['W_pred_A'] = tf.Variable(
            initializer([2 * self.embedding_size, self.embedding_size]), name='W_pred_A')
        all_weights['b_pred_A'] = tf.Variable(tf.zeros([self.embedding_size]), name='b_pred_A')
        all_weights['W_final_A'] = tf.Variable(
            initializer([self.embedding_size, self.n_items_A]), name='W_final_A')

        all_weights['W_pred_B'] = tf.Variable(
            initializer([2 * self.embedding_size, self.embedding_size]), name='W_pred_B')
        all_weights['b_pred_B'] = tf.Variable(tf.zeros([self.embedding_size]), name='b_pred_B')
        all_weights['W_final_B'] = tf.Variable(
            initializer([self.embedding_size, self.n_items_B]), name='W_final_B')

        return all_weights

    def build_network(self):

        self.all_weights = self._init_weights()

        # 1. single_domain purification network
        with tf.variable_scope('single_domain_purification'):
            self.EU_A, self.ES_A = self.single_domain_gat(
                self.graph_matrix_A, 'A', self.all_weights['user_embedding'],
                self.all_weights['item_embedding_A'])
            self.EU_B, self.ES_B = self.single_domain_gat(
                self.graph_matrix_B, 'B', self.all_weights['user_embedding'],
                self.all_weights['item_embedding_B'])

        # 2. prompt generation network
        with tf.variable_scope('prompt_generation'):
            initial_embeddings = tf.concat([
                self.all_weights['user_embedding'],
                self.all_weights['item_embedding_A'],
                self.all_weights['item_embedding_B']
            ], axis=0)

            prompt_embeddings = self.single_layer_gcn(
                self.cross_domain_graph, initial_embeddings, self.all_weights['W_prompt'])

            self.PU_A, self.PS_A = tf.split(
                prompt_embeddings[:self.n_users + self.n_items_A],
                [self.n_users, self.n_items_A], axis=0)
            self.PU_B, self.PS_B = tf.split(
                prompt_embeddings[:self.n_users + self.n_items_B],
                [self.n_users, self.n_items_B], axis=0)

        # 3. bidirectional distillation network
        with tf.variable_scope('bidirectional_distillation'):
            self.TS_A = self.transformer_encoder(self.ES_A, 'A', is_teacher=True)
            self.TS_B = self.transformer_encoder(self.ES_B, 'B', is_teacher=True)

            self.SS_A = self.transformer_encoder(self.ES_A, 'A', is_teacher=False)
            self.SS_B = self.transformer_encoder(self.ES_B, 'B', is_teacher=False)

            self.TS_A_enhanced = self.TS_A + 0.1 * self.PS_A  # lambda_T
            self.SS_A_enhanced = self.SS_A + 0.1 * self.PS_A  # lambda_S
            self.TS_B_enhanced = self.TS_B + 0.1 * self.PS_B
            self.SS_B_enhanced = self.SS_B + 0.1 * self.PS_B

            self.L_KD_A2B = self.compute_kl_divergence(
                self.TS_A_enhanced, self.SS_B_enhanced, self.temperature)
            self.L_KD_B2A = self.compute_kl_divergence(
                self.TS_B_enhanced, self.SS_A_enhanced, self.temperature)
            self.L_KD = self.L_KD_A2B + self.L_KD_B2A

        with tf.variable_scope('final_prediction'):
            self.EU_A_enhanced = self.EU_A + 0.1 * self.PU_A  # gamma_U
            self.EU_B_enhanced = self.EU_B + 0.1 * self.PU_B

            self.SS_A_enhanced_final = self.SS_A + 0.1 * self.PS_A  # gamma_S
            self.SS_B_enhanced_final = self.SS_B + 0.1 * self.PS_B

            self.h_A = self.integrate_representations(
                self.EU_A_enhanced, self.SS_A_enhanced_final, 'A')
            self.h_B = self.integrate_representations(
                self.EU_B_enhanced, self.SS_B_enhanced_final, 'B')

            self.pred_A = self.make_prediction(self.h_A, 'A')
            self.pred_B = self.make_prediction(self.h_B, 'B')

        with tf.variable_scope('loss'):
            self.L_CE_A = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.target_A, logits=self.pred_A))
            self.L_CE_B = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.target_B, logits=self.pred_B))

            self.total_loss = self.L_CE_A + self.L_CE_B + self.alpha * self.L_KD

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.total_loss)

    def single_domain_gat(self, graph_matrix, domain, user_emb, item_emb):
        initial_embeddings = tf.concat([item_emb, user_emb], axis=0)
        all_embeddings = [initial_embeddings]

        for l in range(self.num_layers):
            layer_embeddings = []
            for h in range(self.num_heads):
                if domain == 'A':
                    W = self.all_weights[f'W_gat_A_l{l}_h{h}']
                    a = self.all_weights[f'a_gat_A_l{l}_h{h}']
                else:
                    W = self.all_weights[f'W_gat_B_l{l}_h{h}']
                    a = self.all_weights[f'a_gat_B_l{l}_h{h}']

                h_transformed = tf.matmul(all_embeddings[-1], W)

                attention_scores = self.compute_attention_scores(
                    h_transformed, graph_matrix, a)

                attention_weights = tf.nn.softmax(attention_scores, axis=-1)
                head_embedding = tf.matmul(attention_weights, h_transformed)
                layer_embeddings.append(head_embedding)

            if self.num_heads > 1:
                concatenated = tf.concat(layer_embeddings, axis=-1)
            else:
                concatenated = layer_embeddings[0]

            activated = tf.nn.elu(concatenated)
            all_embeddings.append(activated)

        final_embeddings = tf.reduce_sum(all_embeddings, axis=0)

        item_final, user_final = tf.split(
            final_embeddings, [self.n_items_A if domain == 'A' else self.n_items_B,
                               self.n_users], axis=0)

        return user_final, item_final

    def compute_attention_scores(self, h, graph_matrix, a):
        graph_tensor = _convert_sp_mat_to_sp_tensor(graph_matrix)

        source = tf.gather(h, graph_tensor.indices[:, 0])
        target = tf.gather(h, graph_tensor.indices[:, 1])
        concat = tf.concat([source, target], axis=-1)

        e = tf.nn.leaky_relu(tf.matmul(concat, a))
        e = tf.squeeze(e, axis=-1)

        attention_scores = tf.SparseTensor(
            indices=graph_tensor.indices,
            values=e,
            dense_shape=graph_tensor.dense_shape)

        return attention_scores

    def single_layer_gcn(self, graph_matrix, embeddings, W):
        identity = sp.identity(graph_matrix.shape[0])
        graph_with_self = graph_matrix + identity

        rowsum = np.array(graph_with_self.sum(1)).flatten()
        D_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
        normalized_graph = D_inv_sqrt.dot(graph_with_self).dot(D_inv_sqrt)

        graph_tensor = _convert_sp_mat_to_sp_tensor(normalized_graph)

        propagated = tf.sparse_tensor_dense_matmul(graph_tensor, embeddings)
        transformed = tf.matmul(propagated, W)

        output = tf.nn.relu(transformed)

        return output

    def transformer_encoder(self, embeddings, domain, is_teacher=False):
        current = embeddings

        for l in range(self.transformer_layers):
            if domain == 'A':
                W_attn = self.all_weights[f'W_attn_A_l{l}']
                W_ffn = self.all_weights[f'W_ffn_A_l{l}']
            else:
                W_attn = self.all_weights[f'W_attn_B_l{l}']
                W_ffn = self.all_weights[f'W_ffn_B_l{l}']

            attn_output = self.multihead_attention(current, current, current, W_attn)
            current = tf.contrib.layers.layer_norm(current + attn_output)
            ffn_output = tf.layers.dense(current, self.embedding_size, activation=tf.nn.relu)
            ffn_output = tf.layers.dense(ffn_output, self.embedding_size)
            current = tf.contrib.layers.layer_norm(current + ffn_output)

        return current

    def multihead_attention(self, query, key, value, W):
        Q = tf.matmul(query, W)
        K = tf.matmul(key, W)
        V = tf.matmul(value, W)

        d_k = tf.cast(tf.shape(K)[-1], tf.float32)
        scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d_k)
        attention_weights = tf.nn.softmax(scores, axis=-1)

        output = tf.matmul(attention_weights, V)

        return output

    def compute_kl_divergence(self, teacher_logits, student_logits, temperature):
        teacher_probs = tf.nn.softmax(teacher_logits / temperature, axis=-1)
        student_probs = tf.nn.softmax(student_logits / temperature, axis=-1)

        kl_div = tf.reduce_sum(
            teacher_probs * tf.log(teacher_probs / student_probs), axis=-1)

        return tf.reduce_mean(kl_div)

    def integrate_representations(self, user_emb, item_emb, domain):
        concat = tf.concat([user_emb, item_emb], axis=-1)

        if domain == 'A':
            W = self.all_weights['W_pred_A']
            b = self.all_weights['b_pred_A']
        else:
            W = self.all_weights['W_pred_B']
            b = self.all_weights['b_pred_B']

        transformed = tf.matmul(concat, W) + b
        output = tf.nn.gelu(transformed)

        return output

    def make_prediction(self, hidden_state, domain):
        if domain == 'A':
            W = self.all_weights['W_final_A']
            logits = tf.matmul(hidden_state, W)
        else:
            W = self.all_weights['W_final_B']
            logits = tf.matmul(hidden_state, W)

        return logits

