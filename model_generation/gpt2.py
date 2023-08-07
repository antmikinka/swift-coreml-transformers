"""
Recreate the Core ML model from scratch using
coremltools' neural_network.NeuralNetworkBuilder
"""
import coremltools
import coremltools.models.datatypes as datatypes
from coremltools.models import neural_network as neural_network
from coremltools.models.utils import save_spec
import numpy as np

# get weights
from transformers import GPT2LMHeadModel
model_name = "gpt2"
lm_head_model = GPT2LMHeadModel.from_pretrained(model_name)
model = lm_head_model.transformer

wte = model.wte.weight.data.numpy().transpose() # shape (768, 50257) /!\ i hate this
wpe = model.wpe.weight.data.numpy().transpose() # shape (768, 1024)
#wpe.weight	[1024,768]	F32
#wte.weight	[50257,768]	F32

sequence_length = 64
steps = 12

# build model
input_features = [
	('input_ids', datatypes.Array(sequence_length)),
	('position_ids', datatypes.Array(sequence_length)),
]
output_features = [('output_logits', None)]

builder = neural_network.NeuralNetworkBuilder(
	input_features,
	output_features,
	mode=None,
	disable_rank5_shape_mapping=True,
)

'''
The builder.add_expand_dims function is used to add additional dimensions to an input tensor. 
This is useful for models that expect input tensors to have a specific rank.

In the code you provided, the input_ids and position_ids tensors are both rank-1 tensors. 
The builder.add_expand_dims function is used to expand them to rank-5 tensors. 
This is necessary because the transformer model expects the input tensors to have rank-5.

The axes argument to the builder.add_expand_dims function specifies the dimensions to be expanded. 
In the code you provided, the axes argument is set to [1, 2, 3, 4]. This means that the first four 
	dimensions of the input tensors will be expanded.

The result of the builder.add_expand_dims function is two new tensors, input_ids_expanded_to_rank5 and 
	position_ids_expanded_to_rank5. These tensors have rank-5, which is the expected rank for the input 
 	tensors of the transformer model.

I hope this explanation is helpful! Let me know if you have any other questions.
'''

builder.add_expand_dims(
	name='input_ids_expanded_to_rank5',
	input_name='input_ids',
	output_name='input_ids_expanded_to_rank5',
	axes=(1, 2, 3, 4)
)
builder.add_expand_dims(
	name='position_ids_expanded_to_rank5',
	input_name='position_ids',
	output_name='position_ids_expanded_to_rank5',
	axes=(1, 2, 3, 4)
)
builder.add_embedding(
	name='token_embeddings',
	input_name='input_ids_expanded_to_rank5',
	output_name='token_embeddings',
	W=wte,
	b=None,
	input_dim=50257,
	output_channels=768,
	has_bias=False,
)
builder.add_embedding(
	name='positional_embeddings',
	input_name='position_ids_expanded_to_rank5',
	output_name='positional_embeddings',
	W=wpe,
	b=None,
	input_dim=1024,
	output_channels=768,
	has_bias=False,
)

# Input:, Output: (seq, 1, 768, 1, 1)
builder.add_add_broadcastable(
	name='embeddings_addition',
	input_names=['token_embeddings', 'positional_embeddings'],
	output_name=f'{0}_previous_block'
)


#-----------STARTING LN_1 BIAS LN_1 WEIGHT------------------------
for i in range(steps):
	print(i)
	ln_weight = model.h[i].ln_1.weight.data.numpy().reshape((1, 1, 768, 1, 1))
	#h.0.ln_1.weight	[768]	F32
	ln_bias = model.h[i].ln_1.bias.data.numpy().reshape((1, 1, 768, 1, 1))
	#h.0.ln_1.bias	[768]	F32
	ln_epsilon = model.h[i].ln_1.eps
	#CHANGING THE LN_NUMBER(LN_1) TO LN_2
	
	builder.add_mvn(
		name=f"{i}_block_ln_1",
		input_name=f"{i}_previous_block",
		# output_name=f"{i}_block_ln_1_output",
		output_name=f"{i}_block_ln_1",
		across_channels=True,
		normalize_variance=True,
		epsilon=ln_epsilon
	)

	builder.add_scale(
		name=f"{i}_block_ln_1_scaled",
		input_name=f"{i}_block_ln_1",
		output_name=f"{i}_block_ln_1_scaled",
		W=ln_weight,
		b=ln_bias,
		has_bias=True,
		shape_scale=[768],
		shape_bias=[768]
	)

	builder.add_transpose(
		name=f"{i}_block_ln_1_reshape",
		input_name=f"{i}_block_ln_1_scaled",
		output_name=f"{i}_block_ln_1_scaled_transposed",
		axes=(1, 0, 2, 3, 4)
	)
#-----------ENDING LN_1 BIAS----------------LN_1 WEIGHT------------------------
#------------STARTING ATTN C BIAS-----------ATTN C WEIGHT-------------------------
'''
The code you provided is a part of the transformer model implementation in the Swift-CoreML-Transformers library. 
It is used to compute the attention weights for the self-attention layer in the encoder block.
The first few lines of code define the weights and biases for the inner product layer that is used to compute the 
	attention weights. The inner product layer has 768 input channels and 2304 output channels. 
 
 The weights are initialized with the values from the model.h[i].attn.c_attn.weight tensor, 
 	and the biases are initialized with the values from the model.h[i].attn.c_attn.bias tensor.

The next few lines of code split the output of the inner product layer into three tensors: q, k, and v. 
	These tensors represent the query, key, and value vectors for the self-attention layer.

The next few lines of code reshape the q, k, and v tensors to have the same shape as the input sequence. 
	This is necessary because the batched_mat_mul layer that is used to compute the attention weights 
 	requires the input tensors to have the same shape.

The next line of code computes the attention weights using the batched_mat_mul layer. 
The q and k tensors are multiplied together, and the output is scaled by 1/8. 
The attention weights are then added with the bias tensor.

The final line of code adds a scale layer to the attention weights. 
This layer normalizes the attention weights so that they sum to 1.

The overall purpose of this code is to compute the attention weights for the self-attention layer in the encoder block. 
The attention weights are used to determine how much attention each input token should pay to each other token in the 
sequence. This allows the model to learn long-range dependencies between tokens in the sequence.
'''
#---------------------------------------------------------------------------------
'''
Sure, I can explain each builder module in the code you provided:

* `conv_1D_bias = model.h[i].attn.c_attn.bias.data.numpy().reshape((1, 1, 2304, 1, 1))`

This line of code retrieves the bias weights from the attention layer of the model at index `i`. The bias weights are a 1D tensor of size 2304. The `reshape` function is used to reshape the bias weights to a 5D tensor with the following dimensions:

    * Batch size: 1
    * Channel size: 1
    * Height: 2304
    * Width: 1
    * Depth: 1

* `conv_1D_weights = model.h[i].attn.c_attn.weight.data.numpy().transpose().reshape((1, 768, 2304, 1, 1))`

This line of code retrieves the weight matrix from the attention layer of the model at index `i`. The weight matrix is a 2D tensor of size 768 x 2304. The `transpose` function is used to transpose the weight matrix, so that the channels are the first dimension. The `reshape` function is then used to reshape the weight matrix to a 5D tensor with the same dimensions as the bias weights.

* `builder.add_inner_product(
		name=f"{i}_block_attn_conv",
		input_name=f"{i}_block_ln_1_scaled_transposed",
		output_name=f"{i}_block_attn_conv",
		input_channels=768,
		output_channels=2304,
		W=conv_1D_weights,
		b=conv_1D_bias,
		has_bias=True
	)`

This line of code adds an inner product layer to the neural network builder. The inner product layer takes as input the output of the layer `f"{i}_block_ln_1_scaled_transposed` and produces an output of size 2304. The `W` and `b` parameters are the weight matrix and bias vector from the attention layer. The `has_bias` parameter is set to `True` to indicate that the inner product layer has a bias vector.

* `builder.add_split(
		name=f"{i}_block_attn_qkv_split",
		input_name=f"{i}_block_attn_conv",
		output_names=[f"{i}_block_attn_q", f"{i}_block_attn_k", f"{i}_block_attn_v"]
	)`

This line of code adds a split layer to the neural network builder. The split layer takes as input the output of the inner product layer and produces three outputs: `f"{i}_block_attn_q`, `f"{i}_block_attn_k`, and `f"{i}_block_attn_v`. These three outputs are the query, key, and value vectors for the attention layer.

I hope this explanation is helpful! Let me know if you have any other questions.
'''
	conv_1D_bias = model.h[i].attn.c_attn.bias.data.numpy().reshape((1, 1, 2304, 1, 1))
	#h.0.attn.c_attn.bias	[2304]	F32
	
	conv_1D_weights = model.h[i].attn.c_attn.weight.data.numpy().transpose().reshape((1, 768, 2304, 1, 1))
	#h.0.attn.c_attn.weight	[768,2304]	F32
	
	builder.add_inner_product(
		name=f"{i}_block_attn_conv",
		input_name=f"{i}_block_ln_1_scaled_transposed",
		output_name=f"{i}_block_attn_conv",
		input_channels=768,
		output_channels=2304,
		W=conv_1D_weights,
		b=conv_1D_bias,
		has_bias=True
	)

	#TAKES BLOCK ATTN - SPLITS IT INTO THREE BLOCK ATTN Q/K/V
	builder.add_split(
		name=f"{i}_block_attn_qkv_split",
		input_name=f"{i}_block_attn_conv",
		output_names=[f"{i}_block_attn_q", f"{i}_block_attn_k", f"{i}_block_attn_v"]
	)
#-------------------------------------------------------------------------------------------------

	#ATTN Query
	builder.add_rank_preserving_reshape(
		name=f"{i}_block_attn_q_reshape",
		input_name=f"{i}_block_attn_q",
		output_name=f"{i}_block_attn_q_reshape",
		output_shape=(1, 1, sequence_length, 12, 64)
	)

	builder.add_transpose(
		name=f"{i}_block_attn_q_reshape_permuted",
		input_name=f"{i}_block_attn_q_reshape",
		output_name=f"{i}_block_attn_q_reshape_permuted",
		axes=(0, 1, 3, 2, 4)
	)


	builder.add_rank_preserving_reshape(
		name=f"{i}_block_attn_k_reshape",
		input_name=f"{i}_block_attn_k",
		output_name=f"{i}_block_attn_k_reshape",
		output_shape=(1, 1, sequence_length, 12, 64)
	)

	builder.add_transpose(
		name=f"{i}_block_attn_k_reshape_permuted",
		input_name=f"{i}_block_attn_k_reshape",
		output_name=f"{i}_block_attn_k_reshape_permuted",
		axes=(0, 1, 3, 4, 2)
	)



	builder.add_rank_preserving_reshape(
		name=f"{i}_block_attn_v_reshape",
		input_name=f"{i}_block_attn_v",
		output_name=f"{i}_block_attn_v_reshape",
		output_shape=(1, 1, sequence_length, 12, 64)
	)

	builder.add_transpose(
		name=f"{i}_block_attn_v_reshape_permuted",
		input_name=f"{i}_block_attn_v_reshape",
		output_name=f"{i}_block_attn_v_reshape_permuted",
		axes=(0, 1, 3, 2, 4)
	)



	builder.add_batched_mat_mul(
		name=f"{i}_block_attn_qv_matmul",
		input_names=[f"{i}_block_attn_q_reshape_permuted", f"{i}_block_attn_k_reshape_permuted"],
		output_name=f"{i}_block_attn_qv_matmul"
	)

	builder.add_scale(
		name=f"{i}_block_attn_qv_matmul_scaled",
		input_name=f"{i}_block_attn_qv_matmul",
		output_name=f"{i}_block_attn_qv_matmul_scaled",
		W=np.array(1/8),
		b=0,
		has_bias=False
	)



	bias_0 = model.h[i].attn.bias
	nd = ns = sequence_length
	b = (model.h[i].attn.bias[:, :, ns-nd:ns, :ns]).unsqueeze(0)

	builder.add_scale(
		name=f"{i}_block_attn_bias",
		input_name=f"{i}_block_attn_qv_matmul_scaled",
		output_name=f"{i}_block_attn_bias",
		W=b,
		b=None,
		has_bias=False,
		shape_scale=[1, sequence_length, sequence_length]
	)

	bias_constant_0 = - 1e4 * (1 - b)

	builder.add_bias(
		name=f"{i}_block_attn_afterbias",
		input_name=f"{i}_block_attn_bias",
		output_name=f"{i}_block_attn_afterbias",
		# output_name=f"output_logits",
		b=bias_constant_0,
		shape_bias=[1, sequence_length, sequence_length],
	)

	builder.add_squeeze(
		name=f"{i}_squeezit",
		input_name=f"{i}_block_attn_afterbias",
		output_name=f"{i}_squeezit",
		axes=[0, 1]
	)

	builder.add_softmax(
		name=f"{i}_block_attn_softmax",
		input_name=f"{i}_squeezit",
		output_name=f"{i}_block_attn_softmax",
	)

	builder.add_expand_dims(
		name=f"{i}_expandit",
		input_name=f"{i}_block_attn_softmax",
		output_name=f"{i}_expandit",
		axes=[0, 1]
	)

	builder.add_batched_mat_mul(
		name=f"{i}_block_full_attention",
		input_names=[f"{i}_expandit", f"{i}_block_attn_v_reshape_permuted"],
		output_name=f"{i}_block_full_attention"
	)



	builder.add_transpose(
		name=f"{i}_block_full_attention_merged_t",
		input_name=f"{i}_block_full_attention",
		output_name=f"{i}_block_full_attention_merged_t",
		axes=[0, 1, 3, 2, 4]
	)

	builder.add_rank_preserving_reshape(
		name=f"{i}_block_full_attention_merged",
		input_name=f"{i}_block_full_attention_merged_t",
		output_name=f"{i}_block_full_attention_merged",
		output_shape=[1, 1, 1, sequence_length, 768]
	)

	builder.add_transpose(
		name=f"{i}_block_attn_conv_proj_t",
		input_name=f"{i}_block_full_attention_merged",
		output_name=f"{i}_block_attn_conv_proj_t",
		axes=[0, 3, 4, 1, 2]
	)

#---------------STARTING ATTN C PROJ BIAS-----ATTN C PROJ WEIGHT--------------
	conv_1D_proj_bias = model.h[i].attn.c_proj.bias.data.numpy().reshape((1, 1, 768, 1, 1))
	conv_1D_proj_weights = model.h[i].attn.c_proj.weight.data.numpy().transpose().reshape((1, 768, 768, 1, 1))

	# Input:, Output: (1, 3, 768, 1, 1)
	builder.add_inner_product(
		name=f"{i}_block_attn_conv_proj",
		input_name=f"{i}_block_attn_conv_proj_t",
		output_name=f"{i}_block_attn_conv_proj",
		input_channels=768,
		output_channels=768,
		W=conv_1D_proj_weights,
		b=conv_1D_proj_bias,
		has_bias=True
	)
	# Input: (seq, 1, 768, 1, 1), Output: (1, seq, 768, 1, 1)
	builder.add_transpose(
		name=f"{i}_previous_block_t",
		input_name=f'{i}_previous_block',
		output_name=f"{i}_previous_block_t",
		axes=[1, 0, 2, 3, 4]
	)

	# Input: [(1, seq, 768, 1, 1), (1, seq, 768, 1, 1)], Output: (1, seq, 768, 1, 1)
	builder.add_add_broadcastable(
		name=f"{i}_block_xa_sum",
		input_names=[f"{i}_previous_block_t", f"{i}_block_attn_conv_proj"],
		output_name=f"{i}_block_xa_sum",
		# output_name=f"output_logits"
	)
#------------ENDING ATTN C BIAS----------ATTN C WEIGHT-------------------------

#-------------STARTING LN_2 BIAS--------LN_2 WEIGHT---------------
	ln_2_weight = model.h[i].ln_2.weight.data.numpy().reshape((1, 1, 768, 1, 1))
	ln_2_bias = model.h[i].ln_2.bias.data.numpy().reshape((1, 1, 768, 1, 1))
	ln_2_epsilon = model.h[i].ln_2.eps

	# Input: (1, seq, 768, 1, 1), Output:
	builder.add_mvn(
		name=f"{i}_block_ln_2",
		input_name=f"{i}_block_xa_sum",
		output_name=f"{i}_block_ln_2",
		across_channels=True,
		normalize_variance=True,
		epsilon=ln_2_epsilon
	)

	builder.add_scale(
		name=f"{i}_block_ln_2_scaled",
		input_name=f"{i}_block_ln_2",
		# output_name=f"output_logits",
		output_name=f"{i}_block_ln_2_scaled",
		W=ln_2_weight,
		b=ln_2_bias,
		has_bias=True,
		shape_scale=[768],
		shape_bias=[768]
	)

#-------------ENDING LN_2 BIAS--------LN_2 WEIGHT---------------
#---------------STARTING MLP C FC BIAS-----MLP C FC WEIGHT----------------------
	mlp_conv_1D_fc_bias = model.h[i].mlp.c_fc.bias.data.numpy().reshape((1, 1, 3072, 1, 1))
	mlp_conv_1D_fc_weights = model.h[i].mlp.c_fc.weight.data.numpy().transpose().reshape((1, 768, 3072, 1, 1))

	# Input:, Output: (1, 3, 3072, 1, 1)
	builder.add_inner_product(
		name=f"{i}_block_mlp_conv_fc",
		input_name=f"{i}_block_ln_2_scaled",
		output_name=f"{i}_block_mlp_conv_fc",
		# output_name=f"output_logits",
		input_channels=768,
		output_channels=3072,
		W=mlp_conv_1D_fc_weights,
		b=mlp_conv_1D_fc_bias,
		has_bias=True
	)

	builder.add_gelu(
		name=f"{i}_block_mlp_gelu",
		input_name=f"{i}_block_mlp_conv_fc",
		output_name=f"{i}_block_mlp_gelu",
		# output_name=f"output_logits",
		mode='TANH_APPROXIMATION'
	)
#---------------ENDING MLP C FC BIAS-----MLP C FC WEIGHT----------------------
#---------------STARTING MLP C PROJ BIAS-----MLP C PROJ WEIGHT----------------------
	mlp_conv_1D_proj_bias = model.h[i].mlp.c_proj.bias.data.numpy().reshape((1, 1, 768, 1, 1))
	mlp_conv_1D_proj_weights = model.h[i].mlp.c_proj.weight.data.numpy().transpose().reshape((1, 3072, 768, 1, 1))

	# Input:, Output: (1, 3, 3072, 1, 1)
	builder.add_inner_product(
		name=f"{i}_block_mlp_conv_proj",
		input_name=f"{i}_block_mlp_gelu",
		output_name=f"{i}_block_mlp_conv_proj",
		# output_name=f"output_logits",
		input_channels=3072,
		output_channels=768,
		W=mlp_conv_1D_proj_weights,
		b=mlp_conv_1D_proj_bias,
		has_bias=True
	)

	builder.add_add_broadcastable(
		name=f"{i}_block_xm_sum",
		input_names=[f"{i}_block_xa_sum", f"{i}_block_mlp_conv_proj"],
		# output_name=f"output_logits"
		output_name=f"{i + 1}_previous_block_final"
	)

	builder.add_transpose(
		name=f"{i}_block_xm_sum_t",
		input_name=f"{i + 1}_previous_block_final",
		output_name=f"{i + 1}_previous_block",
		axes=[1, 0, 2, 3, 4]
	)
#---------------ENDING MLP C PROJ BIAS-----MLP C PROJ WEIGHT----------------------

#POSSIBILY SETTING UF FOR A METHOD OF REPEATING LN_1 THEN LN_2, THATS WHY ITS NAMED LN_F
ln_f_weight = model.ln_f.weight.data.numpy().reshape((1, 1, 768, 1, 1))
ln_f_bias = model.ln_f.bias.data.numpy().reshape((1, 1, 768, 1, 1))
ln_f_epsilon = model.ln_f.eps

# Input: (1, seq, 768, 1, 1), Output:
builder.add_mvn(
	name=f"ln_f",
	input_name=f"{steps}_previous_block_final",
	output_name=f"ln_f",
	# output_name=f"output_logits",
	across_channels=True,
	normalize_variance=True,
	epsilon=ln_f_epsilon
)

builder.add_scale(
	name=f"ln_f_scaled",
	input_name=f"ln_f",
	output_name=f"ln_f_scaled",
	# output_name=f"output_logits",
	W=ln_f_weight,
	b=ln_f_bias,
	has_bias=True,
	shape_scale=[768],
	shape_bias=[768]
)

lm_head_weights = lm_head_model.lm_head.weight.data.numpy().reshape((1, 50257, 768, 1, 1))

builder.add_inner_product(
	name="lm_head",
	input_name="ln_f_scaled",
	output_name="output_logits",
	input_channels=768,
	output_channels=50257,
	W=lm_head_weights,
	b=None,
	has_bias=False
)

# compile spec to model
mlmodel = coremltools.models.MLModel(builder.spec)

save_spec(builder.spec, f'../Resources/{model_name}-{sequence_length}-{steps}-2.mlmodel')
# model = coremltools.models.MLModel('gpt2.mlmodel')

# input_ids = np.zeros(sequence_length)
# position_ids = np.arange(sequence_length).astype(np.float)

# input_data = {
# 	'input_ids': input_ids,
# 	'position_ids': position_ids,
# }

# predictions = mlmodel.predict(input_data)["output_logits"]
# equal = np.amax(predictions - mlp_conv_proj.detach().numpy())

# print(predictions)


# save_spec(builder.spec, 'gpt2.mlmodel')
