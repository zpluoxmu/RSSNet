import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import tf_util
from transform_nets import input_transform_net
from tensorflow.contrib import rnn

def placeholder_inputs(batch_size, num_point):
  pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
  labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
 
  slice_number = 20
  image_resolution = 10    
  x_z = tf.placeholder(tf.float32,[None, slice_number, image_resolution*image_resolution + 7])
  x_y = tf.placeholder(tf.float32,[None, slice_number, image_resolution*image_resolution + 7])
  x_x = tf.placeholder(tf.float32,[None, slice_number, image_resolution*image_resolution + 7])    
    
  return pointclouds_pl, labels_pl, x_z, x_y, x_x

def get_model(point_cloud, is_training, x_z, x_y, x_x, bn_decay=None):
  """ Classification, input is BxNx3, output Bx40 """
  batch_size = point_cloud.get_shape()[0].value
  num_point = point_cloud.get_shape()[1].value
  end_points = {}
  k = 20

  slice_number = 20
  image_resolution = 10
  n_steps = slice_number
  n_input = image_resolution * image_resolution + 7
  n_hidden = 64   
  n_classes = 3
#---------------------GRU--------------------------------------#
  lstm_cell1_z = rnn.GRUCell(n_hidden*2 )
  lstm_cell2_z = rnn.GRUCell(n_hidden*3 )
  lstm_cell3_z = rnn.GRUCell(n_hidden*3 )
  lstm_cell4_z = rnn.GRUCell(n_hidden*3 )
  mlstm_cell1_z = rnn.MultiRNNCell([lstm_cell1_z] , state_is_tuple=True)
  mlstm_cell2_z = rnn.MultiRNNCell([mlstm_cell1_z,lstm_cell2_z] , state_is_tuple=True)
  mlstm_cell3_z = rnn.MultiRNNCell([mlstm_cell2_z,lstm_cell3_z] , state_is_tuple=True)
  mlstm_cell4_z = rnn.MultiRNNCell([mlstm_cell3_z,lstm_cell4_z] , state_is_tuple=True)
  
  x_img_z = tf.reshape(x_z,[-1,n_steps, n_input])
  zx_img_z=tf.unstack(x_img_z,n_steps,1)
  outputs_z, _, _ = rnn.static_bidirectional_rnn(mlstm_cell4_z,mlstm_cell4_z, zx_img_z, dtype=tf.float32)    
  """---Here the shape of outputs_z is (slice_number = 20, batch_size = 32, (n_hidden*3)*2)-----------"""
  #print('this is the shape of net4', (outputs_z).shape)
  """--------------------------------------------vote-------------------------------------------------"""
  with tf.variable_scope('vote') as sc:
      vote_w=tf.get_variable('vote_w',[slice_number, 1, 1],tf.float32,initializer=tf.random_uniform_initializer)*0.01

  vote_outputs_z = tf.multiply(outputs_z, vote_w)
  ptr_temp = tf.squeeze(vote_outputs_z[0,:,:])
  for i in range(1,slice_number):
      ptr_temp = tf.add( tf.squeeze(vote_outputs_z[i,:,:]), ptr_temp)
  ptr_temp = tf.reshape(ptr_temp, [-1, 1, 1, (n_hidden*3)*2])
  rnn_feat_expand = tf.tile(ptr_temp, [1, num_point, 1, 1])
  """---Here the shape of ptr_temp is (batch_size = 32, (n_hidden*3)*2)-----------"""      
  #print('this is the value of vote_w', vote_w)
#  x_img_y = tf.reshape(x_y,[-1,n_steps, n_input])
#  yx_img_y=tf.unstack(x_img_y,n_steps,1)
#  outputs_y, output_states_y, _ = rnn.static_bidirectional_rnn(mlstm_cell4_z,mlstm_cell4_z, yx_img_y, dtype=tf.float32)    
#
#  x_img_x = tf.reshape(x_x,[-1,n_steps, n_input])
#  xx_img_x=tf.unstack(x_img_x,n_steps,1)
#  outputs_x, output_states_x, _ = rnn.static_bidirectional_rnn(mlstm_cell4_z,mlstm_cell4_z, xx_img_x, dtype=tf.float32)  
  
#---------------------DGCNN--------------------------------------#  
  adj_matrix = tf_util.pairwise_distance(point_cloud)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)

  with tf.variable_scope('transform_net1') as sc:
    transform = input_transform_net(edge_feature, is_training, bn_decay, K=3)

  point_cloud_transformed = tf.matmul(point_cloud, transform)
  adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(point_cloud_transformed, nn_idx=nn_idx, k=k)

  net = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn1', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net1 = net

  adj_matrix = tf_util.pairwise_distance(net)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

  net = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn2', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net2 = net 
  
#---------------------Generating Attention Mask--------------------------------------#    
  ptr_temp = tf.reshape(ptr_temp, [-1, 1, 1, (n_hidden*3)*2])
  rnn_feat_expand = tf.tile(ptr_temp, [1, num_point, 1, 1])
  concat_feature2 = tf.concat([net, rnn_feat_expand], 3)  
  concat_feature2 = tf_util.conv2d(concat_feature2, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='attention_1', bn_decay=bn_decay)
#---------------------Embedding Attention Fusion--------------------------------------#                                   
  edge_feature_att = tf.add(net, tf.multiply(net, concat_feature2))
  edge_feature_att = tf.reduce_max(edge_feature_att, axis=-2, keep_dims=True)
  
  adj_matrix = tf_util.pairwise_distance(net)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k) 
  
  net = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn3', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net3 = net

  adj_matrix = tf_util.pairwise_distance(net)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)  
#---------------------Embedding Attention Fusion--------------------------------------#  
#  edge_feature = tf.add(edge_feature, tf.multiply(edge_feature, concat_feature2))  
#---------------------Embedding Attention Fusion--------------------------------------#
  net = tf_util.conv2d(edge_feature, 128, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn4', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net4 = net
  print('this is the shape of net4', (net4).shape)
  net = tf_util.conv2d(tf.concat([net1, net2, net3, net4,rnn_feat_expand,edge_feature_att], axis=-1), 1024, [1, 1], 
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='agg', bn_decay=bn_decay)
 
  net = tf.reduce_max(net, axis=1, keep_dims=True)   
  
  net = tf.reshape(net, [batch_size, -1]) 
  print('this is the shape of net', (net).shape)
#  ptr_temp = tf.reshape(ptr_temp, [batch_size, -1])  
#  
#  concat_feat = tf.concat([net, ptr_temp], 1)
#  concat_feat = tf.reshape(concat_feat, [batch_size, -1])
  
  net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                                scope='fc1', bn_decay=bn_decay)
  net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                         scope='dp1')
  net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                scope='fc2', bn_decay=bn_decay)
  net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                        scope='dp2')
  net = tf_util.fully_connected(net, n_classes, activation_fn=None, scope='fc3')  

  return net, end_points


def get_loss(net, label, end_points):
  """ pred: B*NUM_CLASSES,
      label: B, """
  labels = tf.one_hot(indices=label, depth=3)
  
#  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
  loss_net = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=net, label_smoothing=0.2)
#  loss_y = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=y, label_smoothing=0.2)
  
  classify_loss = tf.reduce_mean(loss_net )
  return classify_loss


if __name__=='__main__':
  batch_size = 2
  num_pt = 124
  pos_dim = 3

  input_feed = np.random.rand(batch_size, num_pt, pos_dim)
  label_feed = np.random.rand(batch_size)
  label_feed[label_feed>=0.5] = 1
  label_feed[label_feed<0.5] = 0
  label_feed = label_feed.astype(np.int32)

  with tf.Graph().as_default():
    input_pl, label_pl = placeholder_inputs(batch_size, num_pt)
    pos, ftr = get_model(input_pl, tf.constant(True))
    # loss = get_loss(logits, label_pl, None)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      feed_dict = {input_pl: input_feed, label_pl: label_feed}
      res1, res2 = sess.run([pos, ftr], feed_dict=feed_dict)
      print (res1.shape)
      print (res1)

      print (res2.shape)
      print (res2)












