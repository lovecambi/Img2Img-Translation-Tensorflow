# -*- coding: utf-8 -*-
"""
Created on Mon May  8 19:58:09 2017

@author: fankai
"""

import tensorflow as tf
import tensorflow.contrib.layers as tcl
import numpy as np
import os

from glob import glob
from ops import *

datanames = glob(os.path.join("data","line3","*.jpg"))
finalpath = "data/celebA3/"

class Img2ImgCGAN(object):
    
    def __init__(self, image_size, input_c_dim, output_c_dim, 
                 train_mode=True, model_path="model/CGAN", outpath="imgs/", output_iters=100,
                 gf_dim=64, df_dim=64, L1_lambda=100, 
                 batch_size=64, d_learning_rate=1e-4, g_learning_rate=3e-4, eps=1e-8):
        """
        Args:
        
        """
        self.H, self.W = image_size
        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim
        
        self.train_mode = train_mode
        self.model_path = model_path
        self.outpath = outpath
        self.output_iters = output_iters
        
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.L1_lambda = L1_lambda
        
        self.batch_size = batch_size
        
        # build model        
        self.real_A = tf.placeholder(tf.float32,[self.batch_size, self.H, self.W, self.input_c_dim])
        self.real_B = tf.placeholder(tf.float32,[self.batch_size, self.H, self.W, self.output_c_dim])
        
        if self.H == 64:
            self.generator = self.generator64
        elif self.H == 128:
            self.generator = self.generator128
        elif self.H == 256:
            self.generator = self.generator256
        else:
            raise NotImplementedError
        
        self.fake_B = self.generator(self.real_A)
        self.real_AB = tf.concat([self.real_A, self.real_B], axis=3) 
        self.fake_AB = tf.concat([self.real_A, self.fake_B], axis=3)
        
        rl = self.discriminator(self.real_AB)
        gl = self.discriminator(self.fake_AB, reuse=True)
        
        self.d_loss = ganloss(rl) + ganloss(gl, 0.01)
        self.l1_loss = self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))
        self.g_loss = ganloss(gl) + self.l1_loss
        
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        
        # optimizer
        self.d_optimizer = tf.train.AdamOptimizer(d_learning_rate, beta1=0.5, beta2=0.999)
        d_grads = self.d_optimizer.compute_gradients(self.d_loss, self.d_vars)
        clip_d_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in d_grads if grad is not None]
        self.d_optimizer = self.d_optimizer.apply_gradients(clip_d_grads)
        
        self.g_optimizer = tf.train.AdamOptimizer(g_learning_rate, beta1=0.5, beta2=0.999)
        g_grads = self.g_optimizer.compute_gradients(self.g_loss, self.g_vars)
        clip_g_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in g_grads if grad is not None]
        self.g_optimizer = self.g_optimizer.apply_gradients(clip_g_grads)
    

    def train(self, trainnames, outimgpath, testnames, max_epoch=100, K=1):
        
        with tf.Session() as sess:
            
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            
            i = 0
            for epoch in range(max_epoch):
                
                for excerpt in iterate_minibatches_u(len(trainnames), self.batch_size, shuffle=True):   
                    
                    batch_names = trainnames[excerpt]
                    itrain = [ scipy.misc.imread(img_name) for img_name in batch_names ]
                    itrain = np.array(itrain).astype(np.float32)[:,:,:,None] / 127.5 - 1   
                    otrain = [ scipy.misc.imread(outimgpath + os.path.basename(img_name)) for img_name in batch_names ]
                    otrain = np.array(otrain).astype(np.float32) / 127.5 - 1  
                    
                    _, Ld = sess.run([self.d_optimizer, self.d_loss], 
                                     feed_dict={self.real_A: itrain, 
                                                self.real_B: otrain})
                    
                    _, Lg, Ll1 = sess.run([self.g_optimizer, self.g_loss, self.l1_loss], 
                                     feed_dict={self.real_A: itrain, 
                                                self.real_B: otrain})
                
                    if i % self.output_iters == 0:
                        
                        print("Iter=%d Train: Ld: %f Lg: %f Ll1: %f" % (i, Ld, Lg, Ll1))
                        
                        self.get_test_performance(sess, testnames, outimgpath, i)
                        
                    i += 1
                    
                self.save_model(saver, sess, step=epoch)
                
    
    def save_model(self, saver, sess, step):
        """
        save model with path error checking
        """
        if self.model_path is None:
            my_path = "model/myckpt" # default path in tensorflow saveV2 format
            # try to make directory
            if not os.path.exists("model"):
                try:
                    os.makedirs("model")
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
        else: 
            my_path = self.model_path
                
        saver.save(sess, my_path, global_step=step)
    
    
    def get_test_performance(self, sess, testnames, outimgpath, i):
        
        #self.train_mode = False
        Lds, Lgs, Ll1s = [], [], []
        for excerpt in iterate_minibatches_u(len(testnames), self.batch_size, shuffle=False):
            
            batch_names = testnames[excerpt]
            itest = [ scipy.misc.imread(img_name) for img_name in batch_names ]
            itest = np.array(itest).astype(np.float32)[:,:,:,None] / 127.5 - 1   
            otest = [ scipy.misc.imread(outimgpath + os.path.basename(img_name)) for img_name in batch_names ]
            otest = np.array(otest).astype(np.float32) / 127.5 - 1 
                                    
            Ld, Lg, Ll1 = sess.run([self.d_loss, self.g_loss, self.l1_loss], 
                              feed_dict={self.real_A: itest, 
                                         self.real_B: otest})
            
            Lds.append(Ld)
            Lgs.append(Lg)
            Ll1s.append(Ll1)
            
        ofake = sess.run(self.fake_B, feed_dict={self.real_A: itest})
        
        n = int(np.sqrt(self.batch_size))
        assert (n * n == self.batch_size)

        fake_img = toshow(ofake)
        inpt_img = toshow(itest)
        true_img = toshow(otest)
        
        scipy.misc.imsave(self.outpath + "fake" + str(i) + ".jpg", fake_img)
        scipy.misc.imsave(self.outpath + "init.jpg", inpt_img)
        scipy.misc.imsave(self.outpath + "final.jpg", true_img)
        
        Ld_ = np.mean(np.array(Lds), axis=0)
        Lg_ = np.mean(np.array(Lgs), axis=0)
        Ll1_ = np.mean(np.array(Ll1s), axis=0)
        
        print("Iter=%d Test: Ld: %f Lg: %f Ll1: %f" % (i, Ld_, Lg_, Ll1_))
        #self.train_mode = True
        
    
    def generator256(self, x, y=None):
        
        with tf.variable_scope("generator"):
            
            e1 = conv2d(x, self.gf_dim, bias=True, name='g_e1_conv')  # bs x H/2 x W/2 x gf
            
            e2 = conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv')  # bs x H/4 x W/4 x 2*gf
            e2 = batch_norm(e2, self.train_mode, name='g_bn_e2')
            
            e3 = conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv')  # bs x H/8 x W/8 x 4*gf
            e3 = batch_norm(e3, self.train_mode, name='g_bn_e3')
            
            e4 = conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv')  # bs x H/16 x W/16 x 8*gf
            e4 = batch_norm(e4, self.train_mode, name='g_bn_e4')
            
            e5 = conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv')  # bs x H/32 x W/32 x 8*gf
            e5 = batch_norm(e5, self.train_mode, name='g_bn_e5')
            
            e6 = conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv')  # bs x H/64 x W/64 x 8*gf
            e6 = batch_norm(e6, self.train_mode, name='g_bn_e6')
            
            e7 = conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv')  # bx x H/128 x W/128 x 8*gf
            e7 = batch_norm(e7, self.train_mode, name='g_bn_e7')
            
            e8 = conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv')  # bx x H/256 x W/256 x 8*gf
            e8 = batch_norm(e8, self.train_mode, name='g_bn_e8')
            
            d1 = deconv2d(tf.nn.relu(e8), [self.batch_size, int(self.H/128), int(self.W/128), self.gf_dim*8], name='g_d1')
            d1 = tf.nn.dropout(batch_norm(d1, name='g_bn_d1'), 0.5)
            d1 = tf.concat([d1, e7], 3)  # bs x H/128 x W/128 x 16*gf
            
            d2 = deconv2d(tf.nn.relu(d1), [self.batch_size, int(self.H/64), int(self.W/64), self.gf_dim*8], name='g_d2')
            d2 = tf.nn.dropout(batch_norm(d2, self.train_mode, name='g_bn_d2'), 0.5)
            d2 = tf.concat([d2, e6], 3)  # bs x H/64 x W/64 x 16*gf
            
            d3 = deconv2d(tf.nn.relu(d2), [self.batch_size, int(self.H/32), int(self.W/32), self.gf_dim*8], name='g_d3')
            d3 = tf.nn.dropout(batch_norm(d3, self.train_mode, name='g_bn_d3'), 0.5)
            d3 = tf.concat([d3, e5], 3)
            
#            dy = deconv2d(y, [self.batch_size, int(self.H/32), int(self.W/32), self.gf_dim], 
#                          int(self.H/32), int(self.W/32), 1, 1, bias=True, padding='VALID', name='g_dy')                    
#            d3 = tf.concat([d3, e5, dy], 3)  # bs x H/32 x W/32 x 17*gf
            
            d4 = deconv2d(tf.nn.relu(d3), [self.batch_size, int(self.H/16), int(self.W/16), self.gf_dim*8], name='g_d4')
            d4 = batch_norm(d4, self.train_mode, name='g_bn_d4')
            d4 = tf.concat([d4, e4], 3)  # bs x H/16 x W/16 x 16*gf
            
            d5 = deconv2d(tf.nn.relu(d4), [self.batch_size, int(self.H/8), int(self.W/8), self.gf_dim*4], name='g_d5')
            d5 = batch_norm(d5, self.train_mode, name='g_bn_d5')
            d5 = tf.concat([d5, e3], 3)  # bs x H/8 x W/8 x 8*gf
            
            d6 = deconv2d(tf.nn.relu(d5), [self.batch_size, int(self.H/4), int(self.W/4), self.gf_dim*2], name='g_d6')
            d6 = batch_norm(d6, self.train_mode, name='g_bn_d6')
            d6 = tf.concat([d6, e2], 3)  # bs x H/128 x W/128 x 4*gf
            
            d7 = deconv2d(tf.nn.relu(d6), [self.batch_size, int(self.H/2), int(self.W/2), self.gf_dim], name='g_d7')
            d7 = batch_norm(d7, self.train_mode, name='g_bn_d7')
            d7 = tf.concat([d7, e1], 3)  # bs x H/2 x W/2 x 2*gf
            
            d8 = deconv2d(tf.nn.relu(d7), [self.batch_size, self.H, self.W, self.ouput_c_dim], bias=True, name='g_d8')
            
            return tf.nn.tanh(d8)
    
    
    def generator128(self, x, y=None):
        
        with tf.variable_scope("generator"):
            
            e1 = conv2d(x, self.gf_dim, bias=True, name='g_e1_conv')  # bs x H/2 x W/2 x gf
            
            e2 = conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv')  # bs x H/4 x W/4 x 2*gf
            e2 = batch_norm(e2, self.train_mode, name='g_bn_e2')
            
            e3 = conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv')  # bs x H/8 x W/8 x 4*gf
            e3 = batch_norm(e3, self.train_mode, name='g_bn_e3')
            
            e4 = conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv')  # bs x H/16 x W/16 x 8*gf
            e4 = batch_norm(e4, self.train_mode, name='g_bn_e4')
            
            e5 = conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv')  # bs x H/32 x W/32 x 8*gf
            e5 = batch_norm(e5, self.train_mode, name='g_bn_e5')
            
            e6 = conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv')  # bs x H/64 x W/64 x 8*gf
            e6 = batch_norm(e6, self.train_mode, name='g_bn_e6')
            
            e7 = conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv')  # bx x H/128 x W/128 x 8*gf
            e7 = batch_norm(e7, self.train_mode, name='g_bn_e7')
                      
            d1 = deconv2d(tf.nn.relu(e7), [self.batch_size, int(self.H/64), int(self.W/64), self.gf_dim*8], name='g_d1')
            d1 = tf.nn.dropout(batch_norm(d1, name='g_bn_d1'), 0.5)
            d1 = tf.concat([d1, e6], 3)  # bs x H/64 x W/64 x 16*gf
            
            d2 = deconv2d(tf.nn.relu(d1), [self.batch_size, int(self.H/32), int(self.W/32), self.gf_dim*8], name='g_d2')
            d2 = tf.nn.dropout(batch_norm(d2, self.train_mode, name='g_bn_d2'), 0.5)
            d2 = tf.concat([d2, e5], 3)  # bs x H/32 x W/32 x 16*gf
            
            d3 = deconv2d(tf.nn.relu(d2), [self.batch_size, int(self.H/16), int(self.W/16), self.gf_dim*8], name='g_d3')
            d3 = tf.nn.dropout(batch_norm(d3, self.train_mode, name='g_bn_d3'), 0.5)
            d3 = tf.concat([d3, e4], 3)
            
#            dy = deconv2d(y, [self.batch_size, int(self.H/16), int(self.W/16), self.gf_dim], 
#                          int(self.H/16), int(self.W/16), 1, 1, bias=True, padding='VALID', name='g_dy')           
#            d3 = tf.concat([d3, e4, dy], 3)  # bs x H/16 x W/16 x 17*gf
            
            d4 = deconv2d(tf.nn.relu(d3), [self.batch_size, int(self.H/8), int(self.W/8), self.gf_dim*4], name='g_d4')
            d4 = batch_norm(d4, self.train_mode, name='g_bn_d4')
            d4 = tf.concat([d4, e3], 3)  # bs x H/8 x W/8 x 8*gf
            
            d5 = deconv2d(tf.nn.relu(d4), [self.batch_size, int(self.H/4), int(self.W/4), self.gf_dim*2], name='g_d5')
            d5 = batch_norm(d5, self.train_mode, name='g_bn_d5')
            d5 = tf.concat([d5, e2], 3)  # bs x H/4 x W/4 x 4*gf
            
            d6 = deconv2d(tf.nn.relu(d5), [self.batch_size, int(self.H/2), int(self.W/2), self.gf_dim], name='g_d6')
            d6 = batch_norm(d6, self.train_mode, name='g_bn_d6')
            d6 = tf.concat([d6, e1], 3)  # bs x H/2 x W/2 x 2*gf
            
            d7 = deconv2d(tf.nn.relu(d6), [self.batch_size, self.H, self.W, self.output_c_dim], name='g_d7')
            
            return tf.nn.tanh(d7)
    
    
    def generator64(self, x, y=None):
        
        with tf.variable_scope("generator"):
            
            e1 = conv2d(x, self.gf_dim, bias=True, name='g_e1_conv')  # bs x H/2 x W/2 x gf
            
            e2 = conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv')  # bs x H/4 x W/4 x 2*gf
            e2 = batch_norm(e2, self.train_mode, name='g_bn_e2')
            
            e3 = conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv')  # bs x H/8 x W/8 x 4*gf
            e3 = batch_norm(e3, self.train_mode, name='g_bn_e3')
            
            e4 = conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv')  # bs x H/16 x W/16 x 8*gf
            e4 = batch_norm(e4, self.train_mode, name='g_bn_e4')
            
            e5 = conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv')  # bs x H/32 x W/32 x 8*gf
            e5 = batch_norm(e5, self.train_mode, name='g_bn_e5')
            
            e6 = conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv')  # bs x H/64 x W/64 x 8*gf
            e6 = batch_norm(e6, self.train_mode, name='g_bn_e6')
                                  
            d1 = deconv2d(tf.nn.relu(e6), [self.batch_size, int(self.H/32), int(self.W/32), self.gf_dim*8], name='g_d1')
            d1 = tf.nn.dropout(batch_norm(d1, name='g_bn_d1'), 0.5)
            d1 = tf.concat([d1, e5], 3)  # bs x H/32 x W/32 x 16*gf
            
            d2 = deconv2d(tf.nn.relu(d1), [self.batch_size, int(self.H/16), int(self.W/16), self.gf_dim*8], name='g_d2')
            d2 = tf.nn.dropout(batch_norm(d2, self.train_mode, name='g_bn_d2'), 0.5)
            d2 = tf.concat([d2, e4], 3)
            
#            dy = deconv2d(y, [self.batch_size, int(self.H/16), int(self.W/16), self.gf_dim], 
#                          int(self.H/16), int(self.W/16), 1, 1, bias=True, padding='VALID', name='g_dy')
#            d2 = tf.concat([d2, e4, dy], 3)  # bs x H/16 x W/16 x 17*gf
            
            d3 = deconv2d(tf.nn.relu(d2), [self.batch_size, int(self.H/8), int(self.W/8), self.gf_dim*4], name='g_d3')
            d3 = tf.nn.dropout(batch_norm(d3, self.train_mode, name='g_bn_d3'), 0.5)
            d3 = tf.concat([d3, e3], 3)  # bs x H/8 x W/8 x 8*gf
            
            d4 = deconv2d(tf.nn.relu(d3), [self.batch_size, int(self.H/4), int(self.W/4), self.gf_dim*2], name='g_d4')
            d4 = batch_norm(d4, self.train_mode, name='g_bn_d4')
            d4 = tf.concat([d4, e2], 3)  # bs x H/4 x W/4 x 4*gf
            
            d5 = deconv2d(tf.nn.relu(d4), [self.batch_size, int(self.H/2), int(self.W/2), self.gf_dim], name='g_d5')
            d5 = batch_norm(d5, self.train_mode, name='g_bn_d5')
            d5 = tf.concat([d5, e1], 3)  # bs x H/2 x W/2 x 2*gf
            
            d6 = deconv2d(tf.nn.relu(d5), [self.batch_size, self.H, self.W, self.output_c_dim], name='g_d6')
            
            return tf.nn.tanh(d6)
            
    
    def discriminator(self, x, y=None, reuse=None):
        
        with tf.variable_scope("discriminator", reuse=reuse):
            
            h0 = lrelu(conv2d(x, self.df_dim, bias=True, name='d_h0_conv'))  # bx x H/2 x W/2 x df
            
            h1 = conv2d(h0, self.df_dim*2, name='d_h1_conv')
            h1 = lrelu(batch_norm(h1, self.train_mode, name='d_bn1'))  # bx x H/4 x W/4 x 2*df
            
            h2 = conv2d(h1, self.df_dim*4, name='d_h2_conv')
            h2 = lrelu(batch_norm(h2, self.train_mode, name='d_bn2'))  # bx x H/8 x W/8 x 4*df
        
            h3 = conv2d(h2, self.df_dim*8, name='d_h3_conv')
            h3 = lrelu(batch_norm(h3, self.train_mode, name='d_bn3'))  # # bx x H/16 x W/16 x 8*df
            
            h4 = conv2d(h3, 1, int(self.H/16), int(self.W/16), 1, 1, bias=True, name='d_h4_conv')  # bs x 1 x 1 x 1
            
            return h4


if __name__ == "__main__":
    
    indices = np.arange(len(datanames))
    np.random.RandomState(np.random.randint(1,2147462579)).shuffle(indices)
    parition = int(len(datanames) * 0.8)
    datanames = np.array(datanames)
    trainnames = datanames[indices[:parition]]
    testnames = datanames[indices[parition:]]
    
    mymodel = Img2ImgCGAN(image_size=[128, 128], input_c_dim=1, output_c_dim=3, train_mode=True)                       
    mymodel.train(trainnames, finalpath, testnames, max_epoch=10)
