# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:41:05 2014

@author: dell
"""

"""
"""
import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logisticSgd import LogisticRegression, load_data
from mlp import HiddenLayer
from rbm import RBM

isdebug = False
class DBN(object):
    """Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,first_layer_size = 500,
                 hidden_layers_sizes1=[500],
                hidden_layers_sizes2 = [500], n_outs1=10,n_outs2=10):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """

        self.sigmoid_layers1 = []
        self.sigmoid_layers2 = []
        self.rbm_layers1 = []
        self.rbm_layers2 = []
        self.params1 = []
        self.params2 = []
        self.params_first = []
        self.n_layers1 = len(hidden_layers_sizes1)
        self.n_layers2 = len(hidden_layers_sizes2) 
        self.first_sigmoid_layer = 0

        assert self.n_layers1 > 0 or self.n_layers2 > 0

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y1 = T.ivector('y1')
        self.y2 = T.ivector('y2')# the labels are presented as 1D vector
                                 # of [int] labels
        self.field = T.ivector('field')
        
        
        
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        
        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.
        
        self.first_sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=self.x,
                                        n_in=n_ins,
                                        n_out=first_layer_size,
                                        activation=T.nnet.sigmoid)
        self.params_first.extend(self.first_sigmoid_layer.params)
        self.params1.extend(self.first_sigmoid_layer.params)
        self.params2.extend(self.first_sigmoid_layer.params)
        
        #self.first_sigmoid_layer.output size:batch_size * first_layer_size                                
        self.first_logLayer = LogisticRegression(
            input = self.first_sigmoid_layer.output,
            n_in = first_layer_size,
            n_out = 2)   
        self.params_first.extend(self.first_logLayer.params)
            
        self.first_finetune_cost = self.first_logLayer.negative_log_likelihood(self.field)
        self.errors_first = self.first_logLayer.errors(self.field)
        self.first_rbm = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=self.x,
                            n_visible=n_ins,
                            n_hidden=first_layer_size,
                            W=self.first_sigmoid_layer.W,
                            hbias=self.first_sigmoid_layer.b)
        
        
        for i in xrange(self.n_layers1):
                     
            
            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            
            if i == 0:
                layer_input = self.first_sigmoid_layer.output*self.first_logLayer.y_pred.dimshuffle((0,'x'))
                #layer_input = self.first_sigmoid_layer.output
            else:
                layer_input = self.sigmoid_layers1[-1].output            
            
            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = first_layer_size
            else:
                input_size = hidden_layers_sizes1[i]

            

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes1[i],
                                        activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers1.append(sigmoid_layer)

            # its arguably a philosophical question...  but we are
            # going to only declare that the parameters of the
            # sigmoid_layers are parameters of the DBN. The visible
            # biases in the RBM are parameters of those RBMs, but not
            # of the DBN.
            self.params1.extend(sigmoid_layer.params)

            # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes1[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b)
            self.rbm_layers1.append(rbm_layer)
            
         
            
        
        for i in xrange(self.n_layers2):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = first_layer_size
            else:
                input_size = hidden_layers_sizes2[i]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            if i == 0:
                layer_input = T.mul(self.first_sigmoid_layer.output,(1-self.first_logLayer.y_pred.dimshuffle((0,'x'))))
            else:
                layer_input = self.sigmoid_layers2[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes2[i],
                                        activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers2.append(sigmoid_layer)

            # its arguably a philosophical question...  but we are
            # going to only declare that the parameters of the
            # sigmoid_layers are parameters of the DBN. The visible
            # biases in the RBM are parameters of those RBMs, but not
            # of the DBN.
            self.params2.extend(sigmoid_layer.params)

            # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes2[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b)
            self.rbm_layers2.append(rbm_layer)

        # We now need to add a logistic layer on top of the MLP
        
        self.logLayer1 = LogisticRegression(
            input=self.sigmoid_layers1[-1].output,
            n_in=hidden_layers_sizes1[-1],
            n_out=n_outs1)
        self.params1.extend(self.logLayer1.params)
        
        self.logLayer2 = LogisticRegression(
            input=self.sigmoid_layers2[-1].output,
            n_in=hidden_layers_sizes2[-1],
            n_out=n_outs2)
        self.params2.extend(self.logLayer2.params)

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        self.finetune_cost2 = self.logLayer2.negative_log_likelihood(self.y2)
        self.finetune_cost1 = self.logLayer1.negative_log_likelihood(self.y1)
        

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors1 = self.logLayer1.errors(self.y1)
        self.errors2 = self.logLayer2.errors(self.y2)
        
    """
    BUG to fix!!
    """
    def pretraining_functions(self, train_set_x,train_set_field,batch_size, k , field):
        '''Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        all_rbms = [self.first_rbm]
        if field == 1 :
            all_rbms = all_rbms + self.rbm_layers1
        else:    
            all_rbms = all_rbms + self.rbm_layers2
        
        for rbm in all_rbms:
        #for i in xrange(len(all_rbms)):
            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                     persistent=None, k=k)
            # compile the theano function
            fn = theano.function(inputs=[index,
                                theano.Param(learning_rate, default=0.1)],
                                     outputs=cost,
                                     updates=updates,
                                     givens={self.x:
                                        train_set_x[batch_begin:batch_end]
                                        })
                # append `fn` to the list of functions
            pretrain_fns.append(fn)           
                
        return pretrain_fns

    def build_finetune_functions(self, datasets, fields,batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                        the has to contain three pairs, `train`,
                        `valid`, `test` in this order, where each pair
                        is formed of two Theano variables, one for the
                        datapoints, the other for the labels
        :type batch_size: int
        :param batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage

        '''

        (train_set_x, train_set_y1,train_set_y2) = datasets[0]
        (valid_set_x, valid_set_y1,valid_set_y2) = datasets[1]
        (test_set_x, test_set_y1,test_set_y2) = datasets[2]
        train_set_field,valid_set_field,test_set_field = fields
        
        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams2 = T.grad(self.finetune_cost2, self.params2)
        gparams1 = T.grad(self.finetune_cost1, self.params1)
        
        gparams_first = T.grad(self.first_finetune_cost,self.params_first)
        # compute list of fine-tuning updates
        updates1 = []
        updates2 = []
        updates_first = []
        for param, gparam in zip(self.params_first, gparams_first):
            updates_first.append((param, param - gparam * learning_rate))
        for param, gparam in zip(self.params1, gparams1):
            updates1.append((param, param - gparam * learning_rate))
        for param, gparam in zip(self.params2, gparams2):
            updates2.append((param, param - gparam * learning_rate))
          
        train_fn_first = theano.function(inputs=[index],
              outputs=self.first_finetune_cost,
              updates=updates_first,
              givens={self.x: train_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.field: train_set_field[index * batch_size:
                                          (index + 1) * batch_size]})
        train_fn1 = theano.function(inputs=[index],
              outputs=self.finetune_cost1,
              updates=updates1,
              givens={self.x: train_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y1: train_set_y1[index * batch_size:
                                          (index + 1) * batch_size]})
                                          
        train_fn2 = theano.function(inputs=[index],
              outputs=self.finetune_cost2,
              updates=updates2,
              givens={self.x: train_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y2: train_set_y2[index * batch_size:
                                          (index + 1) * batch_size]})

        test_score_i_first = theano.function([index], self.errors_first,
                 givens={self.x: test_set_x[index * batch_size:
                                            (index + 1) * batch_size],
                         self.field: test_set_field[index * batch_size:
                                            (index + 1) * batch_size]})
        test_score_i1 = theano.function([index], self.errors1,
                 givens={self.x: test_set_x[index * batch_size:
                                            (index + 1) * batch_size],
                         self.y1: test_set_y1[index * batch_size:
                                            (index + 1) * batch_size]})
        test_score_i2 = theano.function([index], self.errors2,
                 givens={self.x: test_set_x[index * batch_size:
                                            (index + 1) * batch_size],
                         self.y2: test_set_y2[index * batch_size:
                                            (index + 1) * batch_size]})

        valid_score_i_first = theano.function([index], self.errors_first,
              givens={self.x: valid_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.field: valid_set_field[index * batch_size:
                                          (index + 1) * batch_size]}) 
                                          
        valid_score_i1 = theano.function([index], self.errors1,
              givens={self.x: valid_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y1: valid_set_y1[index * batch_size:
                                          (index + 1) * batch_size]})        
        
        valid_score_i2 = theano.function([index], self.errors2,
              givens={self.x: valid_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y2: valid_set_y2[index * batch_size:
                                          (index + 1) * batch_size]})

        # Create a function that scans the entire validation set
        
        def valid_score_first():
            return [valid_score_i_first(i) for i in xrange(n_valid_batches)]        
        
        def valid_score1():
            return [valid_score_i1(i) for i in xrange(n_valid_batches)]
        def valid_score2():
            return [valid_score_i2(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score_first():
            return [test_score_i_first(i) for i in xrange(n_test_batches)]
        def test_score1():
            return [test_score_i1(i) for i in xrange(n_test_batches)]
        def test_score2():
            return [test_score_i2(i) for i in xrange(n_test_batches)]

        return train_fn1, valid_score1, test_score1,train_fn2,valid_score2,test_score2,train_fn_first,valid_score_first,test_score_first
        
class run_dbn(object):

    def pre_data(self,dataset = r'D:\databases\mnist\mnist.pkl.gz',batch_size = 10):
        self.datasets = load_data(dataset)
        self.batch_size = batch_size;
        self.train_set_x, self.train_set_y = self.datasets[0]
        self.valid_set_x, self.valid_set_y = self.datasets[1]
        self.test_set_x, self.test_set_y = self.datasets[2]
        
        if isdebug:
            self.train_set_x.set_value(self.train_set_x.get_value()[100:200])
            #self.train_set_y.set_value(self.train_set_y.get_value()[0:100])
            self.valid_set_x.set_value(self.valid_set_x.get_value()[100:200])
            #self.valid_set_y.set_value(self.valid_set_y.get_value()[0:100])
            self.test_set_x.set_value(self.test_set_x.get_value()[100:200])
            #self.test_set_y.set_value(self.test_set_y.get_value()[0:100])
            
        self._get_samples_field()
    
        # compute number of minibatches for training, validation and testing
        self.n_train_batches = self.train_set_x.get_value(borrow=True).shape[0] / batch_size
    
    def _get_samples_field(self):
        train_set_val = self.train_set_y.eval()
        valid_set_val = self.valid_set_y.eval()
        test_set_val = self.test_set_y.eval()
        
        self.train_set_field = T.as_tensor_variable(map(lambda x : int(floor(x / 5)),train_set_val))  
        self.valid_set_field = T.as_tensor_variable(map(lambda x : int(floor(x / 5)),valid_set_val))
        self.test_set_field = T.as_tensor_variable(map(lambda x : int(floor(x / 5)),test_set_val))
        
        def get_label1(x):
            if x >= 5:
                return 0
            else:
                return x+1
        def get_label2(x):
            if x < 5:
                return 0
            else:
                return x-4
        
        self.train_set_y1 = T.as_tensor_variable(map(get_label1,train_set_val))        
        self.train_set_y2 = T.as_tensor_variable(map(get_label2,train_set_val)) 
        self.valid_set_y1 = T.as_tensor_variable(map(get_label1,valid_set_val))        
        self.valid_set_y2 = T.as_tensor_variable(map(get_label2,valid_set_val))
        self.test_set_y1 = T.as_tensor_variable(map(get_label1,test_set_val))       
        self.test_set_y2 = T.as_tensor_variable(map(get_label2,test_set_val))
        
    def make_fun(self):
        # numpy random generator
        numpy_rng = numpy.random.RandomState(123)
        print '... building the model'
        # construct the Deep Belief Network
        self.dbn = DBN(numpy_rng=numpy_rng, n_ins=28 * 28,
                       first_layer_size = 1000,
                       hidden_layers_sizes1 = [1000,1000],
                        hidden_layers_sizes2 = [1000,1000],
                  n_outs1=6,n_outs2=6)
                  
        
    
    def pre_train(self,pretraining_epochs = 50,
                 pretrain_lr=0.01, k=1):
        print '... getting the pretraining functions'
        pretraining_fns1 = self.dbn.pretraining_functions(train_set_x=self.train_set_x,
                                                          train_set_field = self.train_set_field,
                                                    batch_size=self.batch_size,
                                                    k=k,field = 1)
        pretraining_fns2 = self.dbn.pretraining_functions(train_set_x=self.train_set_x,
                                                          train_set_field = self.train_set_field,
                                                    batch_size=self.batch_size,
                                                    k=k,field = 2)
    
        print '... pre-training the model'
        #start_time = time.clock()
        ## Pre-train layer-wise
        for i in xrange(self.dbn.n_layers1+1):
            # go through pretraining epochs
            for epoch in xrange(pretraining_epochs):
                # go through the training set
                c = []
                for batch_index in xrange(self.n_train_batches):
                    c.append(pretraining_fns1[i](index=batch_index,
                                                lr=pretrain_lr))
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                print numpy.mean(c)
                
        for i in xrange(self.dbn.n_layers2+1):
            # go through pretraining epochs
            for epoch in xrange(pretraining_epochs):
                # go through the training set
                c = []
                for batch_index in xrange(self.n_train_batches):
                    c.append(pretraining_fns2[i](index=batch_index,
                                                lr=pretrain_lr))
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                print numpy.mean(c)
        #end_time = time.clock()


    def train(self,finetune_lr=0.1,training_epochs=50):
        print '... getting the finetuning functions'
        datas = ((self.train_set_x,self.train_set_y1,self.train_set_y2),
                 (self.valid_set_x,self.valid_set_y1,self.valid_set_y2),
                 (self.test_set_x,self.test_set_y1,self.test_set_y2))
        self.train_fn1, self.validate_model1, self.test_model1,self.train_fn2, self.validate_model2, self.test_model2,self.train_fn_first,self.validate_model_first,self.test_model_first = self.dbn.build_finetune_functions(
                    datasets=datas, 
                    fields = (self.train_set_field,self.valid_set_field,self.test_set_field),
                    batch_size=self.batch_size,
                    learning_rate=finetune_lr)
    
        print '... finetunning the model'
        # early-stopping parameters
        patience = 4 * self.n_train_batches  # look as this many examples regardless
        patience_increase = 2.    # wait this much longer when a new best is
                                  # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(self.n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch
    
        best_params1 = None
        best_params2 = None
        best_validation_loss1 = numpy.inf
        best_validation_loss2 = numpy.inf
        best_validation_loss_first = numpy.inf;
        test_score1 = 0.
        test_score2 = 0.
        test_score_first = 0.
        start_time = time.clock()
    
    
        done_looping = False
        epoch = 0
    
        #while (epoch < training_epochs) and (not done_looping):
        while (epoch < training_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(self.n_train_batches):
                
                minibatch_avg_cost_first = self.train_fn_first(minibatch_index)
                
                iter = (epoch - 1) * self.n_train_batches + minibatch_index
    
                if (iter + 1) % validation_frequency == 0:
    
                    validation_losses_first = self.validate_model_first()
                    
                    this_validation_loss_first = numpy.mean(validation_losses_first)
                    
                    print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                          (epoch, minibatch_index + 1, self.n_train_batches,
                           this_validation_loss_first * 100.))
                    
    
                    # if we got the best validation score until now
                    if this_validation_loss_first < best_validation_loss_first:
    
                        #improve patience if loss improvement is good enough
                        if (this_validation_loss_first < best_validation_loss_first *
                            improvement_threshold):
                            patience = max(patience, iter * patience_increase)
    
                        # save best validation score and iteration number
                        best_validation_loss_first = this_validation_loss_first
                        best_iter_first = iter
    
                        # test it on the test set
                        test_losses_first = self.test_model_first()
                        test_score_first = numpy.mean(test_losses_first)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, self.n_train_batches,
                               test_score_first * 100.))
                
                if patience <= iter:
                    done_looping = True
                    break    
                
    
    
        done_looping = False
        epoch = 0
    
        while (epoch < training_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(self.n_train_batches):
                
                minibatch_avg_cost1 = self.train_fn1(minibatch_index)
                minibatch_avg_cost2 = self.train_fn2(minibatch_index)
                #minibatch_avg_cost_first = self.train_fn_first(minibatch_index)
                
                iter = (epoch - 1) * self.n_train_batches + minibatch_index
                if (iter + 1) % validation_frequency == 0:
    
                    validation_losses1 = self.validate_model1()
                    validation_losses2 = self.validate_model2()
                    #validation_losses_first = self.validate_model_first()
                    
                    this_validation_loss1 = numpy.mean(validation_losses1)
                    this_validation_loss2 = numpy.mean(validation_losses2)
                    #this_validation_loss_first = numpy.mean(validation_losses_first)
                    
                    print('path 1, epoch %i, minibatch %i/%i, validation error %f %%' % \
                          (epoch, minibatch_index + 1, self.n_train_batches,
                           this_validation_loss1 * 100.))
                    print('path 2, epoch %i, minibatch %i/%i, validation error %f %%' % \
                          (epoch, minibatch_index + 1, self.n_train_batches,
                           this_validation_loss2 * 100.))
                    """
                    print('first layer, epoch %i, minibatch %i/%i, validation error %f %%' % \
                          (epoch, minibatch_index + 1, self.n_train_batches,
                           this_validation_loss_first * 100.))
                    """
    
                    # if we got the best validation score until now
                    if this_validation_loss1 < best_validation_loss1:
    
                        #improve patience if loss improvement is good enough
                        if (this_validation_loss1 < best_validation_loss1 *
                            improvement_threshold):
                            patience = max(patience, iter * patience_increase)
    
                        # save best validation score and iteration number
                        best_validation_loss1 = this_validation_loss1
                        best_iter1 = iter
    
                        # test it on the test set
                        test_losses1 = self.test_model1()
                        test_score1 = numpy.mean(test_losses1)
                        print(('     path 1, epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, self.n_train_batches,
                               test_score1 * 100.))
                               
                    # if we got the best validation score until now
                    if this_validation_loss2 < best_validation_loss2:
    
                        #improve patience if loss improvement is good enough
                        if (this_validation_loss2 < best_validation_loss2 *
                            improvement_threshold):
                            patience = max(patience, iter * patience_increase)
    
                        # save best validation score and iteration number
                        best_validation_loss2 = this_validation_loss2
                        best_iter2 = iter
    
                        # test it on the test set
                        test_losses2 = self.test_model2()
                        test_score2 = numpy.mean(test_losses2)
                        print(('     path 2, epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, self.n_train_batches,
                               test_score2 * 100.))
                    """           
                    # if we got the best validation score until now
                    if this_validation_loss_first < best_validation_loss_first:
    
                        #improve patience if loss improvement is good enough
                        if (this_validation_loss_first < best_validation_loss_first *
                            improvement_threshold):
                            patience = max(patience, iter * patience_increase)
    
                        # save best validation score and iteration number
                        best_validation_loss_first = this_validation_loss_first
                        best_iter_first = iter
    
                        # test it on the test set
                        test_losses_first = self.test_model_first()
                        test_score_first = numpy.mean(test_losses_first)
                        print(('     first layer, epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, self.n_train_batches,
                               test_score_first * 100.))
                    """
                if patience <= iter:
                    done_looping = True
                    break
        end_time = time.clock()
        """
        print(('Optimization complete with best validation score of %f %%,'
               'with test performance %f %%') %
                     (best_validation_loss * 100., test_score * 100.))
        """
        """             
        print >> sys.stderr, ('The fine tuning code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time)
                                                  / 60.))
        """

def test_DBN(finetune_lr=0.1, pretraining_epochs=50,
             pretrain_lr=0.01, k=1, training_epochs=50,
             dataset='Z:\share\databases\mnist\mnist.pkl.gz', batch_size=10):
    """
    Demonstrates how to train and test a Deep Belief Network.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining
    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training
    :type k: int
    :param k: number of Gibbs steps in CD/PCD
    :type training_epochs: int
    :param training_epochs: maximal number of iterations ot run the optimizer
    :type dataset: string
    :param dataset: path the the pickled dataset
    :type batch_size: int
    :param batch_size: the size of a minibatch
    """

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print '... building the model'
    # construct the Deep Belief Network
    dbn = DBN(numpy_rng=numpy_rng, n_ins=28 * 28,
              hidden_layers_sizes=[1000, 1000, 1000],
              n_outs=10)

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)

    print '... pre-training the model'
    start_time = time.clock()
    ## Pre-train layer-wise
    for i in xrange(dbn.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)

    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = dbn.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr)

    print '... finetunning the model'
    # early-stopping parameters
    patience = 4 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.    # wait this much longer when a new best is
                              # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))


if __name__ == '__main__':
    #test_DBN()
    x = run_dbn()
    x.pre_data()
    x.make_fun()
    x.pre_train()
    x.train()