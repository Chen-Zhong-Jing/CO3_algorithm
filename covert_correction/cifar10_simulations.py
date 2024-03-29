#------------------------------------------------------------------------------
"""Libraries"""
# Time
import time

# Storage
import os
# import shelve
import pickle
import argparse
from pack.parser_utils import str2bool
from pack.fp4 import fp4_121_bin_edges,fp4_best_scalar
from pack.fp8 import fp8_152_bin_edges,fp8_best_scalar
# Math
import math
import numpy as np
from scipy import stats
# DNN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.cifar10_models import build_model
import torch
from pack.csvec import CSVec
from pack.dweibull_quantizer import dweibullquantizer
# Compressions
# from utils.fp8 import fp8_152_bin_edges

# Plotting
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------
"Parser Arguments"
def parse_args():
    parser = argparse.ArgumentParser(description='Variables for Cifar10 Training')

    # Model Details
    parser.add_argument('--batch-size', type=int, default=64, help='Batch Size for Training and Testing')
    parser.add_argument('--epoch-num', type=int, default=150, help='End Iterations for Training')
    
    parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer to use. Can be either SGD or Adam')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning Rate for model')
    parser.add_argument('--lr-decay', type=float, default=0.1, help='Learning Rate decay coefficient')
    parser.add_argument('--lr-decay-freq', default=50, type=int,help='lr decay frequency in epoch (default: 30)')
    parser.add_argument('--momentum', type=float, default=0, help='Momentum for model')
    parser.add_argument('--model-name', type=str, default='ResNet50V2', help='Model to Load for Training')
    
    modes = ["sketch", "CO3", "uncompress", "tinyscript"]
    parser.add_argument("--mode", choices=modes, default="tinyscript")
    
    # CO3 Details
    quantizations = ["fp4", "fp8"]
    parser.add_argument("--quantization", choices=quantizations, default="fp8", help='quantization type from IEEE754')
    parser.add_argument('--correction-coeff', type=float, default=0.7, help='Correction coefficient for model')
   
    # Sketch Details
    #parser.add_argument("--k", type=int, default=588496)
    parser.add_argument("--num_cols", type=int, default=1176992)
    parser.add_argument("--num_rows", type=int, default=5)
    parser.add_argument("--num_blocks", type=int, default=20)
    
    
    # Tinyscript Details
    parser.add_argument('--R', type=int, default=8, help='How many bit rate')
    parser.add_argument('--M', type=int, default=0, help='M for K-means Quantizer')
    
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"],default=default_device,help="Device (cuda or cpu)")
    
    # Print and Plot
    parser.add_argument('--plot-state', type=str2bool, default='True', help='Whether to plot the results')
    parser.add_argument('--verbose-epoch', type=str2bool, default='True', help='Whether to print the accuracy results on a per batch basis')
    parser.add_argument('--verbose-batch', type=str2bool, default='True', help='Whether to print the accuracy results on a per batch basis')
    
    # Save Details
    parser.add_argument('--filepath', type=str, default='D:/Tinyscript/8bit/', help='Path used for saving the gradients and statistics')
    parser.add_argument('--trial-num', type=int, default=0, help='Simulation index')

    args = parser.parse_args()
    return args

#-----------------------------------------------------------------------------
def run_experiment(args):
    "Global Configurations and Variables"
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    sess = tf.compat.v1.Session(config=config)

    tr_acc = tf.keras.metrics.CategoricalAccuracy()
    #tr_loss = tf.keras.metrics.CategoricalCrossentropy()
    
    # Setting CO3 Details
    if args.quantization == 'fp4':
        bin_centers,bin_edges,bin_dict = fp4_121_bin_edges()
    elif args.quantization == 'fp8':
        bin_centers,bin_edges,bin_dict = fp8_152_bin_edges()
    gamma = args.correction_coeff
    
    cache_dweibull_quantization_table = {}
    
    # -----------------------------------------------------------------------------  
    "Functions"  
    # Performs the optimizer update and extracts the gradients
    def step(X, y, error, s):
        # global opt, model
        # keep track of our gradients
        with tf.GradientTape() as tape:
            # make a prediction using the model and then calculate the
            # loss
            logits = model(X, training=True)
            loss = tf.keras.losses.categorical_crossentropy(y,logits)
            tr_acc.update_state(y,logits)
            # Compute the loss and the initial gradient
        grads = tape.gradient(loss, model.trainable_variables)
        grads_np = []
        quantized_np = []
        grad_plus_error = []
        grads_vec = []
        grads_size = np.zeros(len(grads),dtype=int)
        if args.mode == "CO3":
            for idx in range(len(grads)):
                temp = grads[idx].numpy()
                grads_np.append(temp.flatten())
                shape = np.shape(temp)
                temp += gamma*error[idx]
                grad_plus_error.append(temp.flatten())
                quantized_gradient = np.reshape(bin_centers[np.searchsorted(bin_edges, temp.flatten()*scalar[idx])]/scalar[idx], shape)
                quantized_np.append(quantized_gradient)
                error[idx] = temp - quantized_gradient
                grads[idx] = tf.convert_to_tensor(quantized_gradient.astype(dtype=np.float32))
                    
        elif args.mode == "sketch":
            for idx in range(len(grads)):
                temp = grads[idx].numpy().astype(dtype=np.float32)
                grads_size[idx] = int(temp.size)
                grads_vec.append(torch.from_numpy(temp.flatten()))

            grads_vec = torch.cat(grads_vec)
            grads_vec = grads_vec.to(args.device)
            sketch = CSVec(d=grads_size.sum(), c=args.num_cols,r=args.num_rows, device=args.device,numBlocks=args.num_blocks)
            sketch.accumulateVec(grads_vec)
            """
            # gradient clipping
            if compute_grad and args.max_grad_norm is not None:
                sketch = clip_grad(args.max_grad_norm, sketch)
            """
            update = sketch.unSketch(k=grads_size.sum()).cpu().numpy()

            start = 0
            end = 0
            for idx in range(len(grads)):
                end = start + grads_size[idx]
                grads[idx] = tf.convert_to_tensor(update[start:end].reshape(grads[idx].shape))
                start = end
        elif args.mode == "tinyscript":
            for idx in range(len(grads)):
                temp = grads[idx].numpy().astype(dtype=np.float32)
                grads_np.append(temp.flatten())
                shape = np.shape(temp)
                temp += gamma*error[idx]
                grad_plus_error.append(temp.flatten())

                #thresholds, quantization_centers = dweibullquantizer(temp.flatten(), args.R, 100, args.M, cache_dweibull_quantization_table)
                labels = np.digitize(temp,thresholds[idx])
                index_labels_false = np.where(labels == 2**args.R)
                labels[index_labels_false] = 2**args.R - 1
                quantized_gradient = quantization_centers[idx,labels]
                quantized_np.append(quantized_gradient.flatten())
                error[idx] = temp - quantized_gradient
                grads[idx] = tf.convert_to_tensor(quantized_gradient.astype(dtype=np.float32))
            

        opt.apply_gradients(zip(grads, model.trainable_variables))
        
        return np.array(grads_np,dtype=object),np.mean(loss.numpy()),error,np.array(quantized_np,dtype=object),np.array(grad_plus_error,dtype=object)

     # -----------------------------------------------------------------------------   
    "Preparing and Checking Parser"
    if args.optimizer not in ['SGD', 'Adam']:
        assert False, 'Optimizer not found'
    
    # Check if main path exists
    if not os.path.exists(args.filepath):
        assert False, '\"' + args.filepath + '\" Path does not exist'
    
    # Check if subfolders exists, otherwise it creates them
    filepath = args.filepath + args.model_name + '/'
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath +=  args.optimizer + '/'
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath +=  str(args.trial_num) + '/'
    if not os.path.exists(filepath):
            os.makedirs(filepath)

    # -----------------------------------------------------------------------------
    "Data Loading"
    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    # -----------------------------------------------------------------------------
    "Data Preprocessing"
    # Convert class vectors to binary class matrices.
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    
    # ------------------------------------------------------------------------------
    "Data augmentation"
    datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
    
    datagen.fit(x_train)
    # -----------------------------------------------------------------------------
    "Model + Optimizer Definition"
    # Picking the Model
    model = build_model[args.model_name](x_train)
    quantized_error = []
    for idx in range(len(model.trainable_variables)):
        shape = np.shape(model.trainable_variables[idx].numpy())
        quantized_error.append(np.zeros(shape))
        
    "Initialize Tinyscript quantization tables"
    thresholds = np.zeros((len(model.trainable_variables), 2**args.R-1))
    quantization_centers = np.zeros((len(model.trainable_variables),2**args.R))
    # Initiate the optimizer
    if args.optimizer == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=args.learning_rate,
                                   momentum=args.momentum)
    elif args.optimizer == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    # -----------------------------------------------------------------------------
    "Training the Model"
    batch_size = args.batch_size
    epoch_num = args.epoch_num
    batch_total = math.ceil(len(x_train) / batch_size)
    
    train_acc = np.zeros(epoch_num)
    test_acc = np.zeros(epoch_num)
    train_loss = np.zeros(epoch_num)
    test_loss = np.zeros(epoch_num)
    scalar = np.ones(len(model.trainable_variables))
    
    elapsed = time.time()
    for epoch in range(epoch_num):
        epoch_time = time.time()
        a0 = np.zeros(batch_total)
        l0 = np.zeros(batch_total)
        #opt.learning_rate = args.learning_rate * (args.lr_decay ** (epoch // args.lr_decay_freq))
        for batches, (x_batch,y_batch) in enumerate(datagen.flow(x_train, y_train, batch_size=batch_size)):
            grads,l1,quantized_error,quantized_grad, grad_plus_error = step(x_batch,y_batch,quantized_error,scalar)
    
            a1 = tr_acc.result().numpy()
            a0[batches] = a1
            l0[batches] = l1
            
            if args.verbose_batch:
                print('Batch #', batches, ' Acc:', str(f'{a1:0.4f}'))
            

            if batches == 0:
                avr_grads = np.zeros_like(grads)
                avr_quantized_error = np.zeros_like(quantized_error)
                avr_quantized_grad = np.zeros_like(quantized_grad)
                avr_grads_plus_error = np.zeros_like(grad_plus_error)
                
  
            avr_grads += grads
            avr_quantized_error += quantized_error
            avr_quantized_grad += quantized_grad
            avr_grads_plus_error += grad_plus_error
            
            if (batches + 1) >= batch_total:
                
                avr_grads_plus_error /= batches
                avr_grads /= batches
                
                # Compute the scalar for each layer minimize the L2 loss of fp convertion.
                if args.mode == "CO3" and epoch % 5 == 4:
                    for l in range(len(avr_grads_plus_error)):
                        fit_params = stats.gennorm.fit(avr_grads_plus_error[l])
                        
                        if args.quantization == 'fp4':
                            scalar[l] = fp4_best_scalar(fit_params)
                        elif args.quantization == 'fp8':
                            scalar[l] = fp8_best_scalar(fit_params)
                        
                if args.mode == "tinyscript" and epoch % 5 == 0:
                    for l in range(len(avr_grads)):
                        thresholds[l], quantization_centers[l] = dweibullquantizer(avr_grads[l], args.R, 100, args.M, cache_dweibull_quantization_table)
                        
                "Storing the average of each term"
                data = {'avr_grad': avr_grads,
                        'avr_quantized_grad': avr_quantized_grad/ batches,
                        'avr_grads_plus_error': avr_grads_plus_error,
                        'quantized_error': avr_quantized_error / batches}

                pickle.dump(data, open(filepath + 'Gradient_epoch' + str(epoch) + '.p', "wb"))
                
                break
           
    
        t_loss, t_acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
        
        train_acc[epoch] = np.mean(a0)
        test_acc[epoch] = t_acc
        train_loss[epoch] = np.mean(l0)
        test_loss[epoch] = t_loss
        epoch_time = time.time() - epoch_time
        
        if args.verbose_epoch:
            print("Epoch: " + str(epoch + 1) + '/' + str(epoch_num) + ", Acc:" + str(f'{np.mean(a0): 0.4f}') + ", Val Acc:" + str(f'{t_acc: 0.4f}') + ", Time: %ds" % (epoch_time))
    #------------------------------------------------------------------------------
    "Printing Remaining Statistics"
    elapsed = time.time() - elapsed
    print('Elapsed Time: %.2dD:%.2dH:%.2dM:%.2dS' % (elapsed / 86400, (elapsed / 3600) % 24, (elapsed / 60) % 60, elapsed % 60))
    
    #------------------------------------------------------------------------------
    "Storing Statistics"
    data = {'training_acc': train_acc,
            'test_acc': test_acc,
            'training_loss':train_loss,
            'test_loss':test_loss}
    
    pickle.dump(data, open(filepath + 'Accuracy_and_loss' + '.p', "wb" ))
    
    pickle.dump(cache_dweibull_quantization_table, open(filepath + 'dweibull_4bit_quantization_table' + '.p', "wb" ))
    #------------------------------------------------------------------------------
    "Plotting Results"
    if args.plot_state:
        plt.close('all')
        plt.figure()
        plt.title('Accuracy for ' + args.model_name)
        plt.plot(train_acc, label='Train')
        plt.plot(test_acc, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        
        plt.figure()
        plt.title('Loss for '+ args.model_name)
        plt.plot(train_loss, label='Train')
        plt.plot(test_loss, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
    
if __name__ == '__main__':
    args = parse_args()
    
    run_experiment(args)
