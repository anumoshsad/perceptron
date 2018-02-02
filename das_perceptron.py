#!/usr/bin/python3

import argparse
import numpy as np
#import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--iterations")
parser.add_argument("--lr")
parser.add_argument("--nodev", action = "store_true")
args = parser.parse_args()

# default value
iterations = 10
lr = 1
nodev = False

#print(args)

if args.iterations:
	iterations = int(args.iterations)

if args.lr:
	lr = float(args.lr)

if args.nodev:
	nodev = True

# print(nodev)
# print(iterations)
# print(lr)

def read_file_to_matrix(file_path):
	with open(file_path) as fp:
		lines = fp.read().splitlines()
	X = np.zeros((len(lines), 124))
	Y = np.zeros(len(lines))
	for i,line in enumerate(lines):
		L = line.split()
		Y[i]=int(L[0])
		for j in range(1,len(L)):
			col = int(L[j].split(":")[0])-1   # there are 123 features so my index is 0..122
			X[i,col] = 1
		X[i,123] = 1   # the last column is for bias
	return X, Y


# train_X, train_Y = read_file_to_matrix("../a7a.train")
# dev_X, dev_Y = read_file_to_matrix("../a7a.dev")
# test_X, test_Y = read_file_to_matrix("../a7a.test")

train_X, train_Y = read_file_to_matrix("/u/cs246/data/adult/a7a.train")
dev_X, dev_Y = read_file_to_matrix("/u/cs246/data/adult/a7a.dev")
test_X, test_Y = read_file_to_matrix("/u/cs246/data/adult/a7a.test")

def perceptron(X,Y, iterations, lr=1):
	w = np.zeros(124)
	N = len(X)
	for i in range(iterations):
		for n in range(N):
			if np.dot(w, X[n])*Y[n]<=0:
				w = w + lr*X[n]*Y[n]
	return w


def accuracy(X,Y,w):
	tot = len(X)
	correct=0
	for i in range(tot):
		if np.dot(X[i],w)*Y[i]>0:
			correct+=1
	return correct/tot



# Experimenting with different iterations

def experiment(train_X, train_Y, dev_X, dev_Y, test_X, test_Y, iter = 100, lr = 1):
	iteration_number = []
	training_accuracy = []
	dev_accuracy = []	
	
	w = np.zeros(124)
	N = len(train_X)

	for i in range(1,iter + 1):
		for n in range(N):
			if np.dot(w, train_X[n])*train_Y[n]<=0:
				w = w + lr*train_X[n]*train_Y[n]
		if i==10:
			learned_w = w
		cur_tr_acc = accuracy(train_X, train_Y, w)
		cur_dv_acc = accuracy(test_X, test_Y, w)
		#print("iteration: {}, training_accuracy: {:.2f}%, dev_accuracy: {:.2f}% ".\
		#	format(i, cur_tr_acc*100,cur_dv_acc*100))
		iteration_number.append(i)
		training_accuracy.append(cur_tr_acc)
		dev_accuracy.append(cur_dv_acc)
	
	tr_acc = accuracy(train_X, train_Y, learned_w)
	dev_acc = accuracy(dev_X, dev_Y, learned_w)
	test_acc = accuracy(test_X, test_Y, learned_w)
	print("After 10 iterations,")
	print( "Training accuracy: ", tr_acc)
	print( "Dev accuracy: ", dev_acc)
	print( "Testing accuracy: ", test_acc)
	
	# plt.plot(iteration_number, training_accuracy, label = "Training")
	# plt.plot(iteration_number, dev_accuracy, label = "Dev")
	# plt.xlabel("# of iterations")
	# plt.ylabel("accuracy")
	# plt.title("Accuracy vs iterations")
	# plt.legend()
	# plt.savefig("das_perceptron.png")
	# plt.show()
	

if nodev:
	weights = perceptron(train_X, train_Y, iterations)
	test_accuracy = accuracy(test_X, test_Y, weights)
	print("Test accuracy: ", test_accuracy)
	print("Feature weights (bias last): ",' '.join(map(str, weights)))
else:
	# Expermienting with different itereations. Assuming --nodev and --iterations not givem
	# We run to loop for 100 timese and plot training, dev accuracy. For testing data, we use the 
	# weights we get after 50 iterations
	experiment(train_X, train_Y, dev_X, dev_Y, test_X, test_Y, iter = 100, lr = 1)
























