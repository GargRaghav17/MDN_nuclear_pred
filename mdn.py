import torch 
from   torch import nn
import mdn

import 	tensorflow as tf
from 	tensorflow import keras
from 	tensorflow.keras import layers

import 	numpy as np
from 	sklearn.model_selection import train_test_split

import 	sys


## ------------------------------------------------------------------------------
# Calculate the average standard deviation, more specifically the root 
# mean-sqaure calculation, to test goodness of fit
# -------------------------------------------------------------------------------
# Input:
	# K 			integer 		number of nuclei used in the prediction
	# delME_FRDM 	matrix			
	# delME_mod 	matrix 			 
# -------------------------------------------------------------------------------
# Output:
# -------------------------------------------------------------------------------
def avg_sd(K, delME_FRDM, delME_mod):

	# calculate the difference between model and FRDM for each nuclei
	# corresponding to an index 
	del_total = (delME_FRDM - delME_mod)**2

	# take the sum of all the differences squared
	sum_2 = del_total.sum()

	# divide the total by the total number of nuclei
	sigma_rms = np.sqrt(sum_2/K)

	print("")
	print("The RMS Standard Deviation is: " + str(sigma_rms) + " MeV")
# -------------------------------------------------------------------------------




# Use a probabilistic neural network (single layer) (MDN)
# The idea is to assume the output data resemebles a mix of different probabiliy 
# distributions, insted of calculating the output node, y, from input vec(x) we
# calculate the parameters of the probability distribution using a neural 
# network. 
# - In this case we need the mean (mu) and variance (sigma) for the Gaussian 
#   distribution.

# To better represent probabilistic data sets than traditional deterministic 
# neural networks, the MDN has the additional benefit of predicting the exact 
# posterior distribution of each predicted value

## ------------------------------------------------------------------------------
class pyt_MixtureDensityNetwork:
## ------------------------------------------------------------------------------

	# --------------------------------------------------------------------------
	# Create a model that maps a vector of inputs to a set of outputs
	# --------------------------------------------------------------------------
	# Input:
		# net_type 		string 		refers to type of network nn or mdn
		# input_num 	integer 	number of features in the input layer
		# output_num 	integer  	number of features in the output layer
		# hid_units 	array 		array corresponing to the # of units per layer
	# --------------------------------------------------------------------------
	# Output:
	# --------------------------------------------------------------------------
	def __init__(self, input_num, output_num, hid_units, num_gaussians):

		# initialize the mixture density model
		# ------------------------------------------------------------------
		# mixture density network with on hidden layer
		self.model = mdn.MDN(input_num, hid_units[0], num_gaussians)
		# -------------------------------------------------------------------

		# Compile the model. Uses the adam optimizer
		# -------------------------------------------------------------------
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)
		# -------------------------------------------------------------------
	# --------------------------------------------------------------------------


	# --------------------------------------------------------------------------
	# Train the model
	# --------------------------------------------------------------------------
	# Input:
		# x_data 		numpy array 	input training data 
		# y_data 		numpy array 	corresponding outputs to the training data
		# num_epochs 	integer			number of epochs (iterations)
		# p 			integer 		patience, controls when model "converges" 
		# 								(no improvement)
		# verbosity 	integer 		0 - no update, 1 - progress bar
	# --------------------------------------------------------------------------
	# Output:
	# --------------------------------------------------------------------------
	def fit(self, x_data, y_data, num_epochs, p, verbosity = 0):

		# split into training and validation set
		self.x_train, self.x_val, self.y_train, self.y_val = \
		train_test_split(x_data, y_data, test_size = 0.2, random_state = 1)

		# in order to get correct input dimensions
		# 	training
		xt_tensor = torch.Tensor(self.x_train)#.unsqueeze(1)
		yt_tensor = torch.Tensor(self.y_train).unsqueeze(1)
		# 	validation
		xv_tensor = torch.Tensor(self.x_val)#.unsqueeze(1)
		yv_tensor = torch.Tensor(self.y_val).unsqueeze(1)

		# training losses list
		self.epochs, self.losses, self.val_losses = [], [], []
		# patience counter used in convergence
		val_loss_min, p_cnt = 100000, 0

		# print('')
		# Train the model
		for epoch in range(num_epochs):
			# training 
			# ------------------------------------
			# Forward pass: Compute predicted variables by passing x to the model
			self.t_pi, self.t_sigma, self.t_mu = self.model(xt_tensor)

			# compute training loss
			loss = mdn.mdn_loss(self.t_pi, self.t_sigma, self.t_mu, yt_tensor)

			# store training loss per epoch for the output plots
			self.epochs.append(epoch)
			self.losses.append(loss.item())
			# ------------------------------------

			# validation
			# ------------------------------------
			self.v_pi, self.v_sigma, self.v_mu = self.model(xv_tensor)

			# compute and store validation loss
			val_loss = mdn.mdn_loss(self.v_pi, self.v_sigma, self.v_mu, yv_tensor)
			self.val_losses.append(val_loss.item())
			# ------------------------------------

			# print training and validation loss
			if verbosity == 1:
				print('epoch: ', epoch,' loss: ', loss.item(), \
					  ' val_loss: ', val_loss.item())

			# convergence 
			# -------------------------------------
			if val_loss.item() < val_loss_min: 
				val_loss_min, p_cnt = val_loss.item(), 0 
			else: 
				p_cnt += 1

			# stop the training if the val_loss hasn't 
			# improved in p consecutive epochs 
			if p_cnt >= p: break
			# -------------------------------------


			# adjust model parameters
			# ------------------------------------
			# Zero gradients, perform a backward pass, and update the weights.
			self.optimizer.zero_grad()

			# perform a backward pass (backpropagation)
			loss.backward()

			# Update the parameters
			self.optimizer.step()
			# ------------------------------------
	# --------------------------------------------------------------------------


	# --------------------------------------------------------------------------
	# Display the model summary
	# --------------------------------------------------------------------------
	# --------------------------------------------------------------------------
	def display(self, input_size):
		from torchsummary import summary
		summary(self.model, (input_size))
	# --------------------------------------------------------------------------


	# --------------------------------------------------------------------------
	# 
	# --------------------------------------------------------------------------
	# Input:
		# x_data	numpy array 		input testing data 
	# --------------------------------------------------------------------------
	# Output:
		# y_data 	numpy array 		predictions using the model  
	# --------------------------------------------------------------------------
	def predict(self, x_data):

		# in order to get correct input dimensions
		num_samples = len(x_data)

		# convert to pytorch tensor
		x = torch.Tensor(x_data).unsqueeze(1)
		# predict data
		pi, sigma, mu = self.model(x)

		# convert back to numpy array
		pi_data = pi.detach().cpu().numpy()
		sigma_data = sigma.detach().cpu().numpy()
		mu_data = mu.data.detach().cpu().numpy()

		return pi_data, sigma_data, mu_data
	# --------------------------------------------------------------------------
## ------------------------------------------------------------------------------
