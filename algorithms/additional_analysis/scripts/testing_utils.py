import numpy as np
import networkx as nx
import pandas as pd
import random
import sys
import matplotlib.pyplot as plt
import analytics_utils as sga

def number_known(node_attribute_dict, unknown_att):
	"""Returns the number of known attributes
	Args:
		node_attribute_dict (dict): {'name': 'category'}
		unknown_att (string): value encoding unknown
	Returns:
		known_att (int): number of known attributes
		known_pos (array): an array specifying positions of known attributes
	"""
	
	known = [0 if v==unknown_att else 1 for v in node_attribute_dict.values()]
	known_pos = np.where(np.array(known)==1)[0]
	return len(known_pos), known_pos

	
def mask_node_attributes(node_attribute_dict, num_to_mask, unknown_att):
	"""Masks attributes for a specified fraction of known nodes
	
	Args:
		node_attribute_dict (dict): {'name': 'category'}
		num_to_mask (int): number of known attributes to mask
		unknown_att (string): value encoding unknown
	
	Returns:
		node_attribute_masked (dict): {'name':'category'} 
		masked_nodes (list) : nodes masked
	"""

	node_attribute_masked = node_attribute_dict.copy()
	nodes = list(node_attribute_dict.keys())
	known_att, known_pos = number_known(node_attribute_dict, unknown_att)
	# print(f'Masking {num_to_mask} out of {known_att} known attributes')
	to_mask = random.sample(list(known_pos), num_to_mask)
	masked_nodes = [nodes[i] for i in to_mask]	
	for n in masked_nodes:
		node_attribute_masked[n] = unknown_att
	return node_attribute_masked, masked_nodes
	

def compare_attributes(node_attribute_orig, node_attribute_predicted, masked_nodes):
	"""Calculates the accuracy of predictions for masked attributes
	
	Args:
		node_attribute_orig (dict): {'name': 'category'}
		node_attribute_predicted (dict): {'name': 'category'}
		masked_nodes (list) : nodes masked
	
	Returns:
		accuracy: fraction of correctly predicted attributes
	"""	

	correct = [node_attribute_orig[n]==node_attribute_predicted[n] for n in masked_nodes]
	return np.array(correct).sum()/float(len(masked_nodes))


def analyze_predictions(node_attribute_orig, node_attribute_distribution, masked_nodes, num_classes):
	"""Calculates the accuracy of predictions for masked attributes
	   by prediction confidence
	
	Args:
		node_attribute_orig (dict): {'name': 'category'}
		node_attribute_distribution (dataframe): distribution over predictions
		masked_nodes (list) : nodes masked
	
	Returns:
		accuracy (dict) : accuracy for each prediction confidence category
	"""	

	confidence = []
	incr = 0.05
	val = round(1/float(num_classes), 1) if round(1/float(num_classes), 1) > 1/float(num_classes) else round(1/float(num_classes), 1) + incr
	confidence.append(val)
	while val < (1-incr):
		val += incr
		confidence.append(val)

	predicted = node_attribute_distribution.idxmax(axis=1)
	max_val = node_attribute_distribution.max(axis=1)

	correct = {}
	count = {}
	result = []
	for c in confidence:
		correct[c] = 0
		count[c] = 0
	for n in masked_nodes:
		for c in reversed(confidence):
			if max_val[n] > c:
				count[c] += 1
				if predicted[n] == node_attribute_orig[n]:
					correct[c] += 1
				break
	for c in confidence:
		result.append(correct[c]/float(count[c]) if count[c]>0 else np.nan)
	return result


def run_tests(
	theta,
	method,
	node_attribute_dict,
	masked_counts,
	unknown_att,
	test_num=10,
	alpha=1,
	norm="KL",
	conv_method="exp",
	max_iter=50
):
	"""Runs tests and reports average accuracy of attribute recovery
	
	Args:
		theta (matrix): partial correlation matrix
		method (str): 'analytical' or 'iterative'
		node_attribute_dict (dict): {'name': 'category'}
		masked_counts (array): number of items to mask in each iteration
		test_num (int): number of tests to run in each category
		masked_nodes (list) : nodes masked
		alpha (float): scaling intensity parameter
		norm (string): normalization for iterative method
		conv_method (string): method used to convert partial correlation matrix
		                      into transition probability matrix 
		max_iter (int): maximum number of iterations
	
	Returns:
		accuracy: fraction of correctly predicted attributes
	"""	

	accuracy = []
	known_att, known_pos = number_known(node_attribute_dict, unknown_att)
	num_classes = len(set(node_attribute_dict.values()))

	print(f'Running the {method} method')

	for i in masked_counts:
		#print(f"Running tests with {i} items out of {known_att} masked")
		results = []
		for j in range(test_num):
			node_attribute_masked, masked_nodes = mask_node_attributes(
				node_attribute_dict, 
				i, 
				unknown_att)
			node_attribute_predicted, node_attribute_distribution = sga.propagate_attribute(
				theta,
				node_attribute_masked,
				unknown_att,
				method=method,
				alpha=alpha,
				norm=norm,
				conv_method=conv_method,
				max_iter=max_iter
			)
			results.append(compare_attributes(
				node_attribute_dict, 
				node_attribute_predicted, 
				masked_nodes
			))
		accuracy.append(np.array(results).mean())
		# print(results)
	return accuracy


def count_predictions(
	node_attribute_dict,
	node_attribute_predicted,
	node_attribute_distribution,
	masked_nodes,
	num_classes
):
	predicted_counts = np.zeros(num_classes)
	original_counts = np.zeros(num_classes)
	attr = np.array(node_attribute_distribution.columns)
	for n in masked_nodes:
		pred = node_attribute_predicted[n]
		# print(pred, attr)
		pred_pos = np.where(attr==pred)[0][0]
		predicted_counts[pred_pos] += 1
		orig = node_attribute_dict[n]
		orig_pos = np.where(attr==orig)[0][0]
		original_counts[orig_pos] +=1
	print("Original: ", original_counts, "Predicted: ", predicted_counts)
	return predicted_counts


def run_tests_confidence(
	theta,
	method,
	node_attribute_dict,
	masked_counts,
	unknown_att,
	test_num=10,
	alpha=1,
	norm=None,
	conv_method = "pos",
	max_iter=50
):
	"""Runs tests and reports average accuracy of attribute recovery
	
	Args:
		theta (matrix): partial correlation matrix
		method (str): 'analytical' or 'iterative'
		node_attribute_dict (dict): {'name': 'category'}
		masked_counts (array): number of items to mask in each iteration
		test_num (int): number of tests to run in each category
		masked_nodes (list) : nodes masked
		alpha (float): scaling intensity parameter 
		norm (string): normalization method for iterative
		conv_method: method for converting partial correlation matrix
		             into transition probability matrix
	
	Returns:
		accuracy: fraction of correctly predicted attributes
	"""	

	accuracy = []
	pred_distrib = []
	known_att, known_pos = number_known(node_attribute_dict, unknown_att)
	num_classes = len(set(node_attribute_dict.values()))

	print(f'Running the {method} method')

	for i in masked_counts:
		#print(f"Running tests with {i} items out of {known_att} masked")
		results = []
		predDistr = []
		for j in range(test_num):
			node_attribute_masked, masked_nodes = mask_node_attributes(
				node_attribute_dict, 
				i, 
				unknown_att)
			node_attribute_predicted, node_attribute_distribution = sga.propagate_attribute(
				theta,
				node_attribute_masked,
				unknown_att,
				method=method,
				alpha=alpha,
				norm=norm,
				conv_method=conv_method,
				max_iter=max_iter
			)
			results.append(analyze_predictions(
				node_attribute_dict, 
				node_attribute_distribution, 
				masked_nodes,
				num_classes
			))
			# predDistr.append(count_predictions(
			# 	node_attribute_dict,
			# 	node_attribute_predicted,
			# 	node_attribute_distribution,
			# 	masked_nodes,
			# 	num_classes
			# ))
		accuracy.append(np.nanmean(np.array(results), axis=0))  
		pred_distrib.append(np.array(predDistr).sum(axis=0))
		# print(results)
	return accuracy, pred_distrib

def plot_accuracy_results(
	accuracy, 
	known_num, 
	masked_counts,
	save_file=None,
	header=''
):
	"""Plot results with masked counts on the X axis and accuracy on Y axis
	Args:
		accuracy (array):  the list of accuracy results for each masked count
		known_num (int): number of known attributes
		masked_counts (array): number of masked attributes in each iteration
		save_file (str): name of the file to save plot to
	"""

	plt.plot(masked_counts, accuracy)	# color='blue', linewidth=3, linestyle='--'
	plt.title(f'{header} Accuracy as a function of the number of known entries masked', fontsize=14)
	plt.xlabel(f'Number of known entries masked (out of the total of {known_num} known)')
	plt.ylabel('Accuarcy')
	plt.show()


def plot_accuracy_by_confidence_results(
	accuracy, 
	known_num, 
	masked_counts,
	num_classes,
	save_file=None
):
	"""Plot results with masked counts on the X axis and accuracy on Y axis
	Args:
		accuracy (array):  the list of accuracy results for each masked count
		known_num (int): number of known attributes
		masked_counts (array): number of masked attributes in each iteration
		save_file (str): name of the file to save plot to
	"""
	# confidence = []
	# incr = 0.05
	# val = round(1/float(num_classes), 1) if round(1/float(num_classes), 1) > 1/float(num_classes) else round(1/float(num_classes), 1) + incr
	# confidence.append(val)
	# while val < (1-incr):
	# 	val += incr
	# 	confidence.append(val)

	# colors = ['darkgreen', 'forestgreen', 'mediumseagreen','limegreen','lime', 'aquamarine', 'turquoise', 'darkturquoise', 'deepskyblue', 'steelblue', 
	#			'navy', 'blue', 'rebeccapurple', 'darkviolet', 'fuchsia', 'deeppink','palevioletred', 'crimson', 'firebrick', 'sienna']
	
	confidence = []
	incr = 0.1
	val = round(1/float(num_classes), 1) if round(1/float(num_classes), 1) > 1/float(num_classes) else round(1/float(num_classes), 1) + incr
	confidence.append(val)
	while val <= (1-incr):
		val += incr
		confidence.append(val)

	colors = ['darkgreen', 'mediumseagreen', 'lime', 'turquoise', 'deepskyblue',  
				'blue', 'darkviolet', 'deeppink','crimson', 'sienna']

	index = 0
	accuracy = np.transpose(np.array(accuracy))

	print(accuracy)

	for c in confidence:
		if c>=0.2:
			plt.plot(masked_counts, accuracy[index], label=round(c, 2), color=colors[index])	# color='blue', linewidth=3, linestyle='--'
		index += 1
	plt.title('Accuracy as a function of the number of known entries masked\nSeparate lines indicate accuracy for different confidence values', fontsize=14)
	plt.xlabel(f'Number of known entries masked (out of the total of {known_num} known)')
	plt.ylabel('Accuarcy')
	plt.legend(bbox_to_anchor=(1,1), loc="upper left")
	plt.show()
