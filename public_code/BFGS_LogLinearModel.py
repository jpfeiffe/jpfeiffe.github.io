# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# author: Joel Pfeiffer, email: jpfeiffer@purdue.edu
# An example of the simple 0/1 implementation of loglinearmodels, where the scope of the labels/features is limited to just
# 0 or 1.  I used the easy tutorial provided
# by Noah Smith for this example (https://www.cs.cmu.edu/~nasmith/papers/smith.tut04.pdf), and refer
# to a couple of equations there.

from scipy.optimize.optimize import fmin_bfgs
import numpy as np
import copy

# samples some really simple 1/0 feature data for both attributes and labels
class SampleData():
    def __init__(self, n, d):
        means = [[], []]
        means[0] = .01 * np.random.randn(1, d)[0] + .3
        means[1] = .01 * np.random.randn(1, d)[0] + .7

        self.X = np.zeros((n, d), dtype=int)
        self.Y = np.zeros((n, 1), dtype=int)        
        for i in range(n):
        	self.Y[i] = 0
        	if np.random.random() > .5:
        		self.Y[i] = 1

        	for j in range(d):
	            if np.random.random() > means[self.Y[i]][j]:
	            	self.X[i,j] = 1
        

# Log linear model for learning.  Utilizes bfgs
class IID_LogLinearModel:
	def __init__(self, labels, attributes, dim):
		self.labels = copy.deepcopy(labels)
		self.attributes = copy.deepcopy(attributes)
		self.dim = dim
		self.n = labels.shape[0]
		return

	# Given the label and attribute vector, create the corresponding feature vector.
	def featurize_one(self, label, attribute_vec):
		new_features = np.zeros((2*2*self.dim+2))
		offset = label*2*self.dim
		for i,attr in enumerate(attribute_vec):
			index = offset + 2*i + int(attr)
			new_features[index] = 1

		new_features[2*2*self.dim+label]=1
		return new_features

	def featurize_all(self):
		self.features = np.zeros((self.n, 2*2*self.dim+2))
		self.features_0 = np.zeros((self.n, 2*2*self.dim+2))
		self.features_1 = np.zeros((self.n, 2*2*self.dim+2))
		# Create corresponding features for all the label/attr pairs
		for i in range(self.n):
			self.features[i] = self.featurize_one(self.labels[i], self.attributes[i])
			self.features_0[i] = self.featurize_one(0, self.attributes[i])
			self.features_1[i] = self.featurize_one(1, self.attributes[i])


	def log_likelihood(self, betas):
		energy_feats = self.features.dot(betas)
		energy_zeros = np.exp(self.features_0.dot(betas))
		energy_ones = np.exp(self.features_1.dot(betas))
		sums = np.log(np.add(energy_zeros, energy_ones))
		ll = np.sum(energy_feats) - np.sum(sums)
		return ll

	def neg_log_likelihood(self, betas):
		return -1*self.log_likelihood(betas)

	def log_gradient_k(self, k):
		return np.sum(self.features[:,k]) - self.prob_zeros.dot(self.features_0[:,k]) - self.prob_ones.dot(self.features_1[:,k])

	def log_gradient(self, betas):
		energy_zeros = np.exp(self.features_0.dot(betas))
		energy_ones = np.exp(self.features_1.dot(betas))
		partition = np.add(energy_zeros,energy_ones)
		self.prob_zeros = np.divide(energy_zeros, partition)
		self.prob_ones = np.divide(energy_ones, partition)
		gradient = np.asarray([self.log_gradient_k(i) for i in range(2*2*self.dim+2)])
		return gradient

	def neg_log_gradient(self,betas):
		grad = self.log_gradient(betas)
		return -1*grad

	# Simple training algorithm at this point -- create the corresponding features, we want to *minimize* the negative
	# log likelihood, using the negative of the log gradient.  To solve, pick any fmin from scipy.optimize and
	# let it go, giving the corresponding function and gradient.
	def train(self):
		self.featurize_all()
		self.betas = np.random.randn(2*2*self.dim+2)
		self.betas = fmin_bfgs(self.neg_log_likelihood, self.betas, fprime=self.neg_log_gradient)
		print self.betas
		return

	def predict(self, label, attribute_vec):
		feat = np.exp(self.betas.dot(self.featurize_one(label, attribute_vec)))
		feat_0 = np.exp(self.betas.dot(self.featurize_one(0, attribute_vec)))
		feat_1 = np.exp(self.betas.dot(self.featurize_one(1, attribute_vec)))
		return feat / (feat_0 + feat_1)


	def mae(self, labels, attributes):
		errs = np.zeros((self.n,1))
		for i in range(labels.shape[0]):
			errs[i] = self.predict(1-labels[i], attributes[i])

		return np.mean(errs)

if __name__ == "__main__":
	np.random.seed(7)
	dim = 10
	syn_train = SampleData(1000, dim)
	syn_test = SampleData(1000, dim)
	llmodel = IID_LogLinearModel(syn_train.Y, syn_train.X, dim)
	llmodel.train()
	print llmodel.mae(syn_test.Y, syn_test.X)
