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
# 12/12/14 -- author: Joel Pfeiffer.  jpfeiffer@purdue.edu
# Simple demonstration of the AGM sampler in conjunction with
# the FCL proposal distribution
#
# This is a simple implementation of the paper:
#
# Attributed Graph Models: Modeling network structure with correlated attributes
# Joseph J. Pfeiffer III, Sebastian Moreno, Timothy La Fond, Jennifer Neville and Brian Gallagher 
# In Proceedings of the 23rd International World Wide Web Conference (WWW 2014), 2014 
#
# Sample datasets that can be found in the right format can be 
# downloaded from http://nld.cs.purdue.edu/agm/

import sys, math, random, copy, matplotlib.pyplot as plt


# Just to double check our graph stats look pretty good
def ComputeDegreeDistribution(network):
	degrees = []
	for vertex, neighbors in network.iteritems():
		degrees.append(len(neighbors))

	degrees.sort()

	vals = range(len(degrees))
	vals = map(float, vals)
	vals = map(lambda x: 1 - x / (len(degrees)-1), vals)

	return vals, degrees

# Just computes the pearson correlation
def ComputeLabelCorrelation(network, labels):
	mean1 = 0.0
	mean2 = 0.0
	total = 0.0
	for vertex,neighbors in network.iteritems():
		for neighbor in neighbors:
			mean1 += labels[vertex]
			mean2 += labels[neighbor]
			total += 1

	mean1 /= total
	mean2 /= total
	std1 = 0.0
	std2 = 0.0
	cov = 0.0

	for vertex,neighbors in network.iteritems():
		for neighbor in neighbors:
			std1 += (labels[vertex]-mean1)**2
			std2 += (labels[neighbor]-mean2)**2
			cov += (labels[vertex]-mean1)*(labels[neighbor]-mean2)

	std1 = math.sqrt(std1)
	std2 = math.sqrt(std2)
	return cov / (std1*std2)



# The FCL sampler we'll use for a proposal distribution
class FastChungLu:
	def __init__(self, network):
		self.vertex_list = []
		self.degree_distribution = []
		for vertex, neighbors in network.iteritems():
			self.vertex_list.append(vertex)
			self.degree_distribution.extend([vertex]*len(neighbors))

	def sample_edge(self):
		vertex1 = self.degree_distribution[random.randint(0,len(self.degree_distribution)-1)]
		vertex2 = self.degree_distribution[random.randint(0,len(self.degree_distribution)-1)]

		return vertex1, vertex2

	def sample_graph(self):
		sample_network = {}
		for vertex in self.vertex_list:
			sample_network[vertex] = {}

		ne = 0
		while ne < len(self.degree_distribution):
			v1, v2 = self.sample_edge()
			if v2 not in sample_network[v1]:
				sample_network[v1][v2] = 1
				sample_network[v2][v1] = 1
				ne+=2

		return sample_network

# A simple A/R that creates the following edge features from the corresponding vertex
# attributes.  Namely, if both are 0, if both are 1, and if both are 2.
class SimpleBernoulliAR:
	# Returns 0/0 -> 0, 0/1->1, 1/0->1, 1/1 -> 2
	def edge_var(self, label1, label2):
		return label1*label1 + label2*label2

	# Requires the true network, a complete sampled network from the proposing distribution
	# then the true labels and a random sample of labels
	def learn_ar(self, true_network, sampled_network, true_labels, sample_labels):
		true_counts = {}
		true_probs = {}
		sample_counts = {}
		sample_probs = {}
		self.ratios = {}
		self.acceptance_probs = {}

		# Determine the attribute distribution in the real network
		for vertex, neighbors in true_network.iteritems():
			for neighbor in neighbors:
				var = self.edge_var(true_labels[vertex], true_labels[neighbor])
				if var not in true_counts:
					# put a small (dirichlet) prior
					true_counts[var] = 1.0
				true_counts[var] += 1
		total = sum(true_counts.values())
		for val, count in true_counts.iteritems():
			true_probs[val] = count / total

		# Determine the attribute distribution in the sampled network
		for vertex, neighbors in sampled_network.iteritems():
			for neighbor in neighbors:
				var = self.edge_var(sample_labels[vertex], sample_labels[neighbor])
				if var not in sample_counts:
					# put a small (dirichlet) prior
					sample_counts[var] = 1.0
				sample_counts[var] += 1.0
		total = sum(sample_counts.values())
		for val, count in sample_counts.iteritems():
			sample_probs[val] = count / total


		# Create the ratio between the true values and sampled values
		for val in true_counts.iterkeys():
			self.ratios[val] = true_probs[val] / sample_probs[val]


		# Normalize to figure out the acceptance probabilities
		max_val = max(self.ratios.values())
		for val, ratio in self.ratios.iteritems():
			self.acceptance_probs[val] = ratio / max_val


	def accept_or_reject(self, label1, label2):
		if (random.random() < self.acceptance_probs[self.edge_var(label1, label2)]):
			return True

		return False

# The AGM process.  Overall, most of the work is done in either the edge_acceptor or the proposing distribution
class AGM:
	# Need to keep track of how many edges to sample
	def __init__(self, network):
		self.ne = 0

		for vertex, neighbors in network.iteritems():
			self.ne += len(neighbors)

	# Create a new graph sample
	def sample_graph(self, proposal_distribution, labels, edge_acceptor):
		sample_network = {}
		for vertex in proposal_distribution.vertex_list:
			sample_network[vertex] = {}

		sampled_ne = 0
		while sampled_ne < self.ne:
			v1, v2 = proposal_distribution.sample_edge()

			# The rejection step.  The first part is just making sure the edge doesn't already exist;
			# the second actually does the acceptance/not acceptance.  This requires the edge_accept
			# to have been previously trained
			if v2 not in sample_network[v1] and edge_acceptor.accept_or_reject(labels[v1], labels[v2]):
				sample_network[v1][v2] = 1
				sample_network[v2][v1] = 1
				sampled_ne += 2

		return sample_network



if __name__ == "__main__":
	# data location
	data = 'Amazon/amazon_DVD_7500'
	data_dir = '../Data/'

	# edge representation
	network = {}
	# corresponding labels
	labels = {}

	# readin the edge file
	with open(data_dir + data + '.edges') as edge_file:
		for line in edge_file:
			fields = map(int, line.strip().split('::'))

			# ids
			id0 = fields[0]
			id1 = fields[1]

			# Remove self-loops
			if id0 == id1:
				continue

			# Check/create new vertices
			if id0 not in network:
				network[id0] = {}
			if id1 not in network:
				network[id1] = {}

			network[id0][id1] = 1
			network[id1][id0] = 1


	# readin the label file
	with open(data_dir + data + '.lab') as label_file:
		for line in label_file:
			fields = map(int, line.strip().split('::'))

			# values
			id = fields[0]
			lab = fields[1]

			# only include items with edges
			if id not in network:
				continue

			# assign the labels
			labels[id] = lab


	print 'Initial Graph Correlation', ComputeLabelCorrelation(network,labels)
	fcl = FastChungLu(network)
	fcl_sample = fcl.sample_graph()

	# Random permutation of labels.  This is shorter code than sampling bernoullis for all,
	# and can be replaced if particular labels should only exist with some guaranteed probability
	# for (e.g.) privacy
	sample_labels_keys = copy.deepcopy(labels.keys())
	sample_labels_items = copy.deepcopy(labels.values())
	random.shuffle(sample_labels_items)
	sample_labels = dict(zip(sample_labels_keys, sample_labels_items))


	# Double check that the FCL correlation is negligible (if this is not near 0 there's something wrong)
	print 'FCL Graph Correlation', ComputeLabelCorrelation(fcl_sample, sample_labels)

	# Now for the AGM steps.  First, just create the AR method using the given data, the proposing distribution,
	# and the random sample of neighbors.
	edge_acceptor = SimpleBernoulliAR()
	edge_acceptor.learn_ar(network, fcl_sample, labels, sample_labels)

	# Now we actually do AGM!  Just plug in your proposing distribution (FCL Example Given) as well as
	# the edge_acceptor, and let it go!
	agm = AGM(network)
	agm_sample = agm.sample_graph(fcl, sample_labels, edge_acceptor)

	# This should be fairly close to the initial graph's correlation
	print 'AGM Graph Correlation', ComputeLabelCorrelation(agm_sample, sample_labels)

	xs, ys = ComputeDegreeDistribution(network)
	plt.plot(xs, ys, label='Original')
	xs, ys = ComputeDegreeDistribution(fcl_sample)
	plt.plot(xs, ys, label='FCL')
	xs, ys = ComputeDegreeDistribution(agm_sample)
	plt.plot(xs, ys, label='AGM-FCL')
	plt.legend()
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('Degree')
	plt.ylabel('CCDF')
	plt.show()

