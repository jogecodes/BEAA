import pybnesian as pbn
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import pyarrow
import sys

def learn_net(learn_data, structure_algorithm, net_type = 'gaussian'):
  if structure_algorithm == 'GHC':
    if net_type == 'gaussian':
      net_bn = pbn.hc(learn_data, bn_type=pbn.GaussianNetworkType())
    elif net_type == 'kde':
      net_bn = pbn.hc(learn_data, bn_type=pbn.KDENetworkType())
  if structure_algorithm == 'PC':
    learn_algo = pbn.PC()
    hypothesis = pbn.LinearCorrelation(learn_data)
    disc_graph = learn_algo.estimate(hypot_test = hypothesis, allow_bidirected = False)
    disc_dag = disc_graph.to_approximate_dag()
    if net_type == 'gaussian':
      net_bn = pbn.GaussianNetwork(disc_dag)
    elif net_type == 'kde':
      net_bn = pbn.KDENetwork(disc_dag)
  else:
    print('Invalid structure algotithm!')
    return None
  net_bn.fit(learn_data)
  return net_bn

def create_initial_cpds(base_nodes, noise_nodes, noise_mean, noise_variance, torus = False):
  init_cpds = []
  zero = sys.float_info.min
  # Adding only random normal noise arcs
  for node in base_nodes:
    noise_parents = noise_nodes
    parents = noise_parents
    random_betas = []
    for i in range(len(noise_parents)):
      multiplier = 1
      if torus == True:
        randnum = np.random.uniform(0,1,1)
        if randnum > 0.5:
          multiplier = -1
      random_betas.append(multiplier * np.random.normal(noise_mean,noise_variance))
    betas = [0] + random_betas
    variance = zero
    init_cpds.append(pbn.LinearGaussianCPD(node, parents, betas, variance))
  return init_cpds

def reset_interface_net(int_nodes, base_nodes, new_arcs, net_type = 'gaussian'):
  if net_type == 'gaussian':
    net_bn = pbn.ConditionalGaussianNetwork(interface_nodes = int_nodes, nodes = base_nodes, arcs = new_arcs)
  elif net_type == 'kde':
    net_bn = pbn.ConditionalKDENetwork(interface_nodes = int_nodes, nodes = base_nodes, arcs = new_arcs)
  return net_bn

def learn_condnet(learn_data, interface_nodes, nodes, forced_arcs = [], banned_arcs = []):
  learn_algo = pbn.PC()
  hypothesis = pbn.LinearCorrelation(learn_data)
  net_graph = learn_algo.estimate_conditional(hypot_test = hypothesis, nodes = nodes, interface_nodes = interface_nodes, allow_bidirected = False, arc_whitelist = forced_arcs, arc_blacklist = banned_arcs)
  net_dag = net_graph.to_approximate_dag()
  net_bn = pbn.ConditionalGaussianNetwork(net_dag)
  net_bn.fit(learn_data)
  return net_bn

def learn_condfromnet(learn_data, interface_nodes, nodes, indep_test = 'LC', approximate = True, forced_arcs = [], banned_arcs = [], banned_edges = [], net_type = 'gaussian'):
  learn_algo = pbn.PC()
  if indep_test == 'LC':
    hypothesis = pbn.LinearCorrelation(learn_data)
  if indep_test == 'MI':
    hypothesis = pbn.MutualInformation(learn_data)
  for nnode in interface_nodes:
    for bnode in interface_nodes:
      banned_edges.append((nnode,bnode))
  net_graph = learn_algo.estimate(hypot_test = hypothesis, allow_bidirected = False, arc_whitelist = forced_arcs, arc_blacklist = banned_arcs, edge_blacklist = banned_edges)
  net_cond = net_graph.conditional_graph(nodes = nodes, interface_nodes = interface_nodes)
  if(approximate):
    net_dag = net_cond.to_dag()
  else:
    net_dag = net_cond.to_approximate_dag()
  if net_type == 'gaussian':
    net_bn = pbn.ConditionalGaussianNetwork(net_dag)
  elif net_type == 'kde':
    net_bn = pbn.ConditionalKDENetwork(net_dag)
  net_bn.fit(learn_data)
  return net_bn

# Generate samples 
def gen_samples(gen_bn, sample_size, sample_mean, sample_variance, torus = False):
  evidence = {}
  for node in gen_bn.interface_nodes():
    multiplier = 1
    if torus == True:
      randnum = np.random.uniform(0,1,1)
      if randnum > 0.5:
        multiplier = -1
    evidence[node] = multiplier*np.random.normal(sample_mean, sample_variance, sample_size)
  sample = gen_bn.sample(evidence = pd.DataFrame(evidence), concat_evidence = True, ordered = True)
  fake_sample = sample.to_pandas()
  return(fake_sample)

def gen_samples_genetically(gen_bn, disc_bn, generation_size, initial_sample_mean, initial_sample_variance, generations, p_cross, best_frac, torus = False):
  evidence = {}
  size_multiplier = round(1/best_frac)
  for node in gen_bn.interface_nodes():
    multiplier = 1
    if torus == True:
      randnum = np.random.uniform(0,1,1)
      if randnum > 0.5:
        multiplier = -1
    evidence[node] = multiplier*np.random.normal(initial_sample_mean, initial_sample_variance, size_multiplier*generation_size)
  sample = gen_bn.sample(evidence = pd.DataFrame(evidence), concat_evidence = True, ordered = True)
  initial_sample = sample.to_pandas()
  initial_sample['logl'] = disc_bn.logl(initial_sample)
  for i in range(generations):
    initial_sample.sort_values('logl', axis=0, ascending = False, inplace = True, ignore_index = True)
    selected_sample = initial_sample.iloc[0:generation_size]
    modified_sample = selected_sample.copy()
    sample_columns = selected_sample.columns
    for index in range(len(selected_sample)):
      randnum = np.random.uniform(0,1,1)
      if randnum > p_cross:
        randindex = np.random.randint(0, len(sample_columns)-1)
        crossindex = np.random.randint(0, len(selected_sample)-1)
        while crossindex==index:
          crossindex = np.random.randint(0, len(selected_sample)-1)
        modified_sample.loc[index, sample_columns[0:randindex]] = selected_sample.loc[crossindex, sample_columns[0:randindex]]
    for node in gen_bn.interface_nodes():
      multiplier = 1
      if torus == True:
        randnum = np.random.uniform(0,1,1)
        if randnum > 0.5:
          multiplier = -1
      evidence[node] = multiplier*np.random.normal(initial_sample_mean, initial_sample_variance, (size_multiplier-1)*generation_size)
    filling_sample = gen_bn.sample(evidence = pd.DataFrame(evidence), concat_evidence = True, ordered = True)
    filling_sample = filling_sample.to_pandas()
    initial_sample = pd.concat([modified_sample, filling_sample], ignore_index = True)
  initial_sample.sort_values('logl', axis=0, ascending = False, inplace = True, ignore_index = True)
  selected_sample = initial_sample.iloc[0:generation_size]
  # print(selected_sample)
  return selected_sample

def gen_samples_gradient(gen_bn, disc_bn, generation_size, initial_sample_mean, initial_sample_variance, iterations, learning_rate = 0.001, delta = 0.01, tol = 0.01, max_gradient_iter = 100000, torus = False):
  starting_evidence = {}
  for node in gen_bn.interface_nodes():
    multiplier = 1
    if torus == True:
      randnum = np.random.uniform(0,1,1)
      if randnum > 0.5:
        multiplier = -1
    starting_evidence[node] = multiplier*np.random.normal(initial_sample_mean, initial_sample_variance, generation_size)
  starting_evidence = pd.DataFrame(starting_evidence)
  starting_sample = gen_bn.sample(evidence = pd.DataFrame(starting_evidence), concat_evidence = True, ordered = True).to_pandas()
  starting_sample['logl'] = disc_bn.logl(starting_sample)
  print('Logl starting mean: '+str(starting_sample['logl'].mean(axis=0)))
  for i in range(iterations):
    for column in gen_bn.interface_nodes():
      value = starting_sample[column]
      top_value = value*(1+delta)
      min_value = value*(1-delta)
      diff_value = top_value - min_value
      top_evidence = starting_evidence.copy()
      top_evidence[column] = top_value
      top_sample = gen_bn.sample(evidence = top_evidence, concat_evidence = True, ordered = True).to_pandas()
      top_sample['logl'] = disc_bn.logl(top_sample)
      # print('top_sample')
      # print(top_sample)
      min_evidence = starting_evidence.copy()
      min_evidence[column] = min_value
      min_sample = gen_bn.sample(evidence = min_evidence, concat_evidence = True, ordered = True).to_pandas()
      min_sample['logl'] = disc_bn.logl(min_sample)
      # print('min_sample')
      # print(min_sample)
      gradient = (top_sample['logl'] - min_sample['logl']).divide(diff_value)
      iterations = 0
      while ((abs(gradient) > tol).any() and (iterations < max_gradient_iter)):
        # print('Iteration '+str(i+1)+', column '+column+', repetition '+str(iterations+1)+':')
        iterations = iterations + 1
        starting_evidence[column] = starting_evidence[column] + learning_rate*gradient
        # print(gradient)
        starting_sample = gen_bn.sample(evidence = pd.DataFrame(starting_evidence), concat_evidence = True, ordered = True).to_pandas()
        starting_sample['logl'] = disc_bn.logl(starting_sample)
        # print('starting_sample')
        # print(starting_sample)
        value = starting_evidence[column]
        top_value = value*(1+delta)
        min_value = value*(1-delta)
        diff_value = top_value - min_value
        top_evidence = starting_evidence.copy()
        top_evidence[column] = top_value
        top_sample = gen_bn.sample(evidence = pd.DataFrame(top_evidence), concat_evidence = True, ordered = True).to_pandas()
        top_sample['logl'] = disc_bn.logl(top_sample)
        # print('top_sample')
        # print(top_sample)
        min_evidence = starting_evidence.copy()
        min_evidence[column] = min_value
        min_sample = gen_bn.sample(evidence = pd.DataFrame(min_evidence), concat_evidence = True, ordered = True).to_pandas()
        min_sample['logl'] = disc_bn.logl(min_sample)
        # print('min_sample')
        # print(min_sample)
        gradient = (top_sample['logl'] - min_sample['logl']).divide(diff_value)
    starting_evidence[column] = starting_evidence[column] - learning_rate*gradient
  final_sample = gen_bn.sample(evidence = pd.DataFrame(starting_evidence), concat_evidence = True, ordered = True).to_pandas()
  final_sample['logl'] = disc_bn.logl(final_sample)
  print('Logl ending mean: '+str(final_sample['logl'].mean(axis=0)))
  return final_sample

# Noise module calculator (anomaly score function)
def ano_score(row, noise_nodes, power = 2):
  total = 0
  for node in noise_nodes:
    node_row = row[node]
    if type(node_row) == np.float64 or type(node_row) == np.float:
      ano_object = node_row
    elif type(node_row) == pyarrow.lib.DoubleArray:
      ano_object = node_row.to_numpy[0]
    else:
      print('AHHHHHH')
      print(type(node_row))
    total = total + pow(abs(ano_object), power)
  return total

def euclidean_mod(row, nodes):
  total = 0
  for node in nodes:
    total = total + pow(abs(row[node]), 2)
  return total

def find_parents_from_sample(network, sample, sample_mean, sample_variance, generation_size = 100, num_iter = 100, p_cross = 0.7, cross_frac = 0.5):
  interface_nodes = network.interface_nodes()
  leaf_nodes = network.nodes()
  initial_evidence = {}
  for node in interface_nodes:
    initial_evidence[node] = np.random.normal(sample_mean, sample_variance, generation_size)
  chromosomes = pd.DataFrame(initial_evidence)
  for node in leaf_nodes:
    chromosomes.loc[:,node] = sample[node]
  num_crossed = round(len(chromosomes)*cross_frac)
  for i in range(num_iter):
    chromosomes['logl'] = network.logl(chromosomes)
    chromosomes.sort_values('logl', axis=0, ascending = False, inplace = True, ignore_index = True)
    for index in range(num_crossed):
      mod_chromosomes = chromosomes.copy()
      randnum = np.random.uniform(0,1,1)
      if randnum > p_cross:
        randindex = np.random.randint(0, len(interface_nodes)-1)
        crossing_nodes = interface_nodes[0:randindex]
        crossindex = np.random.randint(0, len(chromosomes)-1)
        while crossindex==index:
          crossindex = np.random.randint(0, len(chromosomes)-1)
        mod_chromosomes.loc[index, crossing_nodes] = chromosomes.loc[crossindex, crossing_nodes]
    evidence = {}
    for node in interface_nodes:
      evidence[node] = np.random.normal(sample_mean, sample_variance, generation_size-num_crossed)
    filling_sample = pd.DataFrame(evidence)
    chromosomes = pd.concat([mod_chromosomes, filling_sample], ignore_index = True)
  chromosomes.sort_values('logl', axis=0, ascending = False, inplace = True, ignore_index = True)
  selected_sample = chromosomes.iloc[0]
  return selected_sample

def dissect_cpd(string_cpd):
  scpd = str(string_cpd)
  scpd.strip()
  scpd = ''.join(scpd.split())
  leftside = scpd.split('=')[0]
  rightside = scpd.split('=')[1]
  attributes = leftside[leftside.find('(')+1:leftside.find(')')]

  if attributes.find('|') == -1:
      attribute_node = attributes
      parents = []
  else:
      attribute_node = attributes.split('|')[0]
      parents = attributes.split('|')[1]
      parents = parents.split(',')
  distr = rightside[rightside.find('(')+1:rightside.find(')')]

  elements = distr.split(',')
  betas = elements[:1]
  if betas[0].find('+') != -1:
      betas = betas[0].split('+')
      for i in range(len(betas)):
          if betas[i].find('*')!=-1:
              betas[i]=float(betas[i].split('*')[0])
          else:
              betas[i]=float(betas[i])
  else:
      betas[0] = float(betas[0])
  variance = elements[-1]
  result = {
      'attribute' : attribute_node,
      'parents' : parents,
      'betas' : betas,
      'variance' : float(variance)
  }
  return result

def add_noise_to_cpds(cpds, existing_noise_nodes, add_noise_nodes, noise_mean, noise_variance, torus = False):
  zero = sys.float_info.min
  starting_noise = len(existing_noise_nodes)+1
  new_noise = []
  for i in range(add_noise_nodes):
      new_noise.append('noise'+str(i+starting_noise))
  new_cpds = []
  for cpd in cpds:
      dissected_cpd = dissect_cpd(cpd)
      node = dissected_cpd['attribute']
      parents = dissected_cpd['parents']
      betas = dissected_cpd['betas']
      variance = dissected_cpd['variance']
      if variance < zero:
          variance = zero
      for new_parent in new_noise:
              parents.append(new_parent)
              multiplier = 1
              if torus == True:
                  randnum = np.random.uniform(0,1,1)
                  if randnum > 0.5:
                      multiplier = -1
              betas.append(multiplier * np.random.normal(noise_mean,noise_variance))
      new_cpds.append(pbn.LinearGaussianCPD(node, parents, betas, variance))
  return new_cpds