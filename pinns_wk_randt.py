''' PINNs codes for Windkessel parameter estimation

Author: Jeremias Garay Labra (jeremias.garay.l@gmail.com)
Date:   2022-10-13

'''

import warnings
warnings.filterwarnings("ignore")
import os, sys
import ruamel.yaml as yaml
import logging
from pathlib import Path
import time, shutil
import numpy
from scipy.interpolate import interp1d
import torch
import pickle
torch.cuda.empty_cache()

import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import dolfin


class OwnScheduler():
    ''' own scheduler class for more control over reducing learning rate'''
    def __init__(self, optimizer, logging, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, given_epochs = [], threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False):
        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor        
        self.optimizer = optimizer

        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.logging = logging
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.given_epochs = []
        if len(given_epochs) > 0:
            self.given_epochs = given_epochs
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self.lr_track = []
        self.reduced_epochs = []
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()
    
    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)

        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience or (epoch in self.given_epochs):
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        # tracking info
        for param_group in self.optimizer.param_groups:
            self.lr_track.append(float(param_group['lr']))
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                self.reduced_epochs.append(epoch)
                if self.verbose:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                 "%.5d") % epoch
                    self.logging.info('Epoch {}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch_str, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = torch.inf
        else:  # mode == 'max':
            self.mode_worse = -torch.inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def save_info(self, outpath):
        with open(outpath + 'learning_rates.pickle', 'wb') as handle:
            pickle.dump(self.lr_track, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(outpath + 'reduced_epochs.pickle', 'wb') as handle:
            pickle.dump(self.reduced_epochs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        self.logging.info('saving learning rates')
        self.logging.info('Reduced Epochs = {}'.format(self.reduced_epochs))

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)

def save_inputfile(options, inputfile):
    ''' Copy input file of to output directory.

    Args:
        options (dict):   Options dictionary (from YAML input file)
        inputfile (str):  Path of input file to be copied
    '''

    outpath = options['io']['write_path']
    path = Path(outpath)
    path.mkdir(parents=True, exist_ok=True)
    path = path.joinpath('input.yaml')
    shutil.copy2(str(inputfile), str(path))

def ComputeNormals(DATA, windkessel_bnd):
    
    normal_lst = {}
    for k in windkessel_bnd:
        x1 = DATA['xwk'][k][0][0]
        y1 = DATA['ywk'][k][0][0]
        z1 = DATA['zwk'][k][0][0]
        x2 = DATA['xwk'][k][1][0]
        y2 = DATA['ywk'][k][1][0]
        z2 = DATA['zwk'][k][1][0]
        x3 = DATA['xwk'][k][2][0]
        y3 = DATA['ywk'][k][2][0]
        z3 = DATA['zwk'][k][2][0]
        

        v1 = [x2-x1, y2-y1, z2-z1]
        v2 = [x3-x1, y3-y1, z3-z1]
        n = numpy.cross(v2,v1)
        normal_lst[k] = n/(n[0]**2 + n[1]**2 + n[2]**2)**(0.5)
        

    return normal_lst

def LoadMesh(options):

    path_to_mesh = options['data']['mesh'] 
    mesh = dolfin.Mesh()
    hdf = dolfin.HDF5File(mesh.mpi_comm(), path_to_mesh, 'r')
    hdf.read(mesh, '/mesh', False)
    boundaries = dolfin.MeshFunction('size_t' , mesh, mesh.topology().dim() - 1)
    hdf.read(boundaries, '/boundaries')
    hdf.close()
    V1 = dolfin.VectorElement('P' , mesh.ufl_cell(), 1)
    P1 = dolfin.FiniteElement('P' , mesh.ufl_cell(), 1)
    W = dolfin.FunctionSpace(mesh, V1*P1)
    V = W.sub(0).collapse()
    Q = W.sub(1).collapse()

    u = dolfin.Function(V)
    u.rename('u', 'velocity')
    p = dolfin.Function(Q)
    p.rename('p', 'pressure')
    ones = dolfin.interpolate(dolfin.Constant(1), Q)

    
    return [u, p, mesh, boundaries , ones]

def DataReader(options):
    
    DATA = numpy.load( options['data']['input_path'] + 'data.npz', allow_pickle=True)
    CoPoints = numpy.load( options['data']['input_path'] + 'copoints.npz', allow_pickle=True)
    Factors = numpy.load( options['data']['input_path'] + 'normfactors.npz', allow_pickle=True)
    logging.info('Reading data from {}'.format(options['data']['input_path']))
    data_dict = DATA['arr_0'].item()
    copoints_dict = CoPoints['arr_0'].item()
    factors_dict = Factors['arr_0'].item()
    

    return data_dict, copoints_dict, factors_dict

def SetHyperparameters(options):
    
    HP_dict = {}

    HP_dict['device'] = torch.device("cuda")
    
    HP_dict['hidden_layers'] = options['PINNs']['hidden_layers']
    HP_dict['hidden_layers_pi'] = options['PINNs']['hidden_layers_pi']

    HP_dict['inverse_iter_rate'] = options['windkessel']['inverse_problem']['iter_rate']
    
    HP_dict['activation_function'] = options['PINNs']['activation_function']
    HP_dict['activation_function_pi'] = options['PINNs']['activation_function_pi']
    
    HP_dict['batchsize'] = options['PINNs']['batchsize']
    HP_dict['learning_rates'] = options['PINNs']['learning_rates']

    HP_dict['epochs'] = options['PINNs']['epochs']
    HP_dict['neurons_per_layer'] = options['PINNs']['neurons_per_layer'] # neurons per layer
    HP_dict['neurons_per_layer_pi'] = options['PINNs']['neurons_per_layer_pi']
    
    
    HP_dict['lambdas'] = {
        'phys': options['PINNs']['lambdas']['manual_values']['phys'],
        'data': options['PINNs']['lambdas']['manual_values']['data'],
        'BC': options['PINNs']['lambdas']['manual_values']['BC'],
        'windkessel': options['PINNs']['lambdas']['manual_values']['windkessel'],
        'pmean': options['PINNs']['lambdas']['manual_values']['pmean'],
        'initial': options['PINNs']['lambdas']['manual_values']['initial'],
        'gradp': options['PINNs']['lambdas']['manual_values']['gradp'],
    }

    return HP_dict 

def ComputeVelocityVector(out_ux, out_uy, out_uz, u_length, Factors):

    utot_vec = numpy.zeros(u_length)
    utot_vec[0::3] = out_ux[:,0]*Factors['U']
    utot_vec[1::3] = out_uy[:,0]*Factors['U']
    utot_vec[2::3] = out_uz[:,0]*Factors['U']

    return utot_vec

def ComputePressureVector(out_p, Factors):

    out_p = out_p.cpu().data.numpy() # converting to cpu
    p_current = out_p[:,0]*Factors['rho']*Factors['U']**2

    return p_current

def MeanWKPressure(pwk):
    pwk = pwk.cpu().data.numpy()

    return numpy.average(pwk[:,0])

def ComputeVelocityNormal(out_ux, out_uy, out_uz, normals):

    u_normals = {}
    out_ux = out_ux.cpu().data.numpy() # converting to cpu
    out_uy = out_uy.cpu().data.numpy() # converting to cpu
    out_uz = out_uz.cpu().data.numpy() # converting to cpu
    ux_current = out_ux[:,0]
    uy_current = out_uy[:,0]
    uz_current = out_uz[:,0]
    
    for k in normals.keys():
        u_normals[k] = ux_current*normals[k][0] + uy_current*normals[k][1] + uz_current*normals[k][2] 

    return u_normals

def ComputeParamsError(Rd_i, Rd_ref, C_i, C_ref, Rp_i, Rp_ref):
    
    ratio = 0
    L = sum([len(Rd_i),len(C_i),len(Rp_i)])

    for k in Rd_i.keys():
        param_k = Rd_i[k]().item()
        param_k_ref = Rd_ref[k]
        ratio += (1/L)*(param_k-param_k_ref)**2/param_k_ref**2
    
    for k in C_i.keys():
        param_k = C_i[k]().item()
        param_k_ref = C_ref[k]
        ratio += (1/L)*(param_k-param_k_ref)**2/param_k_ref**2
    
    for k in Rp_i.keys():
        param_k = Rp_i[k]().item()
        param_k_ref = Rp_ref[k]
        ratio += (1/L)*(param_k-param_k_ref)**2/param_k_ref**2
    
    return numpy.sqrt(ratio)

def PrintEstimation(inverse_problem, Rd_i, Rd_ref, C_i, C_ref, Rp_i, Rp_ref, param_track=None, logscale=False):
    if inverse_problem:
        for k in Rd_ref.keys():
            if (k in Rd_i.keys()) and (k in Rp_i.keys()) and (k in C_i.keys()):
                Rdk = Rd_i[k].R.item()
                Rpk = Rp_i[k].R.item()
                Ck = C_i[k].C.item()
                if logscale:
                    Rdk = numpy.exp(Rdk)
                    Rpk = numpy.exp(Rpk)
                    Ck = numpy.exp(Ck)
                logging.info('bnd: {} \t Rp:{:.2f}/{:.2f} | Rd:{:.2f}/{:.2f} | C:{:.2f}/{:.2f}'.format(k,numpy.round(Rpk,2), numpy.round(Rp_ref[k],2), 
                            numpy.round(Rdk,2), numpy.round(Rd_ref[k],2), numpy.round(Ck,2), numpy.round(C_ref[k],2) ))
                param_track[k]['Rd'].append(Rdk)
                param_track[k]['Rp'].append(Rpk)
                param_track[k]['C'].append(Ck)
            elif (k in Rd_i.keys()) and (k in Rp_i.keys()) and (k not in C_i.keys()):
                Rdk = Rd_i[k].R.item()
                Rpk = Rp_i[k].R.item()
                if logscale:
                    Rdk = numpy.exp(Rdk)
                    Rpk = numpy.exp(Rpk)
                logging.info('bnd: {} \t Rp:{:.2f}/{:.2f} | Rd:{:.2f}/{:.2f} | C:{:.2f}'.format(k,numpy.round(Rpk,2), numpy.round(Rp_ref[k],2), 
                            numpy.round(Rdk,2), numpy.round(Rd_ref[k],2), numpy.round(C_ref[k],2) ))
                param_track[k]['Rd'].append(Rdk)
                param_track[k]['Rp'].append(Rpk)
            elif (k not in Rd_i.keys()) and (k in Rp_i.keys()) and (k not in C_i.keys()):
                Rpk = Rp_i[k].R.item()
                if logscale:
                    Rpk = numpy.exp(Rpk)
                logging.info('bnd: {} \t Rp:{:.2f}/{:.2f} | Rd:{:.2f} | C:{:.2f}'.format(k,numpy.round(Rpk,2), numpy.round(Rp_ref[k],2), 
                            numpy.round(Rd_ref[k],2), numpy.round(C_ref[k],2) ))
                param_track[k]['Rp'].append(Rpk)
            elif (k not in Rd_i.keys()) and (k not in Rp_i.keys()) and (k in C_i.keys()):
                Ck = C_i[k].C.item()
                if logscale:
                    Ck = numpy.exp(Ck)
                logging.info('bnd: {} \t Rp:{:.2f} | Rd:{:.2f} | C:{:.2f}/{:.2f}'.format(k, numpy.round(Rp_ref[k],2), 
                            numpy.round(Rd_ref[k],2), numpy.round(Ck,2) , numpy.round(C_ref[k],2) ))
                param_track[k]['C'].append(Ck)
            elif k in Rd_i.keys() and (k not in Rp_i.keys()) and (k not in C_i.keys()):
                Rdk = Rd_i[k].R.item()
                if logscale:
                    Rdk = numpy.exp(Rdk)
                logging.info('bnd: {} \t Rp:{:.2f} | Rd:{:.2f}/{:.2f} | C:{:.2f}'.format(k,numpy.round(Rp_ref[k],2), 
                            numpy.round(Rdk,2), numpy.round(Rd_ref[k],2), numpy.round(C_ref[k],2) ))
                param_track[k]['Rd'].append(Rdk)
            elif (k in Rd_i.keys()) and (k not in Rp_i.keys()) and (k in C_i.keys()):
                Rdk = Rd_i[k].R.item()
                Ck = C_i[k].C.item()
                if logscale:
                    Rdk = numpy.exp(Rdk)
                    Ck = numpy.exp(Ck)
                logging.info('bnd: {} \t Rp:{:.2f} | Rd:{:.2f}/{:.2f} | C:{:.2f}/{:.2f}'.format(k,numpy.round(Rp_ref[k],2), 
                            numpy.round(Rdk,2), numpy.round(Rd_ref[k],2), numpy.round(Ck,2) , numpy.round(C_ref[k],2) ))
                param_track[k]['Rd'].append(Rdk)
                param_track[k]['C'].append(Ck)
            elif (k not in Rd_i.keys()) and (k in Rp_i.keys()) and (k in C_i.keys()):
                Rpk = Rp_i[k].R.item()
                Ck = C_i[k].C.item()
                if logscale:
                    Rpk = numpy.exp(Rpk)
                    Ck = numpy.exp(Ck)
                logging.info('bnd: {} \t Rp:{:.2f}/{:.2f} | Rd:{:.2f} | C:{:.2f}/{:.2f}'.format(k,numpy.round(Rpk,2), numpy.round(Rp_ref[k],2), 
                            numpy.round(Rd_ref[k],2), numpy.round(Ck,2), numpy.round(C_ref[k],2) ))
                param_track[k]['Rp'].append(Rpk)
                param_track[k]['C'].append(Ck)
    else:
        for k in Rd_ref.keys():
            logging.info('bnd: {} \t Rp:{} | Rd:{} | C:{}'.format(k,numpy.round(Rp_ref[k],2),numpy.round(Rd_ref[k],2),numpy.round(C_ref[k],2)))

def SaveGradients(net,dict):

    full_param = numpy.array([])

    for i,key in enumerate(dict.keys()):
        if isinstance(net.main[i],nn.Linear):
            wg = net.main[i].weight.grad.cpu().data.numpy()
            bg = net.main[i].bias.grad.cpu().data.numpy()
            dict[key]['weight'] = wg.reshape(-1)
            dict[key]['bias'] = bg.reshape(-1)
            full_param = numpy.concatenate((full_param,dict[key]['weight'],dict[key]['bias']))

    return [numpy.average(abs(full_param)), max(abs(full_param)), numpy.linalg.norm(abs(full_param))]

def PrintLambdaAnnealing(dict):
    if 'pmean' in dict.keys():
        logging.info('Lambda Annealing: \n \u03BB_phys: {:2f} \u03BB_data: {:2f} \u03BB_bc: {:2f} \u03BB_wk: {:2f} \
                \n \u03BB_init: {:2f} \u03BB_pmean: {:2f} \u03BB_gradp: {:2f}'.format(dict['phys'],dict['data'],dict['BC'],dict['windkessel'],
                dict['initial'],dict['pmean'],dict['gradp']))
    else:
        logging.info('Lambda Annealing: \n \u03BB_phys: {:2f} \u03BB_data: {:2f} \u03BB_bc: {:2f} \u03BB_wk: {:2f} \
                \n \u03BB_init: {:2f} \u03BB_gradp: {:2f}'.format(dict['phys'],dict['data'],dict['BC'],dict['windkessel'],
                dict['initial'],dict['gradp']))
    logging.info('-------------------------')

def PINNs(HP, DATA, Factors, CoPoints, options, fenics = None):
    ''' Physics Inform Neuronal Network

    Args:
        HP (dict):      Hiperparameters of the NN
        DATA (dict):    Data dictionary
        CoPoints (dict):    Collocation points
        u (FEniCS func):    optional for visualization
    '''


    # if tracking gradients
    is_tracking_gradients = False
    is_lambda_annealing = options['PINNs']['lambdas']['annealing']['apply']
    tracking_gradients_dict = {}
    gradients_options = ['phys','data','wk','bc','initial','pmean','gradp']
    # if lambda annealing, tracking gradients is set to True
    if is_lambda_annealing:
        is_tracking_gradients = True
        for trm in gradients_options:
            tracking_gradients_dict[trm] = {}
            if trm == 'initial':
                for hl in range(HP['hidden_layers_pi']+1):
                    tracking_gradients_dict[trm]['layer{}'.format(hl)] = {'weight': [], 'bias': []}
            else:
                for hl in range(HP['hidden_layers']+1):
                    tracking_gradients_dict[trm]['layer{}'.format(hl)] = {'weight': [], 'bias': []}


    arch_type = options['PINNs']['architecture']
    if arch_type == '01':
        logging.info('Using Architecture 01: NN(ux) + NN(uy) + NN(uz) + NN(p)')
    elif arch_type == '02':
        logging.info('Using Architecture 02: NN(ux, uy, uz) + NN(p)')
    elif arch_type == '03':
        logging.info('Using Architecture 03: NN(ux, uy, uz, p)')
    elif arch_type == '04':
        logging.info('Using Architecture 04: NN(ux, uy, uz, p) + decoders')
    else:
        raise Exception('architecture not recognized!')


    # if lambda annealing
    
    annealing_iter_rate = 1
    if is_lambda_annealing:
        alpha_annealing = options['PINNs']['lambdas']['annealing']['alpha']
        annealing_iter_rate = options['PINNs']['lambdas']['annealing']['iter_rate']
        annealing_mode = options['PINNs']['lambdas']['annealing']['mode']
        logging.info('Applying Lambda-Annealing method with \u03B1 = {}. Mode: {}'.format(options['PINNs']['lambdas']['annealing']['alpha'],annealing_mode))
        lambda_track = {
            'phys': [],
            'data': [],
            'BC': [],
            'pmean': [],
            'windkessel': [],
            'initial': [],
            'gradp': [],
        }
        annealing_info = {
            'phys': [],
            'data': [],
            'BC': [],
            'pmean': [],
            'windkessel': [],
            'initial': [],
            'gradp': [],
        }

    # windkessel boundaries
    windkessel_bnd = options['windkessel']['boundaries']
    pretrain = options['PINNs']['pretrain']['apply']
    pi_track = {}
    for k in windkessel_bnd:
        pi_track[k] = {}

    loading_nets = []
    if pretrain:
        loading_nets = options['PINNs']['pretrain']['loading_nets']
        logging.info('training from model in : {}'.format(options['PINNs']['pretrain']['model']))
        for nets in loading_nets:
            if nets == 'u':
                logging.info('loading velocity net NN(u)')
            elif nets == 'p':
                logging.info('loading pressure net N(p)')
            elif nets == 'pi':
                logging.info('loading pi nets NN(\u03C0)')
                
    
    loss_track = {
        'phys': [],
        'bc': [],
        'data': [],
        'wk': [],
        'pmean': [],
        'initial': [],
        'eps_params': [],
        'eps_sol': [],
        'gradp': [],
        'tot': [],
    }

    give_mean_pressure = False
    if 'give_mean_pressure' in options['windkessel']:
        give_mean_pressure = options['windkessel']['give_mean_pressure']
        if give_mean_pressure:
            logging.info('Mean Pressure Measurements Found!')


    inverse_problem = options['windkessel']['inverse_problem']['apply']
    estim_params = options['windkessel']['inverse_problem']['search']
    estim_bnds = []
    param_factorization = False
    if inverse_problem:
        inverse_iter_rate = HP['inverse_iter_rate']
        inverse_iter_t0 = options['windkessel']['inverse_problem']['iter_t0']
        estim_bnds = options['windkessel']['inverse_problem']['bnds']
        param_factorization = options['windkessel']['inverse_problem']['factorization']
        range_distance_reg = False
        if 'range_distance_reg' in options['windkessel']['inverse_problem']:
            range_distance_reg = options['windkessel']['inverse_problem']['range_distance_reg']
    

    param_track = {}
    for k in estim_bnds:
        param_track[k] = {'Rd': [] , 'Rp': [], 'C': []}
        

    seed = None
    if 'seed' in options['PINNs']:
        seed = options['PINNs']['seed']
        logging.info('Randomness seed set in: {}'.format(seed))
        torch.manual_seed(seed)
        numpy.random.seed(seed)


    adding_Pt0 = False
    adding_pit_reg = False
    adding_pi0 = True
    
    if 'adding_pi0' in options['PINNs']:
        adding_pi0 = options['PINNs']['adding_pi0']
        if not adding_pi0:
            logging.info('No adding initial pi at wk outlets')

    if 'adding_Pt0' in options['PINNs']:
        adding_Pt0 = options['PINNs']['adding_Pt0']
        if adding_Pt0:
            logging.info('Adding initial pressure at wk outlets assuming Q=0')
    if 'adding_pit_reg' in options['PINNs']:
        adding_pit_reg = options['PINNs']['adding_pit_reg']
        if adding_pit_reg:
            logging.info('Adding pi derivative regularizator')

    divergence_free = False
    if 'divergence-free' in options['PINNs']:
        divergence_free = options['PINNs']['divergence-free']
        if divergence_free:
            logging.info('Using Divergence Free Formulation: NN(\u03A6, p)')


    is_Fourier_features = False
    if 'Fourier-Features' in options['PINNs']:
        is_Fourier_features = options['PINNs']['Fourier-Features']['apply']
        if is_Fourier_features:
            fourier_sigma = options['PINNs']['Fourier-Features']['sigma']
            logging.info('Applying Fourier Features from Tancik et al (2020): (x,y,z) -> \u0191(x,y,z)')
            logging.info('Gaussian features with \u03C3 = {}'.format(fourier_sigma))


    if inverse_problem:
        if range_distance_reg:
            logging.info('Adding Parameter Distance Regularizator: ||\u03B8 - \u03B8_range ||\u00B2')


    couplingRdC = False
    if 'couplingRdC' in options['windkessel']['inverse_problem']:
        couplingRdC = options['windkessel']['inverse_problem']['couplingRdC']

        if 'C' not in estim_params:
            couplingRdC = False

        if couplingRdC:
            logging.info('Coupling Rd with C via: R*C = fix value')
            RC_value = options['windkessel']['inverse_problem']['RC_value']
            logging.info('RC value: {}'.format(RC_value))
            RC_value = RC_value/(Factors['L']/Factors['U'])


    u_fem = fenics[0]
    u_length = len(u_fem.vector().get_local())
    # computing normal
    n = dolfin.FacetNormal(fenics[2])
    ds = dolfin.Measure('ds', domain=fenics[2], subdomain_data=fenics[3])
    device = HP['device']

    def InitTensors(device):
        # collocation points
        x = torch.Tensor(CoPoints['x']).to(device)
        y = torch.Tensor(CoPoints['y']).to(device)
        z = torch.Tensor(CoPoints['z']).to(device)
        t = torch.Tensor(CoPoints['t']).to(device)
        # boundary points
        xb = torch.Tensor(DATA['xb']).to(device)
        yb = torch.Tensor(DATA['yb']).to(device)
        zb = torch.Tensor(DATA['zb']).to(device)
        tb = torch.zeros_like(zb).to(device)
        # mean pressure value
        if give_mean_pressure:
            pwk_mean = torch.Tensor(numpy.array(DATA['pwk_mean'][0])).to(device)
            pwk_mean = pwk_mean.view(len(pwk_mean),-1)
        # windkessel points
        xwk = {}
        ywk = {}
        zwk = {}
        twk = {}
        for k in windkessel_bnd:
            xwk[k] = torch.Tensor(DATA['xwk'][k]).to(device)
            ywk[k] = torch.Tensor(DATA['ywk'][k]).to(device)
            zwk[k] = torch.Tensor(DATA['zwk'][k]).to(device)
            twk[k] = torch.Tensor(DATA['twk'][k]).to(device)

        # data points
        td = {}
        xd = {}
        yd = {}
        zd = {}
        uxd = {}
        uyd = {}
        uzd = {}

        for k in DATA['x'].keys():
            td[k] = torch.Tensor(DATA['t'][k]).to(device)
            xd[k] = torch.Tensor(DATA['x'][k]).to(device)
            yd[k] = torch.Tensor(DATA['y'][k]).to(device)
            zd[k] = torch.Tensor(DATA['z'][k]).to(device)
            uxd[k] = torch.Tensor(DATA['ux'][k]).to(device)
            uyd[k] = torch.Tensor(DATA['uy'][k]).to(device)
            uzd[k] = torch.Tensor(DATA['uz'][k]).to(device)


        for ti in tb:
            ti[0] = td[0][-1].item()*numpy.random.rand(1)[0]

        del td
        
        for key in twk.keys():
            twk[key].requires_grad = True
            if not is_Fourier_features:
                xwk[key].requires_grad = True
                ywk[key].requires_grad = True
                zwk[key].requires_grad = True
        
        if divergence_free:
            for key in xd.keys():
                xd[key].requires_grad = True
                yd[key].requires_grad = True
                zd[key].requires_grad = True
            
            xb.requires_grad = True
            yb.requires_grad = True
            zb.requires_grad = True
            x.requires_grad = True
            y.requires_grad = True
            z.requires_grad = True
            

        if is_Fourier_features:
            Bc = torch.normal(0,fourier_sigma**2,size=x.size()).to(device)
            Bb = torch.normal(0,fourier_sigma**2,size=xb.size()).to(device)
            Bw = {}
            for k in xwk.keys():
                Bw[k] = torch.normal(0,fourier_sigma**2,size=xwk[k].size()).to(device)
            Bd = {}
            for k in xd.keys():
                Bd[k] = torch.normal(0,fourier_sigma**2,size=xd[k].size()).to(device)
            
            # collocaation points
            x_c = torch.cos(2*numpy.pi*Bc*x)
            x_s = torch.sin(2*numpy.pi*Bc*x)
            y_c = torch.cos(2*numpy.pi*Bc*y)
            y_s = torch.sin(2*numpy.pi*Bc*y)
            z_c = torch.cos(2*numpy.pi*Bc*z)
            z_s = torch.sin(2*numpy.pi*Bc*z)
            # boundary points
            xb_c = torch.cos(2*numpy.pi*Bb*xb)
            xb_s = torch.sin(2*numpy.pi*Bb*xb)
            yb_c = torch.cos(2*numpy.pi*Bb*yb)
            yb_s = torch.sin(2*numpy.pi*Bb*yb)
            zb_c = torch.cos(2*numpy.pi*Bb*zb)
            zb_s = torch.sin(2*numpy.pi*Bb*zb)
            # windkessel
            xwk_c = {}
            xwk_s = {}
            ywk_c = {}
            ywk_s = {}
            zwk_c = {}
            zwk_s = {}
            for k in twk.keys():
                xwk_c[k] = torch.cos(2*numpy.pi*Bw[k]*xwk[k])
                xwk_s[k] = torch.sin(2*numpy.pi*Bw[k]*xwk[k])
                ywk_c[k] = torch.cos(2*numpy.pi*Bw[k]*ywk[k])
                ywk_s[k] = torch.sin(2*numpy.pi*Bw[k]*ywk[k])
                zwk_c[k] = torch.cos(2*numpy.pi*Bw[k]*zwk[k])
                zwk_s[k] = torch.sin(2*numpy.pi*Bw[k]*zwk[k])
                # allowing compute derivatives
                xwk_c[k].requires_grad = True
                ywk_c[k].requires_grad = True
                zwk_c[k].requires_grad = True
                xwk_s[k].requires_grad = True
                ywk_s[k].requires_grad = True
                zwk_s[k].requires_grad = True

            # data points
            xd_c = {}
            xd_s = {}
            yd_c = {}
            yd_s = {}
            zd_c = {}
            zd_s = {}
            for k in DATA['x'].keys():
                xd_c[k] = torch.cos(2*numpy.pi*Bd[k]*xd[k])
                xd_s[k] = torch.sin(2*numpy.pi*Bd[k]*xd[k])
                yd_c[k] = torch.cos(2*numpy.pi*Bd[k]*yd[k])
                yd_s[k] = torch.sin(2*numpy.pi*Bd[k]*yd[k])
                zd_c[k] = torch.cos(2*numpy.pi*Bd[k]*zd[k])
                zd_s[k] = torch.sin(2*numpy.pi*Bd[k]*zd[k])


        # changing to float... cuda slower in double?. taken from arzani
        changing_to_float = True
        if changing_to_float:
            x = x.type(torch.cuda.FloatTensor)
            y = y.type(torch.cuda.FloatTensor)
            z = z.type(torch.cuda.FloatTensor)
            t = t.type(torch.cuda.FloatTensor) 
            xb = xb.type(torch.cuda.FloatTensor)
            yb = yb.type(torch.cuda.FloatTensor)
            zb = zb.type(torch.cuda.FloatTensor)
            tb = tb.type(torch.cuda.FloatTensor)
            if give_mean_pressure:
                pwk_mean = pwk_mean.type(torch.cuda.FloatTensor)
            for k in windkessel_bnd:
                xwk[k] = xwk[k].type(torch.cuda.FloatTensor)
                ywk[k] = ywk[k].type(torch.cuda.FloatTensor)
                zwk[k] = zwk[k].type(torch.cuda.FloatTensor)
                twk[k] = twk[k].type(torch.cuda.FloatTensor)
            for k in DATA['x'].keys():
                xd[k] = xd[k].type(torch.cuda.FloatTensor)
                yd[k] = yd[k].type(torch.cuda.FloatTensor)
                zd[k] = zd[k].type(torch.cuda.FloatTensor)
                uxd[k] = uxd[k].type(torch.cuda.FloatTensor)
                uyd[k] = uyd[k].type(torch.cuda.FloatTensor)
                uzd[k] = uzd[k].type(torch.cuda.FloatTensor)

            if is_Fourier_features:
                x_c = x_c.type(torch.cuda.FloatTensor)
                x_s = x_s.type(torch.cuda.FloatTensor)
                y_c = y_c.type(torch.cuda.FloatTensor)
                y_s = y_s.type(torch.cuda.FloatTensor) 
                z_c = z_c.type(torch.cuda.FloatTensor)
                z_s = z_s.type(torch.cuda.FloatTensor)
                Bc = Bc.type(torch.cuda.FloatTensor)
                Bb = Bb.type(torch.cuda.FloatTensor)
                xb_c = xb_c.type(torch.cuda.FloatTensor)
                xb_s = xb_s.type(torch.cuda.FloatTensor)
                yb_c = yb_c.type(torch.cuda.FloatTensor)
                yb_s = yb_s.type(torch.cuda.FloatTensor) 
                zb_c = zb_c.type(torch.cuda.FloatTensor)
                zb_s = zb_s.type(torch.cuda.FloatTensor)
                for k in windkessel_bnd:
                    xwk_c[k] = xwk_c[k].type(torch.cuda.FloatTensor)
                    ywk_c[k] = ywk_c[k].type(torch.cuda.FloatTensor)
                    zwk_c[k] = zwk_c[k].type(torch.cuda.FloatTensor)
                    xwk_s[k] = xwk_s[k].type(torch.cuda.FloatTensor)
                    ywk_s[k] = ywk_s[k].type(torch.cuda.FloatTensor)
                    zwk_s[k] = zwk_s[k].type(torch.cuda.FloatTensor)
                    Bw[k] = Bw[k].type(torch.cuda.FloatTensor)
                for k in DATA['x'].keys():
                    xd_c[k] = xd_c[k].type(torch.cuda.FloatTensor)
                    yd_c[k] = yd_c[k].type(torch.cuda.FloatTensor)
                    zd_c[k] = zd_c[k].type(torch.cuda.FloatTensor)
                    xd_s[k] = xd_s[k].type(torch.cuda.FloatTensor)
                    yd_s[k] = yd_s[k].type(torch.cuda.FloatTensor)
                    zd_s[k] = zd_s[k].type(torch.cuda.FloatTensor)
                    Bd[k] = Bd[k].type(torch.cuda.FloatTensor)
                

        if not is_Fourier_features:
            return x,y,z,t,xb,yb,zb,tb,xd,yd,zd,uxd,uyd,uzd,xwk,ywk,zwk,twk, pwk_mean
        else:
            del x,y,z,xb,yb,zb,xd,yd,zd,xwk,ywk,zwk
            return x_c,x_s,y_c,y_s,z_c,z_s,t,xb_c,xb_s,yb_c,yb_s,zb_c,zb_s,tb,xd_c,xd_s,yd_c,yd_s,zd_c,zd_s,uxd, \
                 uyd,uzd,xwk_c,xwk_s,ywk_c,ywk_s,zwk_c,zwk_s,twk, pwk_mean, Bc, Bw

    if not is_Fourier_features:
        x,y,z,t,xb,yb,zb,tb,xd,yd,zd,uxd,uyd,uzd,xwk,ywk,zwk,twk,pwk_mean = InitTensors(device)
        # dataset
        dataset = TensorDataset(x,y,z,t)
        dataloader = DataLoader(dataset, batch_size=HP['batchsize'], shuffle=True, num_workers = 0, drop_last = True )

    else:
        x_c,x_s,y_c,y_s,z_c,z_s,t,xb_c,xb_s,yb_c,yb_s,zb_c,zb_s,tb,xd_c,xd_s,yd_c,yd_s,zd_c,zd_s,uxd, \
                 uyd,uzd,xwk_c,xwk_s,ywk_c,ywk_s,zwk_c,zwk_s,twk, pwk_mean, Bc, Bw = InitTensors(device)

        # dataset
        dataset = TensorDataset(x_c,x_s,y_c,y_s,z_c,z_s,t,Bc)
        dataloader = DataLoader(dataset, batch_size=HP['batchsize'], shuffle=True, num_workers = 0, drop_last = True )

    

    h_n = HP['neurons_per_layer']
    h_n_pi = HP['neurons_per_layer_pi']
    n_layers = HP['hidden_layers']
    n_layers_pi = HP['hidden_layers_pi']
    input_n = 4
    act_function = HP['activation_function']
    act_function_pi = HP['activation_function_pi']
    
    if is_Fourier_features:
        input_n = 7 #2*(x,y,z) + t
    
    class ActivationFunction(nn.Module):
        def __init__(self, type, inplace=True):
            super(ActivationFunction, self).__init__()
            self.inplace = inplace
            self.function = type

        def forward(self, x):
            # relu
            if self.function == 'relu':
                return torch.relu(x)
            # swish
            elif self.function == 'swish':
                if self.inplace:
                    x.mul_(torch.sigmoid(x))
                    return x
                else:
                    return x*torch.sigmoid(x)
            # tanh
            elif self.function == 'tanh':
                return torch.tanh(x)
    
    class Resistance(nn.Module):
        def __init__(self, range):
            super(Resistance, self).__init__()
            if isinstance(range,list):
                rmin = range[0]
                rmax = range[1]
                center = 0.5*(rmax+rmin)
                ratio = 0.5*(rmax-rmin)
                self.R = Variable((center+ratio*(2*torch.rand(1)-1)).to(device), requires_grad=True)
            else:
                self.R = Variable(torch.tensor([range]).to(device), requires_grad=True)
        def forward(self):
            return self.R
    
    class Capacitance(nn.Module):
        def __init__(self, range):
            super(Capacitance, self).__init__()
            if isinstance(range,list):
                rmin = range[0]
                rmax = range[1]
                center = 0.5*(rmax+rmin)
                ratio = 0.5*(rmax-rmin)
                self.C = Variable((center+ratio*(2*torch.rand(1)-1)).to(device), requires_grad=True)
            else:
                self.C = Variable(torch.tensor([range]).to(device), requires_grad=True)
        def forward(self):
            return self.C

    class NN_all(nn.Module):
        def __init__(self):
            super(NN_all, self).__init__()
            layers = []
            layers.append(nn.Linear(input_n,h_n))
            layers.append(ActivationFunction(act_function))
            for i in range(n_layers):
                layers.append(nn.Linear(h_n,h_n))
                layers.append(ActivationFunction(act_function))
            
            layers.append(nn.Linear(h_n,4))
            self.main = nn.Sequential(*layers)
            self.act_function = act_function
        
        def forward(self, x):
            output = self.main(x)
            return output

    class NN_u(nn.Module):
        def __init__(self):
            super(NN_u, self).__init__()
            layers = []
            layers.append(nn.Linear(input_n,h_n))
            layers.append(ActivationFunction(act_function))
            for i in range(n_layers):
                layers.append(nn.Linear(h_n,h_n))
                layers.append(ActivationFunction(act_function))
            
            layers.append(nn.Linear(h_n,3))
            self.main = nn.Sequential(*layers)
            self.act_function = act_function
        
        def forward(self, x):
            output = self.main(x)
            return output

    class NN_ux(nn.Module):
        def __init__(self):
            super(NN_ux, self).__init__()
            layers = []
            layers.append(nn.Linear(input_n,h_n))
            layers.append(ActivationFunction(act_function))
            for i in range(n_layers):
                layers.append(nn.Linear(h_n,h_n))
                layers.append(ActivationFunction(act_function))
            
            layers.append(nn.Linear(h_n,1))
            self.main = nn.Sequential(*layers)
            self.act_function = act_function
        
        def forward(self, x):
            output = self.main(x)
            return output

    class NN_uy(nn.Module):
        def __init__(self):
            super(NN_uy, self).__init__()
            layers = []
            layers.append(nn.Linear(input_n,h_n))
            layers.append(ActivationFunction(act_function))
            for i in range(n_layers):
                layers.append(nn.Linear(h_n,h_n))
                layers.append(ActivationFunction(act_function))
            
            layers.append(nn.Linear(h_n,1))
            self.main = nn.Sequential(*layers)
            self.act_function = act_function
        
        def forward(self, x):
            output = self.main(x)
            return output

    class NN_uz(nn.Module):
        def __init__(self):
            super(NN_uz, self).__init__()
            layers = []
            layers.append(nn.Linear(input_n,h_n))
            layers.append(ActivationFunction(act_function))
            for i in range(n_layers):
                layers.append(nn.Linear(h_n,h_n))
                layers.append(ActivationFunction(act_function))
            
            layers.append(nn.Linear(h_n,1))
            self.main = nn.Sequential(*layers)
            self.act_function = act_function
        
        def forward(self, x):
            output = self.main(x)
            return output

    class NN_p(nn.Module):
        def __init__(self):
            super(NN_p, self).__init__()
            layers = []
            layers.append(nn.Linear(input_n,h_n))
            layers.append(ActivationFunction(act_function))
            for i in range(n_layers):
                layers.append(nn.Linear(h_n,h_n))
                layers.append(ActivationFunction(act_function))
            
            layers.append(nn.Linear(h_n,1))
            self.main = nn.Sequential(*layers)
            self.act_function = act_function
        
        def forward(self, x):
            output = self.main(x)
            return output

    class NN_pi(nn.Module):
        def __init__(self):
            super(NN_pi, self).__init__()
            layers = []
            layers.append(nn.Linear(1,h_n_pi))
            layers.append(ActivationFunction(act_function_pi))
            for i in range(n_layers_pi):
                layers.append(nn.Linear(h_n_pi,h_n_pi))
                layers.append(ActivationFunction(act_function_pi))
            
            layers.append(nn.Linear(h_n_pi,1))
            self.main = nn.Sequential(*layers)
            self.act_function = act_function_pi
        
        def forward(self, x):
            output = self.main(x)
            return output

    class NN_all_d(nn.Module):
        def __init__(self):
            super(NN_all_d, self).__init__()
            
            self.main = nn.ModuleList([])
            self.main.append(nn.Linear(input_n,h_n))
            for i in range(n_layers):
                self.main.append(nn.Linear(h_n,h_n))
            
            self.main.append(nn.Linear(h_n,4))
            # adding two extra residual layers
            self.main.append(nn.Linear(input_n,h_n))
            self.main.append(nn.Linear(input_n,h_n))
            
            self.activation = ActivationFunction(act_function)
            self.depth = len(self.main)-2
        
        def forward(self, x):

            x_upd = self.main[0](x)
            x_upd = self.activation(x_upd)

            U = self.main[self.depth](x)
            U = self.activation(U)
            V = self.main[self.depth+1](x)
            V = self.activation(V)
            

            for l in range(1,self.depth-1):
                x_upd = self.main[l](x_upd)
                x_upd = self.activation(x_upd)
                x_upd = (1-x_upd)*U + x_upd*V


            x_upd = self.main[self.depth-1](x_upd)

            return x_upd

    class NN_pi_d(nn.Module):
        def __init__(self):
            super(NN_pi_d, self).__init__()
            
            self.main = nn.ModuleList([])
            self.main.append(nn.Linear(1,h_n_pi))
            for i in range(n_layers_pi):
                self.main.append(nn.Linear(h_n_pi,h_n_pi))
            
            self.main.append(nn.Linear(h_n_pi,1))
            # adding two extra residual layers
            self.main.append(nn.Linear(1,h_n_pi))
            self.main.append(nn.Linear(1,h_n_pi))
            
            self.activation = ActivationFunction(act_function_pi)
            self.depth = len(self.main)-2
        
        def forward(self, x):

            x_upd = self.main[0](x)
            x_upd = self.activation(x_upd)

            U = self.main[self.depth](x)
            U = self.activation(U)
            V = self.main[self.depth+1](x)
            V = self.activation(V)
            

            for l in range(1,self.depth-1):
                x_upd = self.main[l](x_upd)
                x_upd = self.activation(x_upd)
                x_upd = (1-x_upd)*U + x_upd*V


            x_upd = self.main[self.depth-1](x_upd)

            return x_upd


    if arch_type == '01':
        nn_ux = NN_ux().to(device)
        nn_uy = NN_uy().to(device)
        nn_uz = NN_uz().to(device)
        nn_p = NN_p().to(device)
    elif arch_type == '02':
        nn_u = NN_u().to(device)
        nn_p = NN_p().to(device)
    elif arch_type == '03':
        nn_all = NN_all().to(device)
    elif arch_type == '04':
        nn_all = NN_all_d().to(device)

    nns_pi = []
    for k in windkessel_bnd:
        if arch_type == '04':
            nns_pi.append(NN_pi_d().to(device))
        else:
            nns_pi.append(NN_pi().to(device))

    if is_lambda_annealing or is_tracking_gradients:
        if arch_type == '01':
            l_network_1 = nn_ux
            l_network_2 = nn_p
        elif arch_type == '02':
            l_network_1 = nn_u
            l_network_2 = nn_p
        elif arch_type == '03':
            l_network_1 = nn_all
            l_network_2 = nn_all


    # INIT PARAMETERS
    # initializing Windkessel parameters: tuples with min and max values
    Rp_range = options['windkessel']['inverse_problem']['Rp_range']
    Rd_range = options['windkessel']['inverse_problem']['Rd_range']
    C_range = options['windkessel']['inverse_problem']['C_range']
    InitRd_WithDict = isinstance(Rd_range,dict)
    InitRp_WithDict = isinstance(Rp_range,dict)
    InitC_WithDict = isinstance(C_range,dict)

    Rd_i = {}
    Rp_i = {}
    C_i = {}
    Rd_ref = {}
    Rp_ref = {}
    C_ref = {}
    r_fact = Factors['Rfac']
    c_fact = Factors['Cfac']

    input_path = options['data']['input_path'] + 'input.yaml'
    with open(input_path , 'r+') as f:
        sim_options = yaml.load(f, Loader=yaml.Loader)

    # init Rd
    for bid in options['windkessel']['boundaries']:
        if inverse_problem and bid in estim_bnds:
            if 'Rd' in estim_params:
                if InitRd_WithDict:
                    Rd_i[bid] = Resistance(Rd_range[bid]).to(device)
                else:
                    Rd_i[bid] = Resistance(Rd_range).to(device)

        for bc in sim_options['boundary_conditions']:
            if bid == bc['id']:
                Rd_ref[bid] = bc['parameters']['R_d']/r_fact
                Rp_ref[bid] = bc['parameters']['R_p']/r_fact
                C_ref[bid] = bc['parameters']['C']/c_fact

    # init Rp
    for bid in options['windkessel']['boundaries']:
        if inverse_problem and bid in estim_bnds:
            if 'Rp' in estim_params:
                if InitRp_WithDict:
                    Rp_i[bid] = Resistance(Rp_range[bid]).to(device)
                else:
                    Rp_i[bid] = Resistance(Rp_range).to(device)
            
        for bc in sim_options['boundary_conditions']:
            if bid == bc['id']:
                Rd_ref[bid] = bc['parameters']['R_d']/r_fact
                Rp_ref[bid] = bc['parameters']['R_p']/r_fact
                C_ref[bid] = bc['parameters']['C']/c_fact

    # init C
    for bid in options['windkessel']['boundaries']:
        if inverse_problem and bid in estim_bnds:
            if 'C' in estim_params:
                if not couplingRdC:
                    if InitC_WithDict:
                        C_i[bid] = Capacitance(C_range[bid]).to(device)
                    else:
                        C_i[bid] = Capacitance(C_range).to(device)
                else:
                    C_i[bid] = Capacitance(RC_value/Rd_i[bid]()).to(device)
                    
        for bc in sim_options['boundary_conditions']:
            if bid == bc['id']:
                Rd_ref[bid] = bc['parameters']['R_d']/r_fact
                Rp_ref[bid] = bc['parameters']['R_p']/r_fact
                C_ref[bid] = bc['parameters']['C']/c_fact



    if pretrain:
        logging.info('(PRETRAIN) reading previous model...')
        if arch_type == '01':
            if 'u' in loading_nets:
                nn_ux.load_state_dict(torch.load(options['PINNs']['pretrain']['model'] + 'sten_data_ux.pt'))
                nn_uy.load_state_dict(torch.load(options['PINNs']['pretrain']['model'] + 'sten_data_uy.pt'))
                nn_uz.load_state_dict(torch.load(options['PINNs']['pretrain']['model'] + 'sten_data_uz.pt'))
            elif 'p' in loading_nets:
                nn_p.load_state_dict(torch.load(options['PINNs']['pretrain']['model'] + 'sten_data_p.pt'))
        elif arch_type == '02':
            if 'u' in loading_nets:
                nn_u.load_state_dict(torch.load(options['PINNs']['pretrain']['model'] + 'sten_data_u.pt'))
            if 'p' in loading_nets:
                nn_p.load_state_dict(torch.load(options['PINNs']['pretrain']['model'] + 'sten_data_p.pt'))
        elif arch_type == '03':
            if 'u' in loading_nets or ('p' in loading_nets):
                nn_all.load_state_dict(torch.load(options['PINNs']['pretrain']['model'] + 'sten_data_all.pt'))
        
        if 'pi' in loading_nets:
            for l,nn_pi in enumerate(nns_pi):
                nn_pi.load_state_dict(torch.load(options['PINNs']['pretrain']['model'] + 'sten_data_pi_'+ str(windkessel_bnd[l]) +'.pt'))


        if False:
            if inverse_problem:
                logging.info('(PRETRAIN) reading last parameter estimation...')
                idx = options['PINNs']['pretrain']['model'][0:-1].rfind('/')
                params_path = options['PINNs']['pretrain']['model'][0:idx] + '/estimation.npz'
                dparams = numpy.load(params_path, allow_pickle=True)
                params = dparams['arr_0'].item()
                for k in Rd_i.keys():
                    Rd_i[k] = Resistance(params[k]['Rd'][-1]).to(device)
                for k in Rp_i.keys():
                    Rp_i[k] = Resistance(params[k]['Rp'][-1]).to(device)
                for k in C_i.keys():
                    C_i[k] = Capacitance(params[k]['C'][-1]).to(device)
                
                del params, dparams


    if inverse_problem:
        if param_factorization:
            for k in Rd_i.keys():
                par_i = numpy.log(Rd_i[k].R.item())
                Rd_i[k] = Resistance(par_i).to(device)
            for k in Rp_i.keys():
                par_i = numpy.log(Rp_i[k].R.item())
                Rp_i[k] = Resistance(par_i).to(device)
            for k in C_i.keys():
                par_i = numpy.log(C_i[k].C.item())
                C_i[k] = Capacitance(par_i).to(device)
        
        if range_distance_reg:
            Rd_range_min = {}
            Rp_range_min = {}
            C_range_min = {}
            Rd_range_max = {}
            Rp_range_max = {}
            C_range_max = {}

            for k in Rd_i.keys():
                Rd_range_min[k] = Resistance(float(Rd_range[0])).to(device)
                Rd_range_max[k] = Resistance(float(Rd_range[1])).to(device)
            for k in Rp_i.keys():
                Rp_range_min[k] = Resistance(float(Rp_range[0])).to(device)
                Rp_range_max[k] = Resistance(float(Rp_range[1])).to(device)
            for k in C_i.keys():
                C_range_min[k] = Capacitance(float(C_range[0])).to(device)
                C_range_max[k] = Capacitance(float(C_range[1])).to(device)
                
            

    # temporal upsampling factors
    t_upsampling_fac = options['PINNs']['temporal_upsampling_factor']
    times_meas = DATA['t'][0]
    dt_meas = times_meas[3] - times_meas[2]
    times_up = numpy.arange(0,times_meas[-1], dt_meas/t_upsampling_fac)

    if times_meas[-1] != times_up[-1]:
        times_up = numpy.append(times_up, times_meas[-1])


    Q_fenics = {}
    Q_interpolator = {}
    
    for k in windkessel_bnd:
        Q_fenics[k] = numpy.zeros_like(times_up)
        Q_interpolator[k] = interp1d(times_up, Q_fenics[k], kind='cubic', fill_value='extrapolate')

     
    # printing information about Windkessel parameters
    if inverse_problem:
        if param_factorization:
            logging.info('Applying Parameter Factorization: 2^\u03B8')
    logging.info('--- Parameter Values ---')

    PrintEstimation(inverse_problem, Rd_i, Rd_ref, C_i, C_ref, Rp_i, Rp_ref, param_track, logscale=param_factorization)

    # initializing weights
    def init_normal(m):
        if type(m) == nn.Linear:
            if act_function == 'relu':
                nn.init.kaiming_normal_(m.weight) # He (relu)
            else:
                nn.init.xavier_normal_(m.weight) # Xavier
            
    def init_normal_pi(m):
        if type(m) == nn.Linear:
            if act_function_pi == 'relu':
                nn.init.kaiming_normal_(m.weight) # He (relu)
            else:
                nn.init.xavier_normal_(m.weight) # Xavier


    if arch_type == '01':
        if not pretrain or ('u' not in loading_nets):
            nn_ux.apply(init_normal)
            nn_uy.apply(init_normal)
            nn_uz.apply(init_normal)
        if not pretrain or ('p' not in loading_nets):
            nn_p.apply(init_normal)
    elif arch_type == '02':
        if not pretrain or ('u' not in loading_nets):
            nn_u.apply(init_normal)
        if not pretrain or ('p' not in loading_nets):
            nn_p.apply(init_normal)
    elif arch_type == '03':
        if not pretrain or ('u' not in loading_nets) or ('p' not in loading_nets):
            nn_all.apply(init_normal)

    # initializing pis
    if not pretrain or ('pi' not in loading_nets):
        for nn_pi in nns_pi:
            nn_pi.apply(init_normal_pi)
        

    # Optimizer: Adam
    if arch_type == '01':
        optimizer_ux = torch.optim.Adam(nn_ux.parameters(), lr=HP['learning_rates']['state']['l'], betas = (0.9,0.99),eps = 10**-15)
        optimizer_uy = torch.optim.Adam(nn_uy.parameters(), lr=HP['learning_rates']['state']['l'], betas = (0.9,0.99),eps = 10**-15)
        optimizer_uz = torch.optim.Adam(nn_uz.parameters(), lr=HP['learning_rates']['state']['l'], betas = (0.9,0.99),eps = 10**-15)    
        optimizer_p = torch.optim.Adam(nn_p.parameters(), lr=HP['learning_rates']['pressure']['l'], betas = (0.9,0.99),eps = 10**-15)
    elif arch_type == '02':
        optimizer_u = torch.optim.Adam(nn_u.parameters(), lr=HP['learning_rates']['state']['l'], betas = (0.9,0.99),eps = 10**-15)
        optimizer_p = torch.optim.Adam(nn_p.parameters(), lr=HP['learning_rates']['pressure']['l'], betas = (0.9,0.99),eps = 10**-15)
    elif arch_type in ['03','04']:
        optimizer_all = torch.optim.Adam(nn_all.parameters(), lr=HP['learning_rates']['state']['l'], betas = (0.9,0.99),eps = 10**-15)

    optimizer_pi_lst = []
    for nn_pi in nns_pi:
        optimizer_pi_lst.append(torch.optim.Adam(nn_pi.parameters(), lr=HP['learning_rates']['state']['l'], betas = (0.9,0.99),eps = 10**-15))
    

    optimizer_Rd_lst = []
    optimizer_Rp_lst = []
    optimizer_C_lst = []
    
    if inverse_problem:
        for key in Rd_i.keys():
            optimizer_Rd_lst.append(torch.optim.Adam([{'params':[Rd_i[key].R], 'lr':HP['learning_rates']['params']['l']}], betas = (0.9,0.99),eps = 10**-15))
        for key in Rp_i.keys():
            optimizer_Rp_lst.append(torch.optim.Adam([{'params':[Rp_i[key].R], 'lr':HP['learning_rates']['params']['l']}], betas = (0.9,0.99),eps = 10**-15))
        for key in C_i.keys():
            if not couplingRdC:
                optimizer_C_lst.append(torch.optim.Adam([{'params':[C_i[key].C], 'lr':HP['learning_rates']['params']['l']}], betas = (0.9,0.99),eps = 10**-15))
        

    def Loss_Phys(x, y, z, t):

        x.requires_grad = True
        y.requires_grad = True
        z.requires_grad = True
        t.requires_grad = True
        
        nn_in = torch.cat((x,y,z,t),1)
        
        if arch_type == '01':
            ux = nn_ux(nn_in)
            uy = nn_uy(nn_in)
            uz = nn_uz(nn_in)
            p = nn_p(nn_in)
        elif arch_type == '02':
            u = nn_u(nn_in)
            ux = u[:,0]
            uy = u[:,1]
            uz = u[:,2]
            p = nn_p(nn_in)
        elif arch_type in ['03','04']:
            up = nn_all(nn_in)
            ux = up[:,0]
            uy = up[:,1]
            uz = up[:,2]
            p = up[:,3]
            
        ux = ux.view(len(ux),-1)
        uy = uy.view(len(uy),-1)
        uz = uz.view(len(uz),-1)
        p = p.view(len(p),-1)

        # computing derivatives
        # ut
        ux_t = torch.autograd.grad(ux, t, grad_outputs=torch.ones_like(t), create_graph = True,only_inputs=True)[0]
        uy_t = torch.autograd.grad(uy, t, grad_outputs=torch.ones_like(t), create_graph = True,only_inputs=True)[0]
        uz_t = torch.autograd.grad(uz, t, grad_outputs=torch.ones_like(t), create_graph = True,only_inputs=True)[0]
        # ux
        ux_x = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        ux_xx = torch.autograd.grad(ux_x, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        ux_y = torch.autograd.grad(ux, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        ux_yy = torch.autograd.grad(ux_y, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        ux_z = torch.autograd.grad(ux, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        ux_zz = torch.autograd.grad(ux_z, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        # uy
        uy_x = torch.autograd.grad(uy, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        uy_xx = torch.autograd.grad(uy_x, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        uy_y = torch.autograd.grad(uy, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        uy_yy = torch.autograd.grad(uy_y, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        uy_z = torch.autograd.grad(uy, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        uy_zz = torch.autograd.grad(uy_z, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        # uz
        uz_x = torch.autograd.grad(uz, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        uz_xx = torch.autograd.grad(uz_x, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        uz_y = torch.autograd.grad(uz, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        uz_yy = torch.autograd.grad(uz_y, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        uz_z = torch.autograd.grad(uz, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        uz_zz = torch.autograd.grad(uz_z, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        # grad p
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        p_z = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]

        Re = Factors['Re']


        loss_1 = ux_t + ux*ux_x + uy*ux_y + uz*ux_z - (1/Re)*( ux_xx + ux_yy + ux_zz) + p_x
        loss_2 = uy_t + ux*uy_x + uy*uy_y + uz*uy_z - (1/Re)*( uy_xx + uy_yy + uy_zz) + p_y
        loss_3 = uz_t + ux*uz_x + uy*uz_y + uz*uz_z - (1/Re)*( uz_xx + uz_yy + uz_zz) + p_z
        loss_4 = ux_x  + uy_y + uz_z  # continuity

        # MSE loss
        loss_f = nn.MSELoss()

        loss = loss_f(loss_1,torch.zeros_like(loss_1)) \
                + loss_f(loss_2,torch.zeros_like(loss_2)) \
                + loss_f(loss_3,torch.zeros_like(loss_3)) \
                + loss_f(loss_4,torch.zeros_like(loss_4))

        return loss

    def Loss_Phys_Fourier(x_c, x_s, y_c, y_s, z_c, z_s, Bc, t):

        x_c.requires_grad = True
        y_c.requires_grad = True
        z_c.requires_grad = True
        x_s.requires_grad = True
        y_s.requires_grad = True
        z_s.requires_grad = True
        t.requires_grad = True
        
        nn_in = torch.cat((x_c,x_s,y_c,y_s,z_c,z_s,t),1)
        
        if arch_type == '01':
            ux = nn_ux(nn_in)
            uy = nn_uy(nn_in)
            uz = nn_uz(nn_in)
            p = nn_p(nn_in)
        elif arch_type == '02':
            u = nn_u(nn_in)
            ux = u[:,0]
            uy = u[:,1]
            uz = u[:,2]
            p = nn_p(nn_in)
        elif arch_type in ['03','04']:
            up = nn_all(nn_in)
            ux = up[:,0]
            uy = up[:,1]
            uz = up[:,2]
            p = up[:,3]
            
        ux = ux.view(len(ux),-1)
        uy = uy.view(len(uy),-1)
        uz = uz.view(len(uz),-1)
        p = p.view(len(p),-1)

        # computing derivatives
        # ut
        ux_t = torch.autograd.grad(ux, t, grad_outputs=torch.ones_like(t), create_graph = True,only_inputs=True)[0]
        uy_t = torch.autograd.grad(uy, t, grad_outputs=torch.ones_like(t), create_graph = True,only_inputs=True)[0]
        uz_t = torch.autograd.grad(uz, t, grad_outputs=torch.ones_like(t), create_graph = True,only_inputs=True)[0]
        # ux
        ux_xc = torch.autograd.grad(ux, x_c, grad_outputs=torch.ones_like(x_c), create_graph = True,only_inputs=True)[0]
        ux_xs = torch.autograd.grad(ux, x_s, grad_outputs=torch.ones_like(x_s), create_graph = True,only_inputs=True)[0]
        ux_xcc = torch.autograd.grad(ux_xc, x_c, grad_outputs=torch.ones_like(x_c), create_graph = True,only_inputs=True)[0]
        ux_xcs = torch.autograd.grad(ux_xc, x_s, grad_outputs=torch.ones_like(x_s), create_graph = True,only_inputs=True)[0]
        ux_xss = torch.autograd.grad(ux_xs, x_s, grad_outputs=torch.ones_like(x_s), create_graph = True,only_inputs=True)[0]
        ux_yc = torch.autograd.grad(ux, y_c, grad_outputs=torch.ones_like(y_c), create_graph = True,only_inputs=True)[0]
        ux_ys = torch.autograd.grad(ux, y_s, grad_outputs=torch.ones_like(y_s), create_graph = True,only_inputs=True)[0]
        ux_ycc = torch.autograd.grad(ux_yc, y_c, grad_outputs=torch.ones_like(y_c), create_graph = True,only_inputs=True)[0]
        ux_ycs = torch.autograd.grad(ux_yc, y_s, grad_outputs=torch.ones_like(y_s), create_graph = True,only_inputs=True)[0]
        ux_yss = torch.autograd.grad(ux_ys, y_s, grad_outputs=torch.ones_like(y_s), create_graph = True,only_inputs=True)[0]
        ux_zc = torch.autograd.grad(ux, z_c, grad_outputs=torch.ones_like(z_c), create_graph = True,only_inputs=True)[0]
        ux_zs = torch.autograd.grad(ux, z_s, grad_outputs=torch.ones_like(z_s), create_graph = True,only_inputs=True)[0]
        ux_zcc = torch.autograd.grad(ux_zc, z_c, grad_outputs=torch.ones_like(z_c), create_graph = True,only_inputs=True)[0]
        ux_zcs = torch.autograd.grad(ux_zc, z_s, grad_outputs=torch.ones_like(z_s), create_graph = True,only_inputs=True)[0]
        ux_zss = torch.autograd.grad(ux_zs, z_s, grad_outputs=torch.ones_like(z_s), create_graph = True,only_inputs=True)[0]
        # uy
        uy_xc = torch.autograd.grad(uy, x_c, grad_outputs=torch.ones_like(x_c), create_graph = True,only_inputs=True)[0]
        uy_xs = torch.autograd.grad(uy, x_s, grad_outputs=torch.ones_like(x_s), create_graph = True,only_inputs=True)[0]
        uy_xcc = torch.autograd.grad(uy_xc, x_c, grad_outputs=torch.ones_like(x_c), create_graph = True,only_inputs=True)[0]
        uy_xcs = torch.autograd.grad(uy_xc, x_s, grad_outputs=torch.ones_like(x_s), create_graph = True,only_inputs=True)[0]
        uy_xss = torch.autograd.grad(uy_xs, x_s, grad_outputs=torch.ones_like(x_s), create_graph = True,only_inputs=True)[0]
        uy_yc = torch.autograd.grad(uy, y_c, grad_outputs=torch.ones_like(y_c), create_graph = True,only_inputs=True)[0]
        uy_ys = torch.autograd.grad(uy, y_s, grad_outputs=torch.ones_like(y_s), create_graph = True,only_inputs=True)[0]
        uy_ycc = torch.autograd.grad(uy_yc, y_c, grad_outputs=torch.ones_like(y_c), create_graph = True,only_inputs=True)[0]
        uy_ycs = torch.autograd.grad(uy_yc, y_s, grad_outputs=torch.ones_like(y_s), create_graph = True,only_inputs=True)[0]
        uy_yss = torch.autograd.grad(uy_ys, y_s, grad_outputs=torch.ones_like(y_s), create_graph = True,only_inputs=True)[0]
        uy_zc = torch.autograd.grad(uy, z_c, grad_outputs=torch.ones_like(z_c), create_graph = True,only_inputs=True)[0]
        uy_zs = torch.autograd.grad(uy, z_s, grad_outputs=torch.ones_like(z_s), create_graph = True,only_inputs=True)[0]
        uy_zcc = torch.autograd.grad(uy_zc, z_c, grad_outputs=torch.ones_like(z_c), create_graph = True,only_inputs=True)[0]
        uy_zcs = torch.autograd.grad(uy_zc, z_s, grad_outputs=torch.ones_like(z_s), create_graph = True,only_inputs=True)[0]
        uy_zss = torch.autograd.grad(uy_zs, z_s, grad_outputs=torch.ones_like(z_s), create_graph = True,only_inputs=True)[0]        
        # uz
        uz_xc = torch.autograd.grad(uz, x_c, grad_outputs=torch.ones_like(x_c), create_graph = True,only_inputs=True)[0]
        uz_xs = torch.autograd.grad(uz, x_s, grad_outputs=torch.ones_like(x_s), create_graph = True,only_inputs=True)[0]
        uz_xcc = torch.autograd.grad(uz_xc, x_c, grad_outputs=torch.ones_like(x_c), create_graph = True,only_inputs=True)[0]
        uz_xcs = torch.autograd.grad(uz_xc, x_s, grad_outputs=torch.ones_like(x_s), create_graph = True,only_inputs=True)[0]
        uz_xss = torch.autograd.grad(uz_xs, x_s, grad_outputs=torch.ones_like(x_s), create_graph = True,only_inputs=True)[0]
        uz_yc = torch.autograd.grad(uz, y_c, grad_outputs=torch.ones_like(y_c), create_graph = True,only_inputs=True)[0]
        uz_ys = torch.autograd.grad(uz, y_s, grad_outputs=torch.ones_like(y_s), create_graph = True,only_inputs=True)[0]
        uz_ycc = torch.autograd.grad(uz_yc, y_c, grad_outputs=torch.ones_like(y_c), create_graph = True,only_inputs=True)[0]
        uz_ycs = torch.autograd.grad(uz_yc, y_s, grad_outputs=torch.ones_like(y_s), create_graph = True,only_inputs=True)[0]
        uz_yss = torch.autograd.grad(uz_ys, y_s, grad_outputs=torch.ones_like(y_s), create_graph = True,only_inputs=True)[0]
        uz_zc = torch.autograd.grad(uz, z_c, grad_outputs=torch.ones_like(z_c), create_graph = True,only_inputs=True)[0]
        uz_zs = torch.autograd.grad(uz, z_s, grad_outputs=torch.ones_like(z_s), create_graph = True,only_inputs=True)[0]
        uz_zcc = torch.autograd.grad(uz_zc, z_c, grad_outputs=torch.ones_like(z_c), create_graph = True,only_inputs=True)[0]
        uz_zcs = torch.autograd.grad(uz_zc, z_s, grad_outputs=torch.ones_like(z_s), create_graph = True,only_inputs=True)[0]
        uz_zss = torch.autograd.grad(uz_zs, z_s, grad_outputs=torch.ones_like(z_s), create_graph = True,only_inputs=True)[0]
        # grad p
        p_xc = torch.autograd.grad(p, x_c, grad_outputs=torch.ones_like(x_c),create_graph = True,only_inputs=True)[0]
        p_xs = torch.autograd.grad(p, x_s, grad_outputs=torch.ones_like(x_s),create_graph = True,only_inputs=True)[0]
        p_yc = torch.autograd.grad(p, y_c, grad_outputs=torch.ones_like(y_c),create_graph = True,only_inputs=True)[0]
        p_ys = torch.autograd.grad(p, y_s, grad_outputs=torch.ones_like(y_s),create_graph = True,only_inputs=True)[0]
        p_zc = torch.autograd.grad(p, z_c, grad_outputs=torch.ones_like(z_c),create_graph = True,only_inputs=True)[0]
        p_zs = torch.autograd.grad(p, z_s, grad_outputs=torch.ones_like(z_s),create_graph = True,only_inputs=True)[0]
        
        # defining physical derivatives
        ux_x = 2*numpy.pi*Bc*(-ux_xc*x_s + ux_xs*x_c)
        ux_y = 2*numpy.pi*Bc*(-ux_yc*y_s + ux_ys*y_c)
        ux_z = 2*numpy.pi*Bc*(-ux_zc*z_s + ux_zs*z_c)
        uy_x = 2*numpy.pi*Bc*(-uy_xc*x_s + uy_xs*x_c)
        uy_y = 2*numpy.pi*Bc*(-uy_yc*y_s + uy_ys*y_c)
        uy_z = 2*numpy.pi*Bc*(-uy_zc*z_s + uy_zs*z_c)
        uz_x = 2*numpy.pi*Bc*(-uz_xc*x_s + uz_xs*x_c)
        uz_y = 2*numpy.pi*Bc*(-uz_yc*y_s + uz_ys*y_c)
        uz_z = 2*numpy.pi*Bc*(-uz_zc*z_s + uz_zs*z_c)

        ux_xx = 4*numpy.pi**2*Bc*Bc*(ux_xcc*x_s*x_s - 2*ux_xcs*x_c*x_s + ux_xss*x_c*x_c - ux_xc*x_c - ux_xs*x_s)
        ux_yy = 4*numpy.pi**2*Bc*Bc*(ux_ycc*y_s*y_s - 2*ux_ycs*y_c*y_s + ux_yss*y_c*y_c - ux_yc*y_c - ux_ys*y_s)
        ux_zz = 4*numpy.pi**2*Bc*Bc*(ux_zcc*z_s*z_s - 2*ux_zcs*z_c*z_s + ux_zss*z_c*z_c - ux_zc*z_c - ux_zs*z_s)
        uy_xx = 4*numpy.pi**2*Bc*Bc*(uy_xcc*x_s*x_s - 2*uy_xcs*x_c*x_s + uy_xss*x_c*x_c - uy_xc*x_c - uy_xs*x_s)
        uy_yy = 4*numpy.pi**2*Bc*Bc*(uy_ycc*y_s*y_s - 2*uy_ycs*y_c*y_s + uy_yss*y_c*y_c - uy_yc*y_c - uy_ys*y_s)
        uy_zz = 4*numpy.pi**2*Bc*Bc*(uy_zcc*z_s*z_s - 2*uy_zcs*z_c*z_s + uy_zss*z_c*z_c - uy_zc*z_c - uy_zs*z_s)
        uz_xx = 4*numpy.pi**2*Bc*Bc*(uz_xcc*x_s*x_s - 2*uz_xcs*x_c*x_s + uz_xss*x_c*x_c - uz_xc*x_c - uz_xs*x_s)
        uz_yy = 4*numpy.pi**2*Bc*Bc*(uz_ycc*y_s*y_s - 2*uz_ycs*y_c*y_s + uz_yss*y_c*y_c - uz_yc*y_c - uz_ys*y_s)
        uz_zz = 4*numpy.pi**2*Bc*Bc*(uz_zcc*z_s*z_s - 2*uz_zcs*z_c*z_s + uz_zss*z_c*z_c - uz_zc*z_c - uz_zs*z_s)
        
        p_x = 2*numpy.pi*Bc*(-p_xc*x_s + p_xs*x_c)
        p_y = 2*numpy.pi*Bc*(-p_yc*y_s + p_ys*y_c)
        p_z = 2*numpy.pi*Bc*(-p_zc*z_s + p_zs*z_c)


        Re = Factors['Re']

        loss_1 = ux_t + ux*ux_x + uy*ux_y + uz*ux_z - (1/Re)*( ux_xx + ux_yy + ux_zz) + p_x
        loss_2 = uy_t + ux*uy_x + uy*uy_y + uz*uy_z - (1/Re)*( uy_xx + uy_yy + uy_zz) + p_y
        loss_3 = uz_t + ux*uz_x + uy*uz_y + uz*uz_z - (1/Re)*( uz_xx + uz_yy + uz_zz) + p_z
        loss_4 = ux_x  + uy_y + uz_z  # continuity


        # MSE loss
        loss_f = nn.MSELoss()
        loss = loss_f(loss_1,torch.zeros_like(loss_1)) \
                + loss_f(loss_2,torch.zeros_like(loss_2)) \
                + loss_f(loss_3,torch.zeros_like(loss_3)) \
                + loss_f(loss_4,torch.zeros_like(loss_4))

        return loss

    def Loss_Phys_Div(x, y, z, t):

        t.requires_grad = True
        
        nn_in = torch.cat((x,y,z,t),1)
        
        if arch_type == '01':
            Fx = nn_ux(nn_in)
            Fy = nn_uy(nn_in)
            Fz = nn_uz(nn_in)
            p = nn_p(nn_in)
        elif arch_type == '02':
            u = nn_u(nn_in)
            Fx = u[:,0]
            Fy = u[:,1]
            Fz = u[:,2]
            p = nn_p(nn_in)
        elif arch_type in ['03','04']:
            up = nn_all(nn_in)
            Fx = up[:,0]
            Fy = up[:,1]
            Fz = up[:,2]
            p = up[:,3]
            
        Fx = Fx.view(len(Fx),-1)
        Fy = Fy.view(len(Fy),-1)
        Fz = Fz.view(len(Fz),-1)
        p = p.view(len(p),-1)

        # computing derivatives
        # x
        Fx_y = torch.autograd.grad(Fx, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fx_yy = torch.autograd.grad(Fx_y, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fx_yyy = torch.autograd.grad(Fx_yy, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fx_yz = torch.autograd.grad(Fx_y, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fx_yzz = torch.autograd.grad(Fx_yz, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fx_yx = torch.autograd.grad(Fx_y, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        Fx_yxx = torch.autograd.grad(Fx_yx, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        Fx_yt = torch.autograd.grad(Fx_y, t, grad_outputs=torch.ones_like(t), create_graph = True,only_inputs=True)[0]
        Fx_z = torch.autograd.grad(Fx, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fx_zz = torch.autograd.grad(Fx_z, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fx_zzz = torch.autograd.grad(Fx_zz, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fx_zy = torch.autograd.grad(Fx_z, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fx_zyy = torch.autograd.grad(Fx_zy, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fx_zx = torch.autograd.grad(Fx_z, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        Fx_zxx = torch.autograd.grad(Fx_zx, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        Fx_zt = torch.autograd.grad(Fx_z, t, grad_outputs=torch.ones_like(t), create_graph = True,only_inputs=True)[0]
        # y
        Fy_x = torch.autograd.grad(Fy, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        Fy_xy = torch.autograd.grad(Fy_x, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fy_xyy = torch.autograd.grad(Fy_xy, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fy_xz = torch.autograd.grad(Fy_x, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fy_xzz = torch.autograd.grad(Fy_xz, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fy_xx = torch.autograd.grad(Fy_x, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        Fy_xxx = torch.autograd.grad(Fy_xx, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        Fy_xt = torch.autograd.grad(Fy_x, t, grad_outputs=torch.ones_like(t), create_graph = True,only_inputs=True)[0]
        Fy_z = torch.autograd.grad(Fy, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fy_zz = torch.autograd.grad(Fy_z, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fy_zzz = torch.autograd.grad(Fy_zz, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fy_zy = torch.autograd.grad(Fy_z, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fy_zyy = torch.autograd.grad(Fy_zy, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fy_zx = torch.autograd.grad(Fy_z, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        Fy_zxx = torch.autograd.grad(Fy_zx, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        Fy_zt = torch.autograd.grad(Fy_z, t, grad_outputs=torch.ones_like(t), create_graph = True,only_inputs=True)[0]
        # z
        Fz_x = torch.autograd.grad(Fz, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        Fz_xz = torch.autograd.grad(Fz_x, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fz_xzz = torch.autograd.grad(Fz_xz, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fz_xy = torch.autograd.grad(Fz_x, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fz_xyy = torch.autograd.grad(Fz_xy, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fz_xx = torch.autograd.grad(Fz_x, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        Fz_xxx = torch.autograd.grad(Fz_xx, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        Fz_xt = torch.autograd.grad(Fz_x, t, grad_outputs=torch.ones_like(t), create_graph = True,only_inputs=True)[0]
        Fz_y = torch.autograd.grad(Fz, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fz_yz = torch.autograd.grad(Fz_y, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fz_yzz = torch.autograd.grad(Fz_yz, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fz_yy = torch.autograd.grad(Fz_y, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fz_yyy = torch.autograd.grad(Fz_yy, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fz_yx = torch.autograd.grad(Fz_y, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        Fz_yxx = torch.autograd.grad(Fz_yx, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        Fz_yt = torch.autograd.grad(Fz_y, t, grad_outputs=torch.ones_like(t), create_graph = True,only_inputs=True)[0]
        # grad p
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        p_z = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]


        Re = Factors['Re']

        loss_1 = (Fz_yt - Fy_zt) + \
                    (Fz_y - Fy_z)*(Fz_yx - Fy_zx) + \
                    (Fx_z - Fz_x)*(Fz_yy - Fy_zy) + \
                    (Fy_x - Fx_y)*(Fz_yz - Fy_zz) - \
                    (1/Re)*(Fz_yxx - Fy_zxx) - \
                    (1/Re)*(Fz_yyy - Fy_zyy) - \
                    (1/Re)*(Fz_yzz - Fy_zzz) + \
                    p_x

        loss_2 = (Fx_zt - Fz_xt) + \
                    (Fz_y - Fy_z)*(Fx_zx - Fz_xx) + \
                    (Fx_z - Fz_x)*(Fx_zy - Fz_xy) + \
                    (Fy_x - Fx_y)*(Fx_zz - Fz_xz) - \
                    (1/Re)*(Fx_zxx - Fz_xxx) - \
                    (1/Re)*(Fx_zyy - Fz_xyy) - \
                    (1/Re)*(Fx_zzz - Fz_xzz) + \
                    p_y

        loss_3 = (Fy_xt - Fx_yt) + \
                    (Fz_y - Fy_z)*(Fy_xx - Fx_yx) + \
                    (Fx_z - Fz_x)*(Fy_xy - Fx_yy) + \
                    (Fy_x - Fx_y)*(Fy_xz - Fx_yz) - \
                    (1/Re)*(Fy_xxx - Fx_yxx) - \
                    (1/Re)*(Fy_xyy - Fx_yyy) - \
                    (1/Re)*(Fy_xzz - Fx_yzz) + \
                    p_z

        # MSE loss
        loss_f = nn.MSELoss()

        loss = loss_f(loss_1,torch.zeros_like(loss_1)) \
                + loss_f(loss_2,torch.zeros_like(loss_2)) \
                + loss_f(loss_3,torch.zeros_like(loss_3))

        return loss

    def Loss_Data():

        loss_f = nn.MSELoss()
        loss = 0

        #ux = Fz_y - Fy_z
        #uy = Fx_z - Fz_x
        #uz = Fy_x - Fx_y

        for k in DATA['x'].keys():
            for l,tk in enumerate(times_meas):
                if not is_Fourier_features:
                    nn_in = torch.cat((xd[k], yd[k], zd[k], tk[0]*torch.ones_like(xd[k])), 1)
                else:
                    nn_in = torch.cat((xd_c[k], xd_s[k], yd_c[k], yd_s[k], zd_c[k], zd_s[k], tk[0]*torch.ones_like(xd_c[k])), 1)

                if arch_type == '01':
                    ux = nn_ux(nn_in)
                    uy = nn_uy(nn_in)
                    uz = nn_uz(nn_in)
                elif arch_type == '02':
                    u = nn_u(nn_in)
                    ux = u[:,0]
                    uy = u[:,1]
                    uz = u[:,2]
                elif arch_type in ['03','04']:
                    up = nn_all(nn_in)
                    ux = up[:,0]
                    uy = up[:,1]
                    uz = up[:,2]
                    
                ux = ux.view(len(ux), -1)
                uy = uy.view(len(uy), -1)
                uz = uz.view(len(uz), -1)
                indx_a = l*len(ux)
                indx_b = indx_a + len(ux)

                if not divergence_free:
                    loss += loss_f(ux, uxd[k][indx_a:indx_b]) + loss_f(uy, uyd[k][indx_a:indx_b]) + loss_f(uz, uzd[k][indx_a:indx_b])

                else:
                    Fy_z = torch.autograd.grad(uy, zd[k], grad_outputs=torch.ones_like(zd[k]), create_graph = True,only_inputs=True)[0]
                    Fz_y = torch.autograd.grad(uz, yd[k], grad_outputs=torch.ones_like(yd[k]), create_graph = True,only_inputs=True)[0]
                    ux2 = Fz_y - Fy_z
                    del Fy_z, Fz_y
                    Fz_x = torch.autograd.grad(uz, xd[k], grad_outputs=torch.ones_like(xd[k]), create_graph = True,only_inputs=True)[0]
                    Fx_z = torch.autograd.grad(ux, zd[k], grad_outputs=torch.ones_like(zd[k]), create_graph = True,only_inputs=True)[0]
                    uy2 = Fx_z - Fz_x
                    del Fz_x, Fx_z
                    Fy_x = torch.autograd.grad(uy, xd[k], grad_outputs=torch.ones_like(xd[k]), create_graph = True,only_inputs=True)[0]
                    Fx_y = torch.autograd.grad(ux, yd[k], grad_outputs=torch.ones_like(yd[k]), create_graph = True,only_inputs=True)[0]
                    uz2 = Fy_x - Fx_y
                    del Fy_x, Fx_y

                    loss += loss_f(ux2, uxd[k][indx_a:indx_b]) + loss_f(uy2, uyd[k][indx_a:indx_b]) + loss_f(uz2, uzd[k][indx_a:indx_b])

        return loss

    def Loss_BC():
        
        loss_f = nn.MSELoss()
        loss = 0
        if not is_Fourier_features:
            nn_in = torch.cat((xb, yb, zb, tb), 1)
        else:
            nn_in = torch.cat((xb_c, xb_s, yb_c, yb_s, zb_c, zb_s, tb), 1)

        if arch_type == '01':
            ux = nn_ux(nn_in)
            uy = nn_uy(nn_in)
            uz = nn_uz(nn_in)
        elif arch_type == '02':
            u = nn_u(nn_in)
            ux = u[:,0]
            uy = u[:,1]
            uz = u[:,2]
        elif arch_type in ['03','04']:
            up = nn_all(nn_in)
            ux = up[:,0]
            uy = up[:,1]
            uz = up[:,2]
            
        ux = ux.view(len(ux), -1)
        uy = uy.view(len(uy), -1)
        uz = uz.view(len(uz), -1)
        
        if not divergence_free:
            loss +=  loss_f(ux, torch.zeros_like(ux)) + \
                    loss_f(uy, torch.zeros_like(uy)) + \
                    loss_f(uz, torch.zeros_like(uz))
        else:
            Fz_y = torch.autograd.grad(uz, yb, grad_outputs=torch.ones_like(yb), create_graph = True,only_inputs=True)[0]
            Fy_z = torch.autograd.grad(uy, zb, grad_outputs=torch.ones_like(zb), create_graph = True,only_inputs=True)[0]
            uxb = Fz_y - Fy_z
            del Fz_y, Fy_z
            Fx_z = torch.autograd.grad(ux, zb, grad_outputs=torch.ones_like(zb), create_graph = True,only_inputs=True)[0]
            Fz_x = torch.autograd.grad(uz, xb, grad_outputs=torch.ones_like(xb), create_graph = True,only_inputs=True)[0]
            uyb = Fx_z - Fz_x
            del Fx_z, Fz_x
            Fy_x = torch.autograd.grad(uy, xb, grad_outputs=torch.ones_like(xb), create_graph = True,only_inputs=True)[0]
            Fx_y = torch.autograd.grad(ux, yb, grad_outputs=torch.ones_like(yb), create_graph = True,only_inputs=True)[0]
            uzb = Fy_x - Fx_y
            del Fy_x, Fx_y
            

            loss += loss_f(uxb, torch.zeros_like(uxb)) + \
                    loss_f(uyb, torch.zeros_like(uyb)) + \
                    loss_f(uzb, torch.zeros_like(uzb))

        return loss

    def Loss_WK(wk_lst, estim_bnds):
        
        loss = 0
        loss_initial = 0
        loss_gradp = 0
        loss_pit_reg = 0
        loss_f = nn.MSELoss()
        qfac = 1/(Factors['U']*Factors['L']**2)

        for lk, k in enumerate(wk_lst):
            if not is_Fourier_features:
                nn_in = torch.cat((xwk[k], ywk[k], zwk[k], twk[k]), 1)
            else:
                nn_in = torch.cat((xwk_c[k], xwk_s[k], ywk_c[k], ywk_s[k], zwk_c[k], zwk_s[k], twk[k]), 1)

            if arch_type in ['01','02']:
                pwk = nn_p(nn_in)
            else:
                up = nn_all(nn_in)
                pwk = up[:,3]

            pwk = pwk.view(len(pwk),-1)
            # estimating pi
            pi = nns_pi[lk](twk[k])
            pi = pi.view(len(pi),-1)
            # defining Q
            Qk = torch.zeros_like(pwk)

            for ti, qq in enumerate(Qk):
                qq[0] = torch.tensor(qfac*Q_interpolator[k](twk[k][ti].item()))

            
            # computing dpi/dt
            pi_t = torch.autograd.grad(pi, twk[k], grad_outputs=torch.ones_like(twk[k]), create_graph = True, only_inputs=True)[0]


            if k in estim_bnds:
                if (k in C_i.keys()) and (k in Rd_i.keys()) and (k in Rp_i.keys()):
                    if not param_factorization:
                        loss_1 = C_i[k]()*pi_t + (1/Rd_i[k]())*pi - Qk
                        loss_2 = pwk - pi - Rp_i[k]()*Qk
                    else:
                        loss_1 = torch.exp(C_i[k]())*pi_t + (1/torch.exp(Rd_i[k]()))*pi - Qk
                        loss_2 = pwk - pi - torch.exp(Rp_i[k]())*Qk
                    RC_factor = Rd_i[k]()*C_i[k]()
                elif (k not in C_i.keys()) and (k in Rd_i.keys()) and (k in Rp_i.keys()):
                    if not param_factorization:
                        loss_1 = C_ref[k]*pi_t + (1/Rd_i[k]())*pi - Qk
                        loss_2 = pwk - pi - Rp_i[k]()*Qk
                    else:
                        loss_1 = C_ref[k]*pi_t + (1/torch.exp(Rd_i[k]()))*pi - Qk
                        loss_2 = pwk - pi - torch.exp(Rp_i[k]())*Qk
                    RC_factor = Rd_i[k]()*C_ref[k]
                elif (k not in C_i.keys()) and (k not in Rd_i.keys()) and (k in Rp_i.keys()):
                    if not param_factorization:
                        loss_1 = C_ref[k]*pi_t + (1/Rd_ref[k])*pi - Qk
                        loss_2 = pwk - pi - Rp_i[k]()*Qk
                    else:
                        loss_1 = C_ref[k]*pi_t + (1/Rd_ref[k])*pi - Qk
                        loss_2 = pwk - pi - torch.exp(Rp_i[k]())*Qk   
                    RC_factor = Rd_ref[k]*C_ref[k]
                elif (k in C_i.keys()) and (k in Rd_i.keys()) and (k not in Rp_i.keys()):
                    if not param_factorization:
                        loss_1 = C_i[k]()*pi_t + (1/Rd_i[k]())*pi - Qk
                        loss_2 = pwk - pi - Rp_ref[k]*Qk
                    else:
                        loss_1 = torch.exp(C_i[k]())*pi_t + (1/torch.exp(Rd_i[k]()))*pi - Qk
                        loss_2 = pwk - pi - Rp_ref[k]*Qk
                    RC_factor = Rd_i[k]()*C_i[k]()
                elif (k not in C_i.keys()) and (k in Rd_i.keys()) and (k not in Rp_i.keys()):
                    if not param_factorization:
                        loss_1 = C_ref[k]*pi_t + (1/Rd_i[k]())*pi - Qk
                        loss_2 = pwk - pi - Rp_ref[k]*Qk
                    else:
                        loss_1 = C_ref[k]*pi_t + (1/torch.exp(Rd_i[k]()))*pi - Qk
                        loss_2 = pwk - pi - Rp_ref[k]*Qk
                    RC_factor = Rd_i[k]()*C_ref[k]
                elif (k in C_i.keys()) and (k not in Rd_i.keys()) and (k in Rp_i.keys()):
                    if not param_factorization:
                        loss_1 = C_i[k]()*pi_t + (1/Rd_ref[k])*pi - Qk
                        loss_2 = pwk - pi - Rp_i[k]()*Qk
                    else:
                        loss_1 = torch.exp(C_i[k]())*pi_t + (1/Rd_ref[k])*pi - Qk
                        loss_2 = pwk - pi - torch.exp(Rp_i[k]())*Qk
                    RC_factor = Rd_ref[k]*C_i[k]()
                elif (k in C_i.keys()) and (k not in Rd_i.keys()) and (k not in Rp_i.keys()):
                    if not param_factorization:
                        loss_1 = C_i[k]()*pi_t + (1/Rd_ref[k])*pi - Qk
                        loss_2 = pwk - pi - Rp_ref[k]*Qk
                    else:
                        loss_1 = torch.exp(C_i[k]())*pi_t + (1/Rd_ref[k])*pi - Qk
                        loss_2 = pwk - pi - Rp_ref[k]*Qk
                    RC_factor = Rd_ref[k]*C_i[k]()
            else:
                loss_1 = C_ref[k]*pi_t + (1/Rd_ref[k])*pi - Qk
                loss_2 = pwk - pi - Rp_ref[k]*Qk
                RC_factor = Rd_ref[k]*C_ref[k]
            

            loss += loss_f(loss_1,torch.zeros_like(loss_1))
            loss += loss_f(loss_2,torch.zeros_like(loss_2))

            if adding_pit_reg:
                if k in estim_bnds and k in C_i.keys():
                    Cfac = C_i[k]()
                else:
                    Cfac = C_ref[k]
                
                upper_bound = torch.abs(torch.max(Qk/Cfac - pi/(RC_factor)))

                if torch.max(torch.abs(pi_t)) > upper_bound:
                    loss_pit_reg += loss_f(pi_t, torch.zeros_like(pi_t))


            # penalizing pressure gradients within the outlet
            if not is_Fourier_features:
                p_x = torch.autograd.grad(pwk, xwk[k], grad_outputs=torch.ones_like(xwk[k]), create_graph = True,only_inputs=True)[0]
                p_y = torch.autograd.grad(pwk, ywk[k], grad_outputs=torch.ones_like(ywk[k]), create_graph = True,only_inputs=True)[0]
                p_z = torch.autograd.grad(pwk, zwk[k], grad_outputs=torch.ones_like(zwk[k]), create_graph = True,only_inputs=True)[0]
            else:
                p_xc = torch.autograd.grad(pwk, xwk_c[k], grad_outputs=torch.ones_like(xwk_c[k]), create_graph = True,only_inputs=True)[0]
                p_xs = torch.autograd.grad(pwk, xwk_s[k], grad_outputs=torch.ones_like(xwk_s[k]), create_graph = True,only_inputs=True)[0]
                p_yc = torch.autograd.grad(pwk, ywk_c[k], grad_outputs=torch.ones_like(ywk_c[k]), create_graph = True,only_inputs=True)[0]
                p_ys = torch.autograd.grad(pwk, ywk_s[k], grad_outputs=torch.ones_like(ywk_s[k]), create_graph = True,only_inputs=True)[0]                
                p_zc = torch.autograd.grad(pwk, zwk_c[k], grad_outputs=torch.ones_like(zwk_c[k]), create_graph = True,only_inputs=True)[0]
                p_zs = torch.autograd.grad(pwk, zwk_s[k], grad_outputs=torch.ones_like(zwk_s[k]), create_graph = True,only_inputs=True)[0]
                p_x = 2*numpy.pi*Bw[k]*(-p_xc*xwk_s[k] + p_xs*xwk_c[k])
                p_y = 2*numpy.pi*Bw[k]*(-p_yc*ywk_s[k] + p_ys*ywk_c[k])
                p_z = 2*numpy.pi*Bw[k]*(-p_zc*zwk_s[k] + p_zs*zwk_c[k])
                

            loss_gradp += loss_f(p_x,torch.zeros_like(p_x))
            loss_gradp += loss_f(p_y,torch.zeros_like(p_y))
            loss_gradp += loss_f(p_z,torch.zeros_like(p_z))
            
            # adding initial condition for pi functions and its derivative
            if adding_pi0:
                t0 = torch.Tensor([0]).to(device)
                t0.requires_grad = True
                pi0 = nns_pi[lk](t0)
                pi0 = pi0.view(len(pi0),-1)

                loss_initial += loss_f(pi0, pwk_mean[0]*torch.ones_like(pi0))


            if adding_Pt0:
                #   forcing initial pressure value at outlets
                if not is_Fourier_features:
                    nn_in = torch.cat((xwk[k], ywk[k], zwk[k], torch.zeros_like(xwk[k])), 1)
                else:
                    nn_in = torch.cat((xwk_c[k], xwk_s[k], ywk_c[k], ywk_s[k], zwk_c[k], zwk_s[k], torch.zeros_like(xwk_c[k])), 1)

                if arch_type in ['01','02']:
                    pwk0 = nn_p(nn_in)
                else:
                    up = nn_all(nn_in)
                    pwk0 = up[:,3]
                
                pwk0 = pwk0.view(len(pwk0),-1)

                loss_initial += loss_f(pwk0, pwk_mean[0]*torch.ones_like(pwk0))
            
        if adding_pi0:
            return loss, loss_initial, loss_gradp, loss_pit_reg
        else:
            return loss, loss_gradp, loss_pit_reg

    def Loss_PMean():

        loss_f = nn.MSELoss()
        mean_pressure = torch.zeros_like(pwk_mean)


        for k in windkessel_bnd:
            for l,tk in enumerate(times_meas):
                if not is_Fourier_features:
                    nn_in = torch.cat((xwk[k], ywk[k], zwk[k], tk[0]*torch.ones_like(xwk[k])), 1)
                else:
                    nn_in = torch.cat((xwk_c[k], xwk_s[k], ywk_c[k], ywk_s[k], zwk_c[k], zwk_s[k], tk[0]*torch.ones_like(xwk_c[k])), 1)

                if arch_type in ['01','02']:
                    pwk = nn_p(nn_in)
                else:
                    up = nn_all(nn_in)
                    pwk = up[:,3]

                pwk = pwk.view(len(pwk),-1)

                mean_pressure[l] += torch.mean(pwk)/len(windkessel_bnd)

        loss = loss_f(mean_pressure, pwk_mean)

        return loss

    def Loss_RangeParams():
        
        loss_f = nn.MSELoss()
        aux = torch.tensor([]).to(device)

        for k in Rd_i.keys():
            if Rd_i[k].R < Rd_range_min[k].R:
                aux = torch.cat((aux, Rd_range_min[k].R - Rd_i[k].R),-1)
            elif Rd_i[k].R > Rd_range_max[k].R:
                aux = torch.cat((aux, Rd_i[k].R - Rd_range_max[k].R),-1)
        for k in Rp_i.keys():
            if Rp_i[k].R < Rp_range_min[k].R:
                aux = torch.cat((aux, Rp_range_min[k].R - Rp_i[k].R),-1)
            elif Rp_i[k].R > Rp_range_max[k].R:
                aux = torch.cat((aux, Rp_i[k].R - Rp_range_max[k].R),-1)
        for k in C_i.keys():
            if C_i[k].C < C_range_min[k].C:
                aux = torch.cat((aux, C_range_min[k].C - C_i[k].C),-1)
            elif C_i[k].C > C_range_max[k].C:
                aux = torch.cat((aux, C_i[k].C - C_range_max[k].C),-1)


        if len(aux) > 0:
            return loss_f(aux,torch.zeros_like(aux))
        else:
            return False

    #if using_wandb:
    #    wandb.watch(nn_all, log='all')
        
    def SaveModel():
        if arch_type == '01':
            torch.save(nn_ux.state_dict(), options['io']['write_path'] + 'model/sten_data_ux.pt')
            torch.save(nn_uy.state_dict(), options['io']['write_path'] + 'model/sten_data_uy.pt')
            torch.save(nn_uz.state_dict(), options['io']['write_path'] + 'model/sten_data_uz.pt')
            torch.save(nn_p.state_dict(), options['io']['write_path'] + 'model/sten_data_p.pt')
        elif arch_type == '02':
            torch.save(nn_u.state_dict(), options['io']['write_path'] + 'model/sten_data_u.pt')
            torch.save(nn_p.state_dict(), options['io']['write_path'] + 'model/sten_data_p.pt')
        elif arch_type in ['03','04']:
            torch.save(nn_all.state_dict(), options['io']['write_path'] + 'model/sten_data_all.pt')
        
        for l,nn_pi in enumerate(nns_pi):
            torch.save(nn_pi.state_dict(), options['io']['write_path'] + 'model/sten_data_pi_' + str(windkessel_bnd[l])+'.pt')
        

    last_batch = len(dataloader)-1
    ########## defining schedulers ##########
    scheduler_pi_lst = []

    if HP['learning_rates']['state']['scheduler']:
        logging.info('initializing learning schedulers...')

        sch_threshold = HP['learning_rates']['state']['threshold']
        sch_factor = HP['learning_rates']['state']['factor']
        sch_patience = HP['learning_rates']['state']['patience']
        if 'given_epochs' in HP['learning_rates']['state']:
            sch_given_epochs = HP['learning_rates']['state']['given_epochs']
        else:
            sch_given_epochs = []
        if 'lmin' in HP['learning_rates']['state']:
            sch_lmin = HP['learning_rates']['state']['lmin']
        else:
            sch_lmin = 0
        lr = HP['learning_rates']['state']['l']
        logging.info('State Scheduler: lr={} \t Threshold={} \t Factor={} \t Patience={} \t GivenEpochs={}'.format(lr,sch_threshold,sch_factor,sch_patience, sch_given_epochs))
        
        if not inverse_problem:
            if_verbose = True
        else:
            if_verbose = False

        if arch_type == '01':
            scheduler_ux = OwnScheduler(optimizer_ux, logging, mode='min', factor=sch_factor, 
                    patience= sch_patience, min_lr = sch_lmin, given_epochs = sch_given_epochs, threshold=sch_threshold, verbose=if_verbose)
            scheduler_uy = OwnScheduler(optimizer_uy, logging, mode='min', factor=sch_factor, 
                    patience= sch_patience, min_lr = sch_lmin, given_epochs = sch_given_epochs, threshold=sch_threshold, verbose=if_verbose)
            scheduler_uz = OwnScheduler(optimizer_uz, logging, mode='min', factor=sch_factor, 
                    patience= sch_patience, min_lr = sch_lmin, given_epochs = sch_given_epochs, threshold=sch_threshold, verbose=if_verbose)
        elif arch_type == '02':
            scheduler_u = OwnScheduler(optimizer_u, logging, mode='min', factor=sch_factor, 
                    patience= sch_patience, min_lr = sch_lmin, given_epochs = sch_given_epochs, threshold=sch_threshold, verbose=if_verbose)
        elif arch_type in ['03','04']:
            scheduler_all = OwnScheduler(optimizer_all, logging, mode='min', factor=sch_factor, 
                    patience= sch_patience, min_lr = sch_lmin, given_epochs = sch_given_epochs, threshold=sch_threshold, verbose=if_verbose)
    
        for opt in optimizer_pi_lst:
            scheduler_pi_lst.append(OwnScheduler(opt, logging, mode='min', factor=sch_factor, 
                    patience= sch_patience, min_lr = sch_lmin, given_epochs = sch_given_epochs, threshold=sch_threshold))
    
    if HP['learning_rates']['pressure']['scheduler']:

        sch_threshold_p = HP['learning_rates']['pressure']['threshold']
        sch_factor_p = HP['learning_rates']['pressure']['factor']
        sch_patience_p = HP['learning_rates']['pressure']['patience']
        if 'given_epochs' in HP['learning_rates']['pressure']:
            sch_given_epochs = HP['learning_rates']['pressure']['given_epochs']
        else:
            sch_given_epochs = []
        if 'lmin' in HP['learning_rates']['pressure']:
            sch_lmin_p = HP['learning_rates']['pressure']['lmin']
        else:
            sch_lmin_p = 0
        lr = HP['learning_rates']['pressure']['l']
        logging.info('Pressure Scheduler: lr={} \t Threshold={} \t Factor={} \t Patience={} \t GivenEpochs={}'.format(lr,sch_threshold_p,sch_factor_p,sch_patience_p, sch_given_epochs))

        if arch_type in ['01','02']:
            scheduler_p = OwnScheduler(optimizer_p, logging, mode='min', factor=schp_factor, 
                        patience=schp_patience, min_lr = sch_lmin_p, given_epochs = sch_given_epochs, threshold=schp_threshold)
        elif arch_type in ['03','04']:
            logging.info('scheduler for pressure is inside state scheduler')

    if HP['learning_rates']['params']['scheduler']:
        if inverse_problem:
            scheduler_params_Rd = []
            scheduler_params_Rp = []
            scheduler_params_C = []
            schp_threshold = HP['learning_rates']['params']['threshold']
            schp_factor = HP['learning_rates']['params']['factor']
            schp_patience = HP['learning_rates']['params']['patience']
            lr = HP['learning_rates']['params']['l']
            if 'given_epochs' in HP['learning_rates']['params']:
                sch_given_epochs = HP['learning_rates']['params']['given_epochs']
            else:
                sch_given_epochs = []
            if 'lmin' in HP['learning_rates']['params']:
                sch_lmin = HP['learning_rates']['params']['lmin']
            else:
                sch_lmin = 0
            logging.info('Inverse Scheduler: lr={} \t Threshold={} \t Factor={} \t Patience={}\t GivenEpochs={}'.format(lr, schp_threshold,schp_factor,schp_patience, sch_given_epochs))

            for opt in optimizer_Rd_lst:
                scheduler_params_Rd.append(OwnScheduler(opt, logging, mode='min', factor=schp_factor, 
                        patience=schp_patience, min_lr = sch_lmin, given_epochs = sch_given_epochs, threshold=schp_threshold ,verbose=True))
            for opt in optimizer_Rp_lst:
                scheduler_params_Rp.append(OwnScheduler(opt, logging, mode='min', factor=schp_factor, 
                        patience=schp_patience, min_lr = sch_lmin, given_epochs = sch_given_epochs, threshold=schp_threshold ,verbose=True))
            for opt in optimizer_C_lst:
                scheduler_params_C.append(OwnScheduler(opt, logging, mode='min', factor=schp_factor, 
                        patience=schp_patience, min_lr = sch_lmin, given_epochs = sch_given_epochs, threshold=schp_threshold ,verbose=True))

    # init annealing
    if is_lambda_annealing:
        HP['lambdas']['phys'] = 1
        HP['lambdas']['data'] = 1
        HP['lambdas']['BC'] = 1
        HP['lambdas']['windkessel'] = 1
        HP['lambdas']['initial'] = 1
        HP['lambdas']['gradp'] = 1
        HP['lambdas']['pmean'] = 1
        PrintLambdaAnnealing(HP['lambdas'])

    logging.info('Starting Traning Iterations')
    logging.info('Allocated GPU Memory: {} GB'.format(numpy.round(torch.cuda.memory_allocated(device)/1e+6,2)))


    #################
    ##  main loop  ##
    #################
    tic = time.time()

    for epoch in range(HP['epochs']):

        for batch_idx, data_batch in enumerate(dataloader):
            if not is_Fourier_features:
                x_in, y_in, z_in, t_in = data_batch
            else:
                xc_in, xs_in, yc_in, ys_in, zc_in, zs_in, B_in, t_in = data_batch

            # setting the gradients to zero
            if arch_type == '01':
                nn_ux.zero_grad()
                nn_uy.zero_grad()
                nn_uz.zero_grad()
                nn_p.zero_grad()
            elif arch_type == '02':
                nn_u.zero_grad()
                nn_p.zero_grad()
            elif arch_type in ['03','04']:
                nn_all.zero_grad()

            for nn_pi in nns_pi:
                nn_pi.zero_grad()
                
            # computing the losses
            if not is_Fourier_features:
                if not divergence_free:
                    loss_phys = Loss_Phys(x_in, y_in, z_in, t_in)
                else:
                    loss_phys = Loss_Phys_Div(x_in, y_in, z_in, t_in)
            else:
                loss_phys = Loss_Phys_Fourier(xc_in, xs_in, yc_in, ys_in, zc_in, zs_in, B_in, t_in)

            loss_bc = Loss_BC()
            loss_data = Loss_Data()
            if adding_pi0:
                loss_wk, loss_initial, loss_gradp, loss_pit_reg = Loss_WK(windkessel_bnd, estim_bnds)
            else:
                loss_wk, loss_gradp, loss_pit_reg = Loss_WK(windkessel_bnd, estim_bnds)

            if give_mean_pressure:
                loss_p = Loss_PMean()
            
            if batch_idx == last_batch:
                if epoch == HP['epochs']-1 or epoch%annealing_iter_rate == 0:
                    if is_tracking_gradients or is_lambda_annealing:
                        # l phys
                        loss_phys.backward(retain_graph=True)
                        lphys_stats = SaveGradients(l_network_1, tracking_gradients_dict['phys'])
                        l_network_1.zero_grad()
                        # l bc
                        loss_bc.backward(retain_graph=True)
                        lbc_stats = SaveGradients(l_network_1, tracking_gradients_dict['bc'])
                        l_network_1.zero_grad()
                        # l data
                        loss_data.backward(retain_graph=True)
                        ldata_stats = SaveGradients(l_network_1, tracking_gradients_dict['data'])
                        l_network_1.zero_grad()
                        # l wk
                        loss_wk.backward(retain_graph=True)
                        lwk_stats = SaveGradients(l_network_2, tracking_gradients_dict['wk'])
                        l_network_2.zero_grad()
                        # l initial
                        if adding_pi0:
                            linit_stats = [0,0,0]
                            for nn_pi in nns_pi:
                                loss_initial.backward(retain_graph=True)
                                linit_stats_k = SaveGradients(nn_pi, tracking_gradients_dict['initial'])
                                nn_pi.zero_grad()
                                linit_stats[0] += linit_stats_k[0]/len(nns_pi)
                                linit_stats[1] += linit_stats_k[1]/len(nns_pi)
                                linit_stats[2] += linit_stats_k[2]/len(nns_pi)
                            
                        # l gradp
                        loss_gradp.backward(retain_graph=True)
                        lgradp_stats = SaveGradients(l_network_2, tracking_gradients_dict['gradp'])
                        l_network_2.zero_grad()
                        # l pmean
                        if give_mean_pressure:
                            loss_p.backward(retain_graph=True)
                            lpmean_stats = SaveGradients(l_network_2, tracking_gradients_dict['pmean'])
                            l_network_2.zero_grad()
                        
    
                    if is_lambda_annealing:
                        if annealing_mode == 'max/mean':
                            HP['lambdas']['phys'] = (1-alpha_annealing)*HP['lambdas']['phys'] + alpha_annealing*lphys_stats[1]/lphys_stats[0]
                            HP['lambdas']['data'] = (1-alpha_annealing)*HP['lambdas']['data'] + alpha_annealing*ldata_stats[1]/ldata_stats[0]
                            HP['lambdas']['BC'] = (1-alpha_annealing)*HP['lambdas']['BC'] + alpha_annealing*lbc_stats[1]/lbc_stats[0]
                            HP['lambdas']['windkessel'] = (1-alpha_annealing)*HP['lambdas']['windkessel'] + alpha_annealing*lwk_stats[1]/lwk_stats[0]
                            if adding_pi0:
                                HP['lambdas']['initial'] = (1-alpha_annealing)*HP['lambdas']['initial'] + alpha_annealing*linit_stats[1]/linit_stats[0]
                            HP['lambdas']['gradp'] = (1-alpha_annealing)*HP['lambdas']['gradp'] + alpha_annealing*lgradp_stats[1]/lgradp_stats[0]
                            if give_mean_pressure:
                                HP['lambdas']['pmean'] = (1-alpha_annealing)*HP['lambdas']['pmean'] + alpha_annealing*lpmean_stats[1]/lpmean_stats[0]
                        elif annealing_mode == 'data/mean':
                            HP['lambdas']['phys'] = (1-alpha_annealing)*HP['lambdas']['phys'] + alpha_annealing*ldata_stats[1]/lphys_stats[0]
                            HP['lambdas']['data'] = (1-alpha_annealing)*HP['lambdas']['data'] + alpha_annealing*ldata_stats[1]/ldata_stats[0]
                            HP['lambdas']['BC'] = (1-alpha_annealing)*HP['lambdas']['BC'] + alpha_annealing*ldata_stats[1]/lbc_stats[0]
                            HP['lambdas']['windkessel'] = (1-alpha_annealing)*HP['lambdas']['windkessel'] + alpha_annealing*ldata_stats[1]/lwk_stats[0]
                            if adding_pi0:
                                HP['lambdas']['initial'] = (1-alpha_annealing)*HP['lambdas']['initial'] + alpha_annealing*ldata_stats[1]/linit_stats[0]
                            HP['lambdas']['gradp'] = (1-alpha_annealing)*HP['lambdas']['gradp'] + alpha_annealing*ldata_stats[1]/lgradp_stats[0]
                            if give_mean_pressure:
                                HP['lambdas']['pmean'] = (1-alpha_annealing)*HP['lambdas']['pmean'] + alpha_annealing*ldata_stats[1]/lpmean_stats[0]
                        elif annealing_mode == 'phys/mean':
                            HP['lambdas']['phys'] = (1-alpha_annealing)*HP['lambdas']['phys'] + alpha_annealing*lphys_stats[1]/lphys_stats[0]
                            HP['lambdas']['data'] = (1-alpha_annealing)*HP['lambdas']['data'] + alpha_annealing*lphys_stats[1]/ldata_stats[0]
                            HP['lambdas']['BC'] = (1-alpha_annealing)*HP['lambdas']['BC'] + alpha_annealing*lphys_stats[1]/lbc_stats[0]
                            HP['lambdas']['windkessel'] = (1-alpha_annealing)*HP['lambdas']['windkessel'] + alpha_annealing*lphys_stats[1]/lwk_stats[0]
                            if adding_pi0:
                                HP['lambdas']['initial'] = (1-alpha_annealing)*HP['lambdas']['initial'] + alpha_annealing*lphys_stats[1]/linit_stats[0]
                            HP['lambdas']['gradp'] = (1-alpha_annealing)*HP['lambdas']['gradp'] + alpha_annealing*lphys_stats[1]/lgradp_stats[0]
                            if give_mean_pressure:
                                HP['lambdas']['pmean'] = (1-alpha_annealing)*HP['lambdas']['pmean'] + alpha_annealing*lphys_stats[1]/lpmean_stats[0]
                        elif annealing_mode == 'physrel/mean':
                            HP['lambdas']['phys'] = 1
                            HP['lambdas']['data'] = (1-alpha_annealing)*HP['lambdas']['data'] + alpha_annealing*lphys_stats[1]/ldata_stats[0]
                            HP['lambdas']['BC'] = (1-alpha_annealing)*HP['lambdas']['BC'] + alpha_annealing*lphys_stats[1]/lbc_stats[0]
                            HP['lambdas']['windkessel'] = (1-alpha_annealing)*HP['lambdas']['windkessel'] + alpha_annealing*lphys_stats[1]/lwk_stats[0]
                            if adding_pi0:
                                HP['lambdas']['initial'] = (1-alpha_annealing)*HP['lambdas']['initial'] + alpha_annealing*lphys_stats[1]/linit_stats[0]
                            HP['lambdas']['gradp'] = (1-alpha_annealing)*HP['lambdas']['gradp'] + alpha_annealing*lphys_stats[1]/lgradp_stats[0]
                            if give_mean_pressure:
                                HP['lambdas']['pmean'] = (1-alpha_annealing)*HP['lambdas']['pmean'] + alpha_annealing*lphys_stats[1]/lpmean_stats[0]
                        elif annealing_mode == 'search/mean':
                            max_grad = numpy.max([lphys_stats[1],ldata_stats[1],lbc_stats[1],lwk_stats[1],linit_stats[1],lgradp_stats[1],lpmean_stats[1]])
                            HP['lambdas']['phys'] = (1-alpha_annealing)*HP['lambdas']['phys'] + alpha_annealing*max_grad/lphys_stats[0]
                            HP['lambdas']['data'] = (1-alpha_annealing)*HP['lambdas']['data'] + alpha_annealing*max_grad/ldata_stats[0]
                            HP['lambdas']['BC'] = (1-alpha_annealing)*HP['lambdas']['BC'] + alpha_annealing*max_grad/lbc_stats[0]
                            HP['lambdas']['windkessel'] = (1-alpha_annealing)*HP['lambdas']['windkessel'] + alpha_annealing*max_grad/lwk_stats[0]
                            if adding_pi0:
                                HP['lambdas']['initial'] = (1-alpha_annealing)*HP['lambdas']['initial'] + alpha_annealing*max_grad/linit_stats[0]
                            HP['lambdas']['gradp'] = (1-alpha_annealing)*HP['lambdas']['gradp'] + alpha_annealing*max_grad/lgradp_stats[0]
                            if give_mean_pressure:
                                HP['lambdas']['pmean'] = (1-alpha_annealing)*HP['lambdas']['pmean'] + alpha_annealing*max_grad/lpmean_stats[0]
                        elif annealing_mode == 'mean/max':
                            HP['lambdas']['phys'] = (1-alpha_annealing)*HP['lambdas']['phys'] + alpha_annealing*(10+lphys_stats[0]/lphys_stats[1])
                            HP['lambdas']['data'] = (1-alpha_annealing)*HP['lambdas']['data'] + alpha_annealing*(10+ldata_stats[0]/ldata_stats[1])
                            HP['lambdas']['BC'] = (1-alpha_annealing)*HP['lambdas']['BC'] + alpha_annealing*(10+lbc_stats[0]/lbc_stats[1])
                            HP['lambdas']['windkessel'] = (1-alpha_annealing)*HP['lambdas']['windkessel'] + alpha_annealing*(10+lwk_stats[0]/lwk_stats[1])
                            if adding_pi0:
                                HP['lambdas']['initial'] = (1-alpha_annealing)*HP['lambdas']['initial'] + alpha_annealing*(10+linit_stats[0]/linit_stats[1])
                            HP['lambdas']['gradp'] = (1-alpha_annealing)*HP['lambdas']['gradp'] + alpha_annealing*(10+lgradp_stats[0]/lgradp_stats[1])
                            if give_mean_pressure:
                                HP['lambdas']['pmean'] = (1-alpha_annealing)*HP['lambdas']['pmean'] + alpha_annealing*(10+lpmean_stats[0]/lpmean_stats[1])



                        PrintLambdaAnnealing(HP['lambdas'])
                        lambda_track['phys'].append(HP['lambdas']['phys'])
                        lambda_track['data'].append(HP['lambdas']['data'])
                        lambda_track['BC'].append(HP['lambdas']['BC'])
                        lambda_track['windkessel'].append(HP['lambdas']['windkessel'])
                        if adding_pi0:
                            lambda_track['initial'].append(HP['lambdas']['initial'])
                        lambda_track['gradp'].append(HP['lambdas']['gradp'])
                        lambda_track['pmean'].append(HP['lambdas']['pmean'])
                        annealing_info['phys'].append(lphys_stats)
                        annealing_info['data'].append(ldata_stats)
                        annealing_info['BC'].append(lbc_stats)
                        annealing_info['windkessel'].append(lwk_stats)
                        if adding_pi0:
                            annealing_info['initial'].append(linit_stats)
                        annealing_info['gradp'].append(lgradp_stats)
                        annealing_info['pmean'].append(lpmean_stats)
                        
            
            loss = HP['lambdas']['phys']*loss_phys + HP['lambdas']['data']*loss_data \
                         + HP['lambdas']['BC']*loss_bc + HP['lambdas']['windkessel']*loss_wk \
                         + HP['lambdas']['gradp']*loss_gradp
            
            if give_mean_pressure:
                loss += HP['lambdas']['pmean']*loss_p

            if adding_pi0:
                loss += HP['lambdas']['initial']*loss_initial


            lmean = 0
            if inverse_problem:
                if range_distance_reg:
                    lmean = numpy.max([HP['lambdas']['phys'],HP['lambdas']['data'],HP['lambdas']['BC'],
                         HP['lambdas']['windkessel'],HP['lambdas']['initial'],HP['lambdas']['gradp'],HP['lambdas']['pmean']])
                    loss_params = Loss_RangeParams()
                    if loss_params:
                        loss += lmean*loss_params
                    
            if adding_pit_reg:
                if lmean == 0:
                    lmean = numpy.mean([HP['lambdas']['phys'],HP['lambdas']['data'],HP['lambdas']['BC'],
                         HP['lambdas']['windkessel'],HP['lambdas']['initial'],HP['lambdas']['gradp'],HP['lambdas']['pmean']])
                
                if loss_pit_reg != 0:
                    loss += lmean*loss_pit_reg
            


            loss.backward()

            ############ stepping optimizers ############
            # 01: velocity and pressure
            if arch_type == '01':
                optimizer_ux.step()
                optimizer_uy.step()
                optimizer_uz.step()
                optimizer_p.step()
            elif arch_type == '02':
                optimizer_u.step()
                optimizer_p.step()
            elif arch_type in ['03','04']:
                optimizer_all.step()
            # 02: pi-functions
            for opt in optimizer_pi_lst:
                opt.step()
            # 03: parameters
            if inverse_problem:
                if (epoch+1)%inverse_iter_rate == 0 and (epoch+1)>inverse_iter_t0:
                    # optimizing resistances
                    for opt in optimizer_Rd_lst:
                        opt.step()
                    for opt in optimizer_Rp_lst:
                        opt.step()
                    # Capacitance
                    if not couplingRdC:
                        for opt in optimizer_C_lst:
                            opt.step()
                    else:
                        for bid in estim_bnds:
                            C_i[bid] = Capacitance(RC_value/Rd_i[bid]()).to(device) 


            ######################
            ## ending minibatch ##
            ######################

        if HP['learning_rates']['state']['scheduler']:
            if arch_type == '01':
                scheduler_ux.step(loss.item())
                scheduler_uy.step(loss.item())
                scheduler_uz.step(loss.item())
            elif arch_type == '02':
                scheduler_u.step(loss.item())
            elif arch_type in ['03','04']:
                scheduler_all.step(loss.item())

            for sch in scheduler_pi_lst:
                sch.step(loss.item())

        if HP['learning_rates']['pressure']['scheduler']:
            if arch_type in ['01','02']:
                scheduler_p.step(loss.item())

        if HP['learning_rates']['params']['scheduler']:
            if inverse_problem:
                if epoch%inverse_iter_rate == 0 and (epoch+1)>inverse_iter_t0:
                    for sch in scheduler_params_Rd:
                        sch.step(loss.item(), epoch=epoch)
                    for sch in scheduler_params_Rp:
                        sch.step(loss.item(), epoch=epoch)
                    for sch in scheduler_params_C:
                        sch.step(loss.item(), epoch=epoch)
    

        # saving losses
        loss_track['phys'].append(loss_phys.item())
        loss_track['data'].append(loss_data.item())
        loss_track['bc'].append(loss_bc.item())
        loss_track['wk'].append(loss_wk.item())
        if adding_pi0:
            loss_track['initial'].append(loss_initial.item())
        loss_track['gradp'].append(loss_gradp.item())
        loss_track['tot'].append(loss.item())
        
        
        if give_mean_pressure:
            loss_track['pmean'].append(loss_p.item())
        
        if epoch == 0:
            logging.info('Allocated GPU Memory on run: {} GB'.format(numpy.round(torch.cuda.memory_allocated(device)/1e+6,2)))
        

        # computing the flow        
        for l,tk in enumerate(times_up):
            if not is_Fourier_features:
                nn_in = torch.cat((x, y, z, tk*torch.ones_like(x)), 1)
            else:
                nn_in = torch.cat((x_c, x_s, y_c, y_s, z_c, z_s, tk*torch.ones_like(x_c)), 1)

            if arch_type == '01':
                ux = nn_ux(nn_in)
                uy = nn_uy(nn_in)
                uz = nn_uz(nn_in)
            elif arch_type == '02':
                u = nn_u(nn_in)
                ux = u[:,0]
                uy = u[:,1]
                uz = u[:,2]
            elif arch_type in ['03','04']:
                up = nn_all(nn_in)
                ux = up[:,0]
                uy = up[:,1]
                uz = up[:,2]
            ux = ux.view(len(ux),-1)
            uy = uy.view(len(uy),-1)
            uz = uz.view(len(uz),-1)
            
            if not divergence_free:
                ux_np = ux.cpu().data.numpy()
                uy_np = uy.cpu().data.numpy()
                uz_np = uz.cpu().data.numpy()
                del ux, uy, uz

                utot_vec = ComputeVelocityVector(ux_np, uy_np, uz_np, u_length, Factors)
                del ux_np, uy_np, uz_np

            else:
                #ux = Fz_y - Fy_z
                #uy = Fx_z - Fz_x
                #uz = Fy_x - Fx_y
                Fz_y = torch.autograd.grad(uz, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
                Fy_z = torch.autograd.grad(uy, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
                ux_np = Fz_y - Fy_z
                ux_np = ux_np.cpu().data.numpy()
                del Fz_y, Fy_z
                Fx_z = torch.autograd.grad(ux, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
                Fz_x = torch.autograd.grad(uz, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
                uy_np = Fx_z - Fz_x
                uy_np = uy_np.cpu().data.numpy()
                del Fx_z, Fz_x
                Fy_x = torch.autograd.grad(uy, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
                Fx_y = torch.autograd.grad(ux, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
                uz_np = Fy_x - Fx_y
                uz_np = uz_np.cpu().data.numpy()
                del Fy_x, Fx_y

                utot_vec = ComputeVelocityVector(ux_np, uy_np, uz_np, u_length, Factors)
                del ux_np, uy_np, uz_np

            # removing solution vectors
            u_fem.vector()[:] = utot_vec
            # computing the flow with FEniCS
            for k in windkessel_bnd:
                Q_fenics[k][l] = dolfin.assemble(dolfin.dot(u_fem, n)*ds(k))
        
        for k in windkessel_bnd:
            Q_interpolator[k] = interp1d(times_up, Q_fenics[k], kind='cubic', fill_value='extrapolate')

        # saving pi functions
        for l,k in enumerate(windkessel_bnd):
            pi_fun = []
            tup = torch.Tensor(1).to(device)
            for t in times_up:                
                tup[0] = t
                pi = nns_pi[l](tup)
                pi_fun.append(pi.item())
            
            pi_track[k][epoch+1] = pi_fun


        # printing values
        if inverse_problem:
            eps_params = ComputeParamsError(Rd_i, Rd_ref, C_i, C_ref, Rp_i, Rp_ref)
            loss_track['eps_params'].append(eps_params)
            logging.info('Train Epoch: {} Total Loss: {:.10f}  \u03B5_params: {}'.format(epoch + 1, numpy.round(loss.item(),5), 
                          numpy.round(eps_params,3) ))
            PrintEstimation(inverse_problem, Rd_i, Rd_ref, C_i, C_ref, Rp_i, Rp_ref, param_track, logscale = param_factorization)
        else:
            logging.info('Train Epoch: {} Total Loss: {:.10f}'.format(epoch + 1, numpy.round(loss.item(),5) ))
        

        logging.info('loss_phys: {}'.format(loss_phys.item()))
        logging.info('loss_bc: {}'.format(loss_bc.item()))
        logging.info('loss_data: {}'.format(loss_data.item()))
        logging.info('loss_wk: {}'.format(loss_wk.item()))
        if give_mean_pressure:
            logging.info('loss_pmean: {}'.format(loss_p.item()))
        if adding_pi0:
            logging.info('loss_initial: {}'.format(loss_initial.item()))
        logging.info('loss_gradp: {}'.format(loss_gradp.item()))
        if inverse_problem:
            if range_distance_reg:
                if loss_params:
                    logging.info('loss_params: {}'.format(loss_params.item()))
                else:
                    logging.info('loss_params: not yet')
        if adding_pit_reg:
            if loss_pit_reg!=0:
                logging.info('loss_pit_reg: {}'.format(loss_pit_reg.item()))
            else:
                logging.info('loss_pit_reg: not yet')

        logging.info('---------------------')
        
        if epoch%250 == 0:
            SaveModel()

    # print elapsed time
    toc = time.time()
    elapseTime = toc - tic
    logging.info("elapse time in hrs : {}".format(numpy.round(elapseTime/60/60,2)))
    # saving the model
    SaveModel()
    logging.info('model data saved in {}'.format(options['io']['write_path'] + 'results'))

    logging.info('saving grad info')
    

    if is_tracking_gradients:
        with open(options['io']['write_path'] + 'grads_info.pickle', 'wb') as handle:
            pickle.dump(tracking_gradients_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # saving the losses and parameters
    numpy.savez_compressed(options['io']['write_path'] + 'loss.npz', loss_track)
    if inverse_problem:
        numpy.savez_compressed(options['io']['write_path'] + 'estimation.npz', param_track)
    

    with open(options['io']['write_path'] + 'pi_functions.pickle', 'wb') as handle:
        pickle.dump(pi_track, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if is_lambda_annealing:
        with open(options['io']['write_path'] + 'lambda_annealing.pickle', 'wb') as handle:
            pickle.dump(lambda_track, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(options['io']['write_path'] + 'annealing_info.pickle', 'wb') as handle:
            pickle.dump(annealing_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if HP['learning_rates']['params']['scheduler']:
        if inverse_problem:
            if len(scheduler_params_Rd) > 0:
                scheduler_params_Rd[0].save_info(options['io']['write_path'])
            else:
                scheduler_params_Rp[0].save_info(options['io']['write_path'])
        else:
            if arch_type == '01':
                scheduler_ux.save_info(options['io']['write_path'])
            elif arch_type == '02':
                scheduler_u.save_info(options['io']['write_path'])
            elif arch_type == '03':
                scheduler_all.save_info(options['io']['write_path'])


    # saving the solution at time-upsampled
    ux_dict = {}
    uy_dict = {}
    uz_dict = {}
    p_dict = {}
    OUTPUT = {}
    OUTPUT['t'] = times_up
    logging.info('saving upsampled output...')
    logging.info('using time meas for the moment!')
    for l,tk in enumerate(times_meas):
        if not is_Fourier_features:
            tup = tk[0]*torch.ones_like(x)
            nn_in = torch.cat((x.requires_grad_(), y.requires_grad_(), z.requires_grad_(), tup.requires_grad_()), 1)
        else:
            tup = tk[0]*torch.ones_like(x_c)
            nn_in = torch.cat((x_c.requires_grad_(), x_s.requires_grad_(), y_c.requires_grad_(), y_s.requires_grad_(), 
                     z_c.requires_grad_(), z_s.requires_grad_(), tup.requires_grad_()), 1)

        if arch_type == '01':
            ux = nn_ux(nn_in)
            uy = nn_uy(nn_in)
            uz = nn_uz(nn_in)
            p = nn_p(nn_in)
        elif arch_type == '02':
            u = nn_u(nn_in)
            ux = u[:,0]
            uy = u[:,1]
            uz = u[:,2]
            p = nn_p(nn_in)
        elif arch_type in ['03','04']:
            up = nn_all(nn_in)
            ux = up[:,0]
            uy = up[:,1]
            uz = up[:,2]
            p = up[:,3]
        
        p = p.view(len(p),-1)
        ux = ux.view(len(ux),-1)
        uy = uy.view(len(uy),-1)
        uz = uz.view(len(uz),-1)

        if not divergence_free:
            ux_f = ux
            uy_f = uy
            uz_f = uz

        else:
            Fz_y = torch.autograd.grad(uz, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
            Fy_z = torch.autograd.grad(uy, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
            ux_f = Fz_y - Fy_z
            del Fz_y, Fy_z
            Fx_z = torch.autograd.grad(ux, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
            Fz_x = torch.autograd.grad(uz, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
            uy_f = Fx_z - Fz_x
            del Fx_z, Fz_x
            Fy_x = torch.autograd.grad(uy, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
            Fx_y = torch.autograd.grad(ux, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
            uz_f = Fy_x - Fx_y
            del Fy_x, Fx_y


        #physical units
        ux_f = ux_f*Factors['U']
        uy_f = uy_f*Factors['U']
        uz_f = uz_f*Factors['U']
        p = p*(Factors['rho']*Factors['U']**2)
           
        ux_dict[l] = ux_f.cpu().data.numpy() # converting to cpu
        uy_dict[l] = uy_f.cpu().data.numpy() # converting to cpu
        uz_dict[l] = uz_f.cpu().data.numpy() # converting to cpu
        p_dict[l] = p.cpu().data.numpy() # converting to cpu
        

    OUTPUT['ux'] = ux_dict
    OUTPUT['uy'] = uy_dict
    OUTPUT['uz'] = uz_dict
    OUTPUT['p'] = p_dict
    
    
    numpy.savez_compressed(options['io']['write_path'] + 'output.npz', t = OUTPUT['t'], 
                 ux = OUTPUT['ux'], uy = OUTPUT['uy'], uz = OUTPUT['uz'], p = OUTPUT['p'])


    logging.info('upsampled solution saved!')


def ROUTINE(options):

    logging.info(r"""

        ____   _             Oo
    ____)   \  )\        o Ooo
  __)      _/    \         oO
 _)      _/     _/''--.__ o
/___    \____.-'   {>(9 /
 )___       ;-...;__;-''
   )___    /)   /|))
     )_   / \/\/ )//
       )_/

    """)


    logging.info('Initializing PINNs Windkessel Algorithm with Random Timing')
    logging.info('savepath: {}'.format(options['io']['write_path']))
    DATA, CoPoints, Factors = DataReader(options)
    HyperParameters_dict = SetHyperparameters(options)
    FENICS_lst = LoadMesh(options)

    PINNs(HyperParameters_dict, DATA, Factors, CoPoints, options, fenics = FENICS_lst)


    

if __name__ == '__main__':


    if len(sys.argv) > 1:
        if os.path.exists(sys.argv[1]):
            inputfile = sys.argv[1]
        else:
            raise Exception('Command line arg given but input file does not exist:'
                            ' {}'.format(sys.argv[1]))
    else:
        raise Exception('An input file is required as argument!')
        # Reading inputfile
    with open(inputfile, 'r+') as f:
        options = yaml.load(f, Loader=yaml.Loader)


    # creating write directory
    path = Path(options['io']['write_path'])
    path.mkdir(parents=True, exist_ok=True)

    # creating write directory
    path_model = Path(options['io']['write_path'] + 'model/')
    path_model.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(format='%(levelname)s:%(message)s', filename= options['io']['write_path'] + '/run.log', level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    save_inputfile(options, inputfile)
    ROUTINE(options)
    
