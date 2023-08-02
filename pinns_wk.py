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
import torch
torch.cuda.empty_cache()
import pickle
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial import ConvexHull
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
    logging.info('Copied input file to {}'.format(path))

def area_from_3_points(x, y, z):
    return (numpy.sum(numpy.cross(x-y, x-z), axis=-1)**2)**(0.5)/2

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

def LoadMesh(options, Factors):

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

    # normalizing by L scale in both modes
    #mesh.coordinates()[:] = mesh.coordinates()/Factors['L']

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
    
    HP_dict['activation_function'] = options['PINNs']['activation_function']
    HP_dict['batchsize'] = options['PINNs']['batchsize']
    HP_dict['learning_rates'] = options['PINNs']['learning_rates']
    HP_dict['inverse_iter_rate'] = options['windkessel']['inverse_problem']['iter_rate']
    
    HP_dict['epochs'] = options['PINNs']['epochs']
    HP_dict['neurons_per_layer'] = options['PINNs']['neurons_per_layer'] # neurons per layer
    
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

    ux_current = out_ux[:,0]*Factors['U']
    uy_current = out_uy[:,0]*Factors['U']
    uz_current = out_uz[:,0]*Factors['U']
    utot_vec = numpy.zeros(u_length)
    utot_vec[0::3] = ux_current
    utot_vec[1::3] = uy_current
    utot_vec[2::3] = uz_current

    return utot_vec

def ComputePressureVector(out_p, Factors):

    out_p = out_p.cpu().data.numpy() # converting to cpu
    p_current = out_p[:,0]*Factors['rho']*Factors['U']**2

    return p_current

def ComputeSolutionError(uvec, uvec_ref, pvec, pvec_ref):
    
    epsilon = 0
    # velocity relative l2 error
    epsilon += numpy.sqrt(numpy.sum((uvec-uvec_ref)**2)/len(uvec))/numpy.sqrt(numpy.sum(uvec_ref**2)/len(uvec))
    # pressure relative l2 error
    epsilon += numpy.sqrt(numpy.sum((pvec-pvec_ref)**2)/len(pvec))/numpy.sqrt(numpy.sum(pvec_ref**2)/len(pvec))

    return epsilon

def ComputeParamsError(params, params_ref, pressure_ratio=1):
    
    ratio = 0
    for k in params.keys():
        param_k = params[k]().item()/pressure_ratio
        param_k_ref = params_ref[k]
        ratio += (1/len(params_ref))*(param_k-param_k_ref)**2/param_k_ref**2
        
    return numpy.sqrt(ratio)

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

def ComputeFlows(coordinates, velocities, normal, on_numpy = False):


    # unpacking the coordinates
    x = coordinates[0]
    y = coordinates[1]
    z = coordinates[2]
    # unpacking the velocities
    ux = velocities[0]
    uy = velocities[1]
    uz = velocities[2]

    if not on_numpy:
        x = x.cpu().data.numpy()
        y = y.cpu().data.numpy()
        z = z.cpu().data.numpy()
        x = numpy.array(x[:,0])
        y = numpy.array(y[:,0])
        z = numpy.array(z[:,0])
        ux = ux.cpu().data.numpy()
        uy = uy.cpu().data.numpy()
        uz = uz.cpu().data.numpy()
        ux = numpy.array(ux[:,0])
        uy = numpy.array(uy[:,0])
        uz = numpy.array(uz[:,0])


    points = numpy.array(list(zip(x,y,z)))
    hull = ConvexHull(points)
    
    Q = 0
    un = ux*normal[0] + uy*normal[1] + uz*normal[2]

    for vertices in hull.simplices:
        mean_value = (un[vertices[0]] + un[vertices[1]] + un[vertices[2]]) / 3
        area = area_from_3_points(points[vertices[0]], points[vertices[1]], points[vertices[2]])
        Q += mean_value*area

    return Q

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

def PINNs(HP, DATA, Factors, CoPoints, options, fenics):
    ''' Physics Inform Neuronal Network

    Args:
        HP (dict):      Hiperparameters of the NN
        DATA (dict):    Data dictionary
        CoPoints (dict):    Collocation points
        u (FEniCS func):    optional for visualization
    '''


    arch_type = options['PINNs']['architecture']
    if arch_type == '01':
        logging.info('Using Architecture 01: NN(ux) + NN(uy) + NN(uz) + NN(p)')
    elif arch_type == '02':
        logging.info('Using Architecture 02: NN(ux, uy, uz) + NN(p)')
    elif arch_type == '03':
        logging.info('Using Architecture 03: NN(ux, uy, uz, p)')
    else:
        raise Exception('architecture not recognized!')



    # if tracking gradients
    is_tracking_gradients = False
    tracking_gradients_dict = {}
    if len(options['Tracking_gradients']['terms']) > 0:
        is_tracking_gradients = True
        for trm in options['Tracking_gradients']['terms']:
            tracking_gradients_dict[trm] = {}
            for hl in range(HP['hidden_layers']+1):
                tracking_gradients_dict[trm]['layer{}'.format(hl)] = {'weight': [], 'bias': []}


    # if lambda annealing
    is_lambda_annealing = options['PINNs']['lambdas']['annealing']['apply']
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
    windkessel_track = {}
    Q_fenics = {}
    P_fenics = {}

    for bid in windkessel_bnd:
        Q_fenics[bid] = 0
        P_fenics[bid] = 0
        windkessel_track[bid] = {
            'Q_fenics': [],
            'P_fenics': [],            
            } 

    if pretrain:
        logging.info('training from model in : {}'.format(options['PINNs']['pretrain']['model']))
    
    loss_track = {
        'tot': [],
        'phys': [],
        'bc': [],
        'data': [],
        'wk': [],
        'pmean': [],
        'gradp': [],
        'eps_sol':[],
        'eps_params': [],
    }

    param_factorization = False
    inverse_problem = options['windkessel']['inverse_problem']['apply']
    estim_bnds = []
    if inverse_problem:
        estim_bnds = options['windkessel']['inverse_problem']['bnds']
        inverse_iter_rate = HP['inverse_iter_rate']
        inverse_iter_t0 = options['windkessel']['inverse_problem']['iter_t0']
        param_factorization = options['windkessel']['inverse_problem']['factorization']
        range_distance_reg = False
        if 'range_distance_reg' in options['windkessel']['inverse_problem']:
            range_distance_reg = options['windkessel']['inverse_problem']['range_distance_reg']
    
    param_track = {}
    for k in estim_bnds:
        param_track[k] = []

    #normals_wk = ComputeNormals(DATA, windkessel_bnd)

    if 'mode' in Factors.keys():
        NormalizationMode = Factors['mode']
    else:
        NormalizationMode = 'reynolds'
        Factors['U'] = Factors['uscale']
        Factors['Rfac'] = Factors['rho']*Factors['U']/Factors['L']**2
        

    if NormalizationMode == 'elliptic':
        logging.info('Elliptic normalization recognized!')
        # factor tuples for normalization
        xfac = Factors['xfac']
        yfac = Factors['yfac']
        zfac = Factors['zfac']
        uxfac = Factors['uxfac']
        uyfac = Factors['uyfac']
        uzfac = Factors['uzfac']
        pfac = Factors['pfac']
    else:
        logging.info('Using {} normalization in data'.format(NormalizationMode))
    
    if NormalizationMode not in ['reynolds','elliptic']:
        raise Exception('Normalization {} mode not recognized!'.format(NormalizationMode))


    seed = None
    if 'seed' in options['PINNs']:
        seed = options['PINNs']['seed']
        logging.info('Randomness seed set in: {}'.format(seed))
        torch.manual_seed(seed)
        numpy.random.seed(seed)


    divergence_free = False
    if 'divergence-free' in options['PINNs']:
        divergence_free = options['PINNs']['divergence-free']
        if divergence_free:
            logging.info('Using Divergence Free Formulation: NN(\u03A6, p)')


    if inverse_problem:
        if range_distance_reg:
            logging.info('Adding Parameter Distance Regularizator: ||\u03B8 - \u03B8_range ||\u00B2')

    normalize_pressure = False
    if 'normalize_pressure' in options['windkessel']:
        normalize_pressure = options['windkessel']['normalize_pressure']
        if normalize_pressure:
            logging.info('Normalizing pressure at P mean term')

    save_xdmf = False
    if 'XDMF' in options:
        if options['XDMF']['save']:
            save_xdmf = True
            logging.info('Saving XDMF files for velocity and pressure')
            xdmf_u = dolfin.XDMFFile(options['io']['write_path'] + 'u_epoch.xdmf')
            xdmf_p = dolfin.XDMFFile(options['io']['write_path'] + 'p_epoch.xdmf')

    u_fem = fenics[0]
    u_length = len(u_fem.vector().get_local())
    p_fem = fenics[1]
    u_fem.rename('u','velocity')
    p_fem.rename('p','pressure')
    # computing normal
    n = dolfin.FacetNormal(fenics[2])
    ds = dolfin.Measure('ds', domain=fenics[2], subdomain_data=fenics[3])
    ones = fenics[4]
    
    
    def InitTensors(device):    
        # collocation points
        x = torch.Tensor(CoPoints['x']).to(device)
        y = torch.Tensor(CoPoints['y']).to(device)
        z = torch.Tensor(CoPoints['z']).to(device)
        # boundary points
        xb = torch.Tensor(DATA['xb']).to(device)
        yb = torch.Tensor(DATA['yb']).to(device)
        zb = torch.Tensor(DATA['zb']).to(device)
    
        # mean pressure value
        pwk_mean = torch.Tensor(DATA['pwk_mean']).to(device)
        pwk_mean = torch.mean(pwk_mean)
        # windkessel points
        xwk = {}
        ywk = {}
        zwk = {}

        for k in windkessel_bnd:
            xwk[k] = torch.Tensor(DATA['xwk'][k]).to(device)
            ywk[k] = torch.Tensor(DATA['ywk'][k]).to(device)
            zwk[k] = torch.Tensor(DATA['zwk'][k]).to(device)
    
        # data points
        xd = {}
        yd = {}
        zd = {}
        uxd = {}
        uyd = {}
        uzd = {}
        for k in DATA['x'].keys():
            xd[k] = torch.Tensor(DATA['x'][k]).to(device)
            yd[k] = torch.Tensor(DATA['y'][k]).to(device)
            zd[k] = torch.Tensor(DATA['z'][k]).to(device)
            uxd[k] = torch.Tensor(DATA['ux'][k]).to(device)
            uyd[k] = torch.Tensor(DATA['uy'][k]).to(device)
            uzd[k] = torch.Tensor(DATA['uz'][k]).to(device)

        for key in xwk.keys():
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


        # changing to float... cuda slower in double?. taken from arzani
        changing_to_float = True
        if changing_to_float:
            x = x.type(torch.cuda.FloatTensor)
            y = y.type(torch.cuda.FloatTensor)
            z = z.type(torch.cuda.FloatTensor)
            xb = xb.type(torch.cuda.FloatTensor)
            yb = yb.type(torch.cuda.FloatTensor)
            zb = zb.type(torch.cuda.FloatTensor)
            pwk_mean = pwk_mean.type(torch.cuda.FloatTensor)
            
            for k in windkessel_bnd:
                xwk[k] = xwk[k].type(torch.cuda.FloatTensor)
                ywk[k] = ywk[k].type(torch.cuda.FloatTensor)
                zwk[k] = zwk[k].type(torch.cuda.FloatTensor)
            for k in DATA['x'].keys():
                xd[k] = xd[k].type(torch.cuda.FloatTensor)
                yd[k] = yd[k].type(torch.cuda.FloatTensor)
                zd[k] = zd[k].type(torch.cuda.FloatTensor)
                uxd[k] = uxd[k].type(torch.cuda.FloatTensor)
                uyd[k] = uyd[k].type(torch.cuda.FloatTensor)
                uzd[k] = uzd[k].type(torch.cuda.FloatTensor)


        return x,y,z,xb,yb,zb,xwk,ywk,zwk,xd,yd,zd,uxd,uyd,uzd,pwk_mean

    device = HP['device']
    x,y,z,xb,yb,zb,xwk,ywk,zwk,xd,yd,zd,uxd,uyd,uzd,pwk_mean = InitTensors(device)
    # dataset
    dataset = TensorDataset(x,y,z)
    dataloader = DataLoader(dataset, batch_size=HP['batchsize'], shuffle=True, num_workers = 0, drop_last = True )

    h_n = HP['neurons_per_layer']
    n_layers = HP['hidden_layers']
    input_n = 3
    act_function = HP['activation_function']

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
        
        def forward(self, x):
            output = self.main(x)
            return output


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


    wk_r = {}
    wk_r_ref = {}
    r_fact = Factors['Rfac']
    input_path = options['data']['input_path'] + 'input.yaml'
    Rtot_range = options['windkessel']['inverse_problem']['Rtot_range']

    with open(input_path , 'r+') as f:
        sim_options = yaml.load(f, Loader=yaml.Loader)
    for bid in options['windkessel']['boundaries']:
        if inverse_problem and bid in estim_bnds:
            wk_r[bid] = Resistance(Rtot_range).to(device)
        for bc in sim_options['boundary_conditions']:
            if bid == bc['id']:
                wk_r_ref[bid] = (bc['parameters']['R_p'] + bc['parameters']['R_d'])/r_fact
        
    if pretrain:
        logging.info('reading previous model...')
        if arch_type == '01':
            nn_ux.load_state_dict(torch.load(options['PINNs']['pretrain']['model'] + 'sten_data_ux.pt'))
            nn_uy.load_state_dict(torch.load(options['PINNs']['pretrain']['model'] + 'sten_data_uy.pt'))
            nn_uz.load_state_dict(torch.load(options['PINNs']['pretrain']['model'] + 'sten_data_uz.pt')) 
            nn_p.load_state_dict(torch.load(options['PINNs']['pretrain']['model'] + 'sten_data_p.pt'))
        elif arch_type == '02':
            nn_u.load_state_dict(torch.load(options['PINNs']['pretrain']['model'] + 'sten_data_u.pt'))
            nn_p.load_state_dict(torch.load(options['PINNs']['pretrain']['model'] + 'sten_data_p.pt'))
        elif arch_type == '03':
            nn_all.load_state_dict(torch.load(options['PINNs']['pretrain']['model'] + 'sten_data_all.pt'))
        
        if inverse_problem:
            logging.info('(PRETRAIN) reading last parameter estimation...')
            params_path = options['PINNs']['pretrain']['model'] + 'estimation.npz'
            dparams = numpy.load(params_path, allow_pickle=True)
            params = dparams['arr_0'].item()
            for k in wk_r.keys():
                wk_r[k] = Resistance(params[k][-1]/r_fact).to(device)
            
            del params, dparams
                
    logging.info('true values for total resistances')
    if inverse_problem:
        for elem in wk_r_ref.items():
            if elem[0] in wk_r.keys():
                rinit = wk_r[elem[0]].R.item()
                logging.info('boundary: {} \t Ri_tot: {} \t R_tot: {}'.format(elem[0],round(rinit,2), round(elem[1],2)))
            else:
                logging.info('boundary: {} \t Ri_tot: --- \t R_tot: {}'.format(elem[0], round(elem[1],2)))
    else:
        for elem in wk_r_ref.items():
            logging.info('boundary: {} \t R_tot: {}'.format(elem[0],round(elem[1],2)))


    if inverse_problem:
        if param_factorization:
            logging.info('Applying Parameter Factorization: 2^\u03B8')
            initial_factors = {}
            for k in wk_r.keys():
                initial_factors[k] = wk_r[k].R.item()
                wk_r[k] = Resistance(1.0).to(device)

        if range_distance_reg:
            Rtot_range_min = {}
            Rtot_range_max = {}
            if not param_factorization: 
                for k in wk_r.keys():
                    Rtot_range_min[k] = Resistance(float(Rtot_range[0])).to(device)
                    Rtot_range_max[k] = Resistance(float(Rtot_range[1])).to(device)
            else:
                for k in wk_r.keys():
                    theta_min = numpy.log2(Rtot_range[0]/initial_factors[k])
                    theta_max = numpy.log2(Rtot_range[1]/initial_factors[k])
                    
                    Rtot_range_min[k] = Resistance(float(theta_min)).to(device)
                    Rtot_range_max[k] = Resistance(float(theta_max)).to(device)

    if not pretrain:
        # initializing weights
        def init_normal(m):
            if type(m) == nn.Linear:
                if act_function == 'relu':
                    nn.init.kaiming_normal_(m.weight) # He (relu)
                else:
                    nn.init.xavier_normal_(m.weight) # Xavier
                
        if arch_type == '01':
            nn_ux.apply(init_normal)
            nn_uy.apply(init_normal)
            nn_uz.apply(init_normal)
            nn_p.apply(init_normal)
        elif arch_type == '02':
            nn_u.apply(init_normal)
            nn_p.apply(init_normal)
        elif arch_type == '03':
            nn_all.apply(init_normal)


    # Optimizer: Adam
    if arch_type == '01':
        optimizer_ux = torch.optim.Adam(nn_ux.parameters(), lr=HP['learning_rates']['state']['l'], betas = (0.9,0.99),eps = 10**-15)
        optimizer_uy = torch.optim.Adam(nn_uy.parameters(), lr=HP['learning_rates']['state']['l'], betas = (0.9,0.99),eps = 10**-15)
        optimizer_uz = torch.optim.Adam(nn_uz.parameters(), lr=HP['learning_rates']['state']['l'], betas = (0.9,0.99),eps = 10**-15)    
        optimizer_p = torch.optim.Adam(nn_p.parameters(), lr=HP['learning_rates']['state']['l'], betas = (0.9,0.99),eps = 10**-15)
    elif arch_type == '02':
        optimizer_u = torch.optim.Adam(nn_u.parameters(), lr=HP['learning_rates']['state']['l'], betas = (0.9,0.99),eps = 10**-15)
        optimizer_p = torch.optim.Adam(nn_p.parameters(), lr=HP['learning_rates']['state']['l'], betas = (0.9,0.99),eps = 10**-15)
    elif arch_type == '03':
        optimizer_all = torch.optim.Adam(nn_all.parameters(), lr=HP['learning_rates']['state']['l'], betas = (0.9,0.99),eps = 10**-15)

    optimizer_r_lst = []
    if inverse_problem:
        for key in wk_r.keys():
            optimizer_r_lst.append(torch.optim.Adam([{'params':[wk_r[key].R], 'lr':HP['learning_rates']['params']['l']}], betas = (0.9,0.99),eps = 10**-15))

    def Loss_Phys(x, y, z, normmode):

        x.requires_grad = True
        y.requires_grad = True
        z.requires_grad = True

        nn_in = torch.cat((x,y,z),1)
        
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
        elif arch_type == '03':
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

        if normmode == 'elliptic':
            loss_1 = (uxfac[0]/xfac[0])*(uxfac[0]*ux + uxfac[1])*ux_x + \
                     (uxfac[0]/yfac[0])*(uyfac[0]*uy + uyfac[1])*ux_y + \
                     (uxfac[0]/zfac[0])*(uzfac[0]*uz + uzfac[1])*ux_z - \
                     (1/Re)*(uxfac[0]/xfac[0]**2)*ux_xx - \
                     (1/Re)*(uxfac[0]/yfac[0]**2)*ux_yy - \
                     (1/Re)*(uxfac[0]/zfac[0]**2)*ux_zz + \
                     (pfac[0]/xfac[0])*p_x

            loss_2 = (uyfac[0]/xfac[0])*(uxfac[0]*ux + uxfac[1])*uy_x + \
                     (uyfac[0]/yfac[0])*(uyfac[0]*uy + uyfac[1])*uy_y + \
                     (uyfac[0]/zfac[0])*(uzfac[0]*uz + uzfac[1])*uy_z - \
                     (1/Re)*(uyfac[0]/xfac[0]**2)*uy_xx - \
                     (1/Re)*(uyfac[0]/yfac[0]**2)*uy_yy - \
                     (1/Re)*(uyfac[0]/zfac[0]**2)*uy_zz + \
                     (pfac[0]/yfac[0])*p_y

            loss_3 = (uzfac[0]/xfac[0])*(uxfac[0]*ux + uxfac[1])*uz_x + \
                     (uzfac[0]/yfac[0])*(uyfac[0]*uy + uyfac[1])*uz_y + \
                     (uzfac[0]/zfac[0])*(uzfac[0]*uz + uzfac[1])*uz_z - \
                     (1/Re)*(uzfac[0]/xfac[0]**2)*uz_xx - \
                     (1/Re)*(uzfac[0]/yfac[0]**2)*uz_yy - \
                     (1/Re)*(uzfac[0]/zfac[0]**2)*uz_zz + \
                     (pfac[0]/zfac[0])*p_z

            loss_4 = (uxfac[0]/xfac[0])*ux_x  + (uyfac[0]/yfac[0])*uy_y + (uzfac[0]/zfac[0])*uz_z  # continuity
        
        else:
            loss_1 = ux*ux_x + uy*ux_y + uz*ux_z - (1/Re)*( ux_xx + ux_yy + ux_zz) + p_x
            loss_2 = ux*uy_x + uy*uy_y + uz*uy_z - (1/Re)*( uy_xx + uy_yy + uy_zz) + p_y
            loss_3 = ux*uz_x + uy*uz_y + uz*uz_z - (1/Re)*( uz_xx + uz_yy + uz_zz) + p_z
            loss_4 = ux_x  + uy_y + uz_z  # continuity

        # MSE loss
        loss_f = nn.MSELoss()


        loss = loss_f(loss_1,torch.zeros_like(loss_1)) \
                + loss_f(loss_2,torch.zeros_like(loss_2)) \
                + loss_f(loss_3,torch.zeros_like(loss_3)) \
                + loss_f(loss_4,torch.zeros_like(loss_4))

        return loss

    def Loss_Phys_Div(x, y, z, norm_mode):

        
        nn_in = torch.cat((x,y,z),1)
        
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
        Fx_z = torch.autograd.grad(Fx, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fx_zz = torch.autograd.grad(Fx_z, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fx_zzz = torch.autograd.grad(Fx_zz, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fx_zy = torch.autograd.grad(Fx_z, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fx_zyy = torch.autograd.grad(Fx_zy, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fx_zx = torch.autograd.grad(Fx_z, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        Fx_zxx = torch.autograd.grad(Fx_zx, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        # y
        Fy_x = torch.autograd.grad(Fy, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        Fy_xy = torch.autograd.grad(Fy_x, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fy_xyy = torch.autograd.grad(Fy_xy, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fy_xz = torch.autograd.grad(Fy_x, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fy_xzz = torch.autograd.grad(Fy_xz, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fy_xx = torch.autograd.grad(Fy_x, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        Fy_xxx = torch.autograd.grad(Fy_xx, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        Fy_z = torch.autograd.grad(Fy, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fy_zz = torch.autograd.grad(Fy_z, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fy_zzz = torch.autograd.grad(Fy_zz, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fy_zy = torch.autograd.grad(Fy_z, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fy_zyy = torch.autograd.grad(Fy_zy, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fy_zx = torch.autograd.grad(Fy_z, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        Fy_zxx = torch.autograd.grad(Fy_zx, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        # z
        Fz_x = torch.autograd.grad(Fz, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        Fz_xz = torch.autograd.grad(Fz_x, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fz_xzz = torch.autograd.grad(Fz_xz, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fz_xy = torch.autograd.grad(Fz_x, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fz_xyy = torch.autograd.grad(Fz_xy, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fz_xx = torch.autograd.grad(Fz_x, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        Fz_xxx = torch.autograd.grad(Fz_xx, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        Fz_y = torch.autograd.grad(Fz, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fz_yz = torch.autograd.grad(Fz_y, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fz_yzz = torch.autograd.grad(Fz_yz, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fz_yy = torch.autograd.grad(Fz_y, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fz_yyy = torch.autograd.grad(Fz_yy, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fz_yx = torch.autograd.grad(Fz_y, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        Fz_yxx = torch.autograd.grad(Fz_yx, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        # grad p
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
        p_z = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(z),create_graph = True,only_inputs=True)[0]


        Re = Factors['Re']

        if norm_mode == 'reynolds':
            loss_1 = (Fz_y - Fy_z)*(Fz_yx - Fy_zx) + \
                     (Fx_z - Fz_x)*(Fz_yy - Fy_zy) + \
                     (Fy_x - Fx_y)*(Fz_yz - Fy_zz) - \
                     (1/Re)*(Fz_yxx - Fy_zxx) - \
                     (1/Re)*(Fz_yyy - Fy_zyy) - \
                     (1/Re)*(Fz_yzz - Fy_zzz) + \
                     p_x

            loss_2 = (Fz_y - Fy_z)*(Fx_zx - Fz_xx) + \
                     (Fx_z - Fz_x)*(Fx_zy - Fz_xy) + \
                     (Fy_x - Fx_y)*(Fx_zz - Fz_xz) - \
                     (1/Re)*(Fx_zxx - Fz_xxx) - \
                     (1/Re)*(Fx_zyy - Fz_xyy) - \
                     (1/Re)*(Fx_zzz - Fz_xzz) + \
                     p_y

            loss_3 = (Fz_y - Fy_z)*(Fy_xx - Fx_yx) + \
                     (Fx_z - Fz_x)*(Fy_xy - Fx_yy) + \
                     (Fy_x - Fx_y)*(Fy_xz - Fx_yz) - \
                     (1/Re)*(Fy_xxx - Fx_yxx) - \
                     (1/Re)*(Fy_xyy - Fx_yyy) - \
                     (1/Re)*(Fy_xzz - Fx_yzz) + \
                     p_z

        elif norm_mode == 'elliptic':
            Ax = uxfac[0]/xfac[0]
            Ay = uyfac[0]/yfac[0]
            Az = uzfac[0]/zfac[0]
            
            loss_1 = (uxfac[0]/xfac[0])*(uxfac[0]*Ax*(Fz_y - Fy_z) + uxfac[1])*Ax*(Fz_yx - Fy_zx) + \
                     (uxfac[0]/yfac[0])*(uyfac[0]*Ay*(Fx_z - Fz_x) + uyfac[1])*Ax*(Fz_yy - Fy_zy) + \
                     (uxfac[0]/zfac[0])*(uzfac[0]*Az*(Fy_x - Fx_y) + uzfac[1])*Ax*(Fz_yz - Fy_zz) - \
                     (1/Re)*(uxfac[0]/xfac[0]**2)*Ax*(Fz_yxx - Fy_zxx) - \
                     (1/Re)*(uxfac[0]/yfac[0]**2)*Ax*(Fz_yyy - Fy_zyy) - \
                     (1/Re)*(uxfac[0]/zfac[0]**2)*Ax*(Fz_yzz - Fy_zzz) + \
                     (pfac[0]/xfac[0])*p_x

            loss_2 = (uyfac[0]/xfac[0])*(uxfac[0]*Ax*(Fz_y - Fy_z) + uxfac[1])*Ay*(Fx_zx - Fz_xx) + \
                     (uyfac[0]/yfac[0])*(uyfac[0]*Ay*(Fx_z - Fz_x) + uyfac[1])*Ay*(Fx_zy - Fz_xy) + \
                     (uyfac[0]/zfac[0])*(uzfac[0]*Az*(Fy_x - Fx_y) + uzfac[1])*Ay*(Fx_zz - Fz_xz) - \
                     (1/Re)*(uyfac[0]/xfac[0]**2)*Ay*(Fx_zxx - Fz_xxx) - \
                     (1/Re)*(uyfac[0]/yfac[0]**2)*Ay*(Fx_zyy - Fz_xyy) - \
                     (1/Re)*(uyfac[0]/zfac[0]**2)*Ay*(Fx_zzz - Fz_xzz) + \
                     (pfac[0]/yfac[0])*p_y

            loss_3 = (uzfac[0]/xfac[0])*(uxfac[0]*Ax*(Fz_y - Fy_z) + uxfac[1])*Az*(Fy_xx - Fx_yx) + \
                     (uzfac[0]/yfac[0])*(uyfac[0]*Ay*(Fx_z - Fz_x) + uyfac[1])*Az*(Fy_xy - Fx_yy) + \
                     (uzfac[0]/zfac[0])*(uzfac[0]*Az*(Fy_x - Fx_y) + uzfac[1])*Az*(Fy_xz - Fx_yz) - \
                     (1/Re)*(uzfac[0]/xfac[0]**2)*Az*(Fy_xxx - Fx_yxx) - \
                     (1/Re)*(uzfac[0]/yfac[0]**2)*Az*(Fy_xyy - Fx_yyy) - \
                     (1/Re)*(uzfac[0]/zfac[0]**2)*Az*(Fy_xzz - Fx_yzz) + \
                     (pfac[0]/zfac[0])*p_z

        # MSE loss
        loss_f = nn.MSELoss()

        loss = loss_f(loss_1,torch.zeros_like(loss_1)) \
                + loss_f(loss_2,torch.zeros_like(loss_2)) \
                + loss_f(loss_3,torch.zeros_like(loss_3))

        return loss

    def Loss_Data():

        loss_f = nn.MSELoss()
        loss = 0

        for k in xd.keys():
            nn_in = torch.cat((xd[k], yd[k], zd[k]), 1)
            if arch_type == '01':
                ux = nn_ux(nn_in)
                uy = nn_uy(nn_in)
                uz = nn_uz(nn_in)
            elif arch_type == '02':
                u = nn_u(nn_in)
                ux = u[:,0]
                uy = u[:,1]
                uz = u[:,2]
            elif arch_type == '03':
                up = nn_all(nn_in)
                ux = up[:,0]
                uy = up[:,1]
                uz = up[:,2]
                
            ux = ux.view(len(ux), -1)
            uy = uy.view(len(uy), -1)
            uz = uz.view(len(uz), -1)


            if not divergence_free:
                loss += loss_f(ux, uxd[k]) + loss_f(uy, uyd[k]) + loss_f(uz, uzd[k])
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

                loss += loss_f(ux2, uxd[k]) + loss_f(uy2, uyd[k]) + loss_f(uz2, uzd[k])



        return loss

    def Loss_BC(normmode):
        
        loss_f = nn.MSELoss()

        nn_in = torch.cat((xb, yb, zb), 1)
        if arch_type == '01':
            ux = nn_ux(nn_in)
            uy = nn_uy(nn_in)
            uz = nn_uz(nn_in)
        elif arch_type == '02':
            u = nn_u(nn_in)
            ux = u[:,0]
            uy = u[:,1]
            uz = u[:,2]
        elif arch_type == '03':
            up = nn_all(nn_in)
            ux = up[:,0]
            uy = up[:,1]
            uz = up[:,2]
            
        ux = ux.view(len(ux), -1)
        uy = uy.view(len(uy), -1)
        uz = uz.view(len(uz), -1)


        if not divergence_free:
            if normmode == 'elliptic':
                uxbc = ux*uxfac[0]+uxfac[1]
                uybc = uy*uyfac[0]+uyfac[1]
                uzbc = uz*uzfac[0]+uzfac[1]
                
                loss = loss_f(uxbc, torch.zeros_like(uxbc)) + \
                        loss_f(uybc, torch.zeros_like(uybc)) + \
                        loss_f(uzbc, torch.zeros_like(uzbc))
            else:
                loss = loss_f(ux, torch.zeros_like(ux)) + \
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
         

            if normmode == 'elliptic':
                uxbc = uxb*uxfac[0]+uxfac[1]
                uybc = uyb*uyfac[0]+uyfac[1]
                uzbc = uzb*uzfac[0]+uzfac[1]
                
                loss = loss_f(uxbc, torch.zeros_like(uxbc)) + \
                        loss_f(uybc, torch.zeros_like(uybc)) + \
                        loss_f(uzbc, torch.zeros_like(uzbc))
            else:
                loss = loss_f(uxb, torch.zeros_like(uxb)) + \
                        loss_f(uyb, torch.zeros_like(uyb)) + \
                        loss_f(uzb, torch.zeros_like(uzb))

        return loss

    def Loss_WK(normmode, wk_lst, estim_bnds):
        
        loss = 0
        loss_f = nn.MSELoss()
        loss_gradp = 0
        qfac = 1/(Factors['U']*Factors['L']**2)


        for k in wk_lst:
            nn_in = torch.cat((xwk[k], ywk[k], zwk[k]), 1)
            if arch_type in ['01','02']:
                pwk = nn_p(nn_in)
            else:
                up = nn_all(nn_in)
                pwk = up[:,3]

            pwk = pwk.view(len(pwk),-1)
            Qk = Q_fenics[k]

            if k in estim_bnds:
                if not param_factorization:
                    #if normmode == 'elliptic':
                    #    loss_k = pwk - (Qk*qfac*wk_r[k]()/pfac[0] - pfac[1]/pfac[0])*torch.ones_like(pwk)
                    #else:
                    if not normalize_pressure:
                        loss_k = pwk - Qk*qfac*wk_r[k]()*torch.ones_like(pwk)
                    else:
                        loss_k = pwk/pwk_mean - Qk*qfac*wk_r[k]()*torch.ones_like(pwk)/pwk_mean
                else:
                    #loss_k = pwk - Qk*qfac*initial_factors[k]*torch.pow(2,wk_r[k]())*torch.ones_like(pwk)
                    loss_k = pwk - Qk*qfac*initial_factors[k]*wk_r[k]()*torch.ones_like(pwk)
                    
                        

            else:
                if normmode == 'elliptic':
                    loss_k = pwk - (Qk*qfac*wk_r_ref[k]/pfac[0] - pfac[1]/pfac[0])*torch.ones_like(pwk)
                else:
                    loss_k = pwk - Qk*qfac*wk_r_ref[k]*torch.ones_like(pwk)



            loss +=  loss_f(loss_k, torch.zeros_like(loss_k))

            p_x = torch.autograd.grad(pwk, xwk[k], grad_outputs=torch.ones_like(xwk[k]), create_graph = True,only_inputs=True)[0]
            p_y = torch.autograd.grad(pwk, ywk[k], grad_outputs=torch.ones_like(ywk[k]), create_graph = True,only_inputs=True)[0]
            p_z = torch.autograd.grad(pwk, zwk[k], grad_outputs=torch.ones_like(zwk[k]), create_graph = True,only_inputs=True)[0]
            
            if normmode == 'elliptic':
                p_x = 1/xfac[0]*p_x
                p_y = 1/yfac[0]*p_y
                p_z = 1/zfac[0]*p_z

            loss_gradp += loss_f(p_x,torch.zeros_like(p_x))
            loss_gradp += loss_f(p_y,torch.zeros_like(p_y))
            loss_gradp += loss_f(p_z,torch.zeros_like(p_z))



        return loss, loss_gradp

    def Loss_PMean():

        loss_f = nn.MSELoss()
        mean_pressure = torch.zeros_like(pwk_mean)


        for k in windkessel_bnd:
            nn_in = torch.cat((xwk[k], ywk[k], zwk[k]), 1)
            
            if arch_type in ['01','02']:
                pwk = nn_p(nn_in)
            else:
                up = nn_all(nn_in)
                pwk = up[:,3]

            pwk = pwk.view(len(pwk),-1)

            mean_pressure += torch.mean(pwk)/len(windkessel_bnd)


        if not normalize_pressure:
            loss = loss_f( mean_pressure, pwk_mean)
        else:
            loss = loss_f( mean_pressure/pwk_mean , torch.ones_like(pwk_mean))

        

        return loss

    def Loss_RangeParams():
        
        loss_f = nn.MSELoss()
        aux = torch.tensor([]).to(device)

        for k in wk_r.keys():
            if wk_r[k].R < Rtot_range_min[k].R:
                aux = torch.cat((aux, Rtot_range_min[k].R - wk_r[k].R),-1)
            elif wk_r[k].R > Rtot_range_max[k].R:
                aux = torch.cat((aux, wk_r[k].R - Rtot_range_max[k].R),-1)


        if len(aux) > 0:
            return loss_f(aux,torch.zeros_like(aux))
        else:
            return False


    if inverse_problem:
        if len(options['windkessel']['inverse_problem']['bnds']) > 0:
           logging.info('estimating windkessel param. for boundaries: {}'.format(options['windkessel']['inverse_problem']['bnds']))


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

    last_batch = len(dataloader)-1

    # main loop
    tic = time.time()
    #last_batch = len(dataloader)-1

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
        logging.info('State Scheduler: lr={} \t Threshold={} \t Factor={} \t Patience={}\t GivenEpochs={}'.format(lr,sch_threshold,sch_factor,sch_patience, sch_given_epochs))
        
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
                    patience= sch_patience ,min_lr = sch_lmin, given_epochs = sch_given_epochs, threshold=sch_threshold, verbose=if_verbose)
        elif arch_type == '02':
            scheduler_u = OwnScheduler(optimizer_u, logging, mode='min', factor=sch_factor, 
                    patience= sch_patience, min_lr = sch_lmin, given_epochs = sch_given_epochs, threshold=sch_threshold, verbose=if_verbose)
        elif arch_type in ['03','04']:
            scheduler_all = OwnScheduler(optimizer_all, logging, mode='min', factor=sch_factor, 
                    patience= sch_patience, min_lr = sch_lmin, given_epochs = sch_given_epochs, threshold=sch_threshold, verbose=if_verbose)
                
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
        
        if arch_type in ['01','02']:
            lr = HP['learning_rates']['pressure']['l']
            logging.info('Pressure Scheduler: lr={} \t Threshold={} \t Factor={} \t Patience={}\t GivenEpochs={}'.format(lr,sch_threshold_p,sch_factor_p,sch_patience_p, sch_given_epochs))
            scheduler_p = OwnScheduler(optimizer_p, logging, mode='min', factor=schp_factor, 
                        patience=schp_patience, min_lr = sch_lmin_p, given_epochs = sch_given_epochs, threshold=schp_threshold)
        elif arch_type in ['03','04']:
            logging.info('scheduler for pressure is inside state scheduler')

    if HP['learning_rates']['params']['scheduler']:
        if inverse_problem:
            scheduler_params = []
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
            logging.info('Inverse Scheduler: lr={} \t Threshold={} \t Factor={} \t Patience={} \t GivenEpochs={}'.format(lr, schp_threshold,schp_factor,schp_patience, sch_given_epochs))
            for opt in optimizer_r_lst:
                scheduler_params.append(OwnScheduler(opt, logging, mode='min', factor=schp_factor, 
                        patience=schp_patience, min_lr = sch_lmin, given_epochs = sch_given_epochs,threshold=schp_threshold ,verbose=True))
                    

    logging.info('Starting Traning Iterations')
    logging.info('Allocated GPU Memory: {} GB'.format(round(torch.cuda.memory_allocated(device)/1e+6,2)))

    # main loop
    for epoch in range(HP['epochs']):
        for batch_idx, (x_in, y_in, z_in) in enumerate(dataloader):

            # setting the gradients to zero
            if arch_type == '01':
                nn_ux.zero_grad()
                nn_uy.zero_grad()
                nn_uz.zero_grad()
                nn_p.zero_grad()
            elif arch_type == '02':
                nn_u.zero_grad()
                nn_p.zero_grad()
            elif arch_type == '03':
                nn_all.zero_grad()

            # computing the losses
            if not divergence_free:
                loss_phys = Loss_Phys(x_in, y_in, z_in, NormalizationMode)
            else:
                loss_phys = Loss_Phys_Div(x_in, y_in, z_in, NormalizationMode)
            loss_data = Loss_Data()
            loss_bc = Loss_BC(NormalizationMode)
            loss_wk, loss_gradp = Loss_WK(NormalizationMode, windkessel_bnd, estim_bnds)
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
                        # l gradp
                        loss_gradp.backward(retain_graph=True)
                        lgradp_stats = SaveGradients(l_network_2, tracking_gradients_dict['gradp'])
                        l_network_2.zero_grad()
                        # l pmean
                        loss_p.backward(retain_graph=True)
                        lpmean_stats = SaveGradients(l_network_2, tracking_gradients_dict['pmean'])
                        l_network_2.zero_grad()
                        
    
                    if is_lambda_annealing:
                        if annealing_mode == 'max/mean':
                            HP['lambdas']['phys'] = (1-alpha_annealing)*HP['lambdas']['phys'] + alpha_annealing*lphys_stats[1]/lphys_stats[0]
                            HP['lambdas']['data'] = (1-alpha_annealing)*HP['lambdas']['data'] + alpha_annealing*ldata_stats[1]/ldata_stats[0]
                            HP['lambdas']['BC'] = (1-alpha_annealing)*HP['lambdas']['BC'] + alpha_annealing*lbc_stats[1]/lbc_stats[0]
                            HP['lambdas']['windkessel'] = (1-alpha_annealing)*HP['lambdas']['windkessel'] + alpha_annealing*lwk_stats[1]/lwk_stats[0]
                            HP['lambdas']['gradp'] = (1-alpha_annealing)*HP['lambdas']['gradp'] + alpha_annealing*lgradp_stats[1]/lgradp_stats[0]
                            HP['lambdas']['pmean'] = (1-alpha_annealing)*HP['lambdas']['pmean'] + alpha_annealing*lpmean_stats[1]/lpmean_stats[0]
                        elif annealing_mode == 'data/mean':
                            HP['lambdas']['phys'] = (1-alpha_annealing)*HP['lambdas']['phys'] + alpha_annealing*ldata_stats[1]/lphys_stats[0]
                            HP['lambdas']['data'] = (1-alpha_annealing)*HP['lambdas']['data'] + alpha_annealing*ldata_stats[1]/ldata_stats[0]
                            HP['lambdas']['BC'] = (1-alpha_annealing)*HP['lambdas']['BC'] + alpha_annealing*ldata_stats[1]/lbc_stats[0]
                            HP['lambdas']['windkessel'] = (1-alpha_annealing)*HP['lambdas']['windkessel'] + alpha_annealing*ldata_stats[1]/lwk_stats[0]
                            HP['lambdas']['gradp'] = (1-alpha_annealing)*HP['lambdas']['gradp'] + alpha_annealing*ldata_stats[1]/lgradp_stats[0]
                            HP['lambdas']['pmean'] = (1-alpha_annealing)*HP['lambdas']['pmean'] + alpha_annealing*ldata_stats[1]/lpmean_stats[0]
                        elif annealing_mode == 'search/mean':
                            max_grad = numpy.max([lphys_stats[1],ldata_stats[1],lbc_stats[1],lwk_stats[1],lgradp_stats[1],lpmean_stats[1]])
                            HP['lambdas']['phys'] = (1-alpha_annealing)*HP['lambdas']['phys'] + alpha_annealing*max_grad/lphys_stats[0]
                            HP['lambdas']['data'] = (1-alpha_annealing)*HP['lambdas']['data'] + alpha_annealing*max_grad/ldata_stats[0]
                            HP['lambdas']['BC'] = (1-alpha_annealing)*HP['lambdas']['BC'] + alpha_annealing*max_grad/lbc_stats[0]
                            HP['lambdas']['windkessel'] = (1-alpha_annealing)*HP['lambdas']['windkessel'] + alpha_annealing*max_grad/lwk_stats[0]
                            HP['lambdas']['gradp'] = (1-alpha_annealing)*HP['lambdas']['gradp'] + alpha_annealing*max_grad/lgradp_stats[0]
                            HP['lambdas']['pmean'] = (1-alpha_annealing)*HP['lambdas']['pmean'] + alpha_annealing*max_grad/lpmean_stats[0]
                        elif annealing_mode == 'mean/max':
                            HP['lambdas']['phys'] = (1-alpha_annealing)*HP['lambdas']['phys'] + alpha_annealing*(10+lphys_stats[0]/lphys_stats[1])
                            HP['lambdas']['data'] = (1-alpha_annealing)*HP['lambdas']['data'] + alpha_annealing*(10+ldata_stats[0]/ldata_stats[1])
                            HP['lambdas']['BC'] = (1-alpha_annealing)*HP['lambdas']['BC'] + alpha_annealing*(10+lbc_stats[0]/lbc_stats[1])
                            HP['lambdas']['windkessel'] = (1-alpha_annealing)*HP['lambdas']['windkessel'] + alpha_annealing*(10+lwk_stats[0]/lwk_stats[1])
                            HP['lambdas']['gradp'] = (1-alpha_annealing)*HP['lambdas']['gradp'] + alpha_annealing*(10+lgradp_stats[0]/lgradp_stats[1])
                            HP['lambdas']['pmean'] = (1-alpha_annealing)*HP['lambdas']['pmean'] + alpha_annealing*(10+lpmean_stats[0]/lpmean_stats[1])



                        PrintLambdaAnnealing(HP['lambdas'])
                        lambda_track['phys'].append(HP['lambdas']['phys'])
                        lambda_track['data'].append(HP['lambdas']['data'])
                        lambda_track['BC'].append(HP['lambdas']['BC'])
                        lambda_track['windkessel'].append(HP['lambdas']['windkessel'])
                        lambda_track['gradp'].append(HP['lambdas']['gradp'])
                        lambda_track['pmean'].append(HP['lambdas']['pmean'])
                        annealing_info['phys'].append(lphys_stats)
                        annealing_info['data'].append(ldata_stats)
                        annealing_info['BC'].append(lbc_stats)
                        annealing_info['windkessel'].append(lwk_stats)
                        annealing_info['gradp'].append(lgradp_stats)
                        annealing_info['pmean'].append(lpmean_stats)




            loss = HP['lambdas']['phys']*loss_phys + HP['lambdas']['data']*loss_data \
                         + HP['lambdas']['BC']*loss_bc + HP['lambdas']['windkessel']*loss_wk \
                         + HP['lambdas']['pmean']*loss_p + HP['lambdas']['gradp']*loss_gradp

            
            lmax = 0
            if inverse_problem:
                if range_distance_reg:
                    lmax = numpy.max([HP['lambdas']['phys'],HP['lambdas']['data'],HP['lambdas']['BC'],
                         HP['lambdas']['windkessel'],HP['lambdas']['initial'],HP['lambdas']['gradp'],HP['lambdas']['pmean']])
                    loss_params = Loss_RangeParams()
                    if loss_params:
                        loss += lmax*loss_params

            
            loss.backward()
            # one step of optimizer
            if arch_type == '01':
                optimizer_ux.step()
                optimizer_uy.step()
                optimizer_uz.step()
                optimizer_p.step()
            elif arch_type == '02':
                optimizer_u.step()
                optimizer_p.step()
            elif arch_type == '03':
                optimizer_all.step()

            if inverse_problem:
                # optimizing resistances
                if (epoch+1)%inverse_iter_rate == 0 and (epoch+1)>inverse_iter_t0:
                    for opt in optimizer_r_lst:
                        opt.step()

        # ending minibatchs
        if HP['learning_rates']['state']['scheduler']:
            if arch_type == '01':
                scheduler_ux.step(loss.item())
                scheduler_uy.step(loss.item())
                scheduler_uz.step(loss.item())
            elif arch_type == '02':
                scheduler_u.step(loss.item())
            elif arch_type in ['03','04']:
                scheduler_all.step(loss.item())

        if HP['learning_rates']['pressure']['scheduler']:
            if arch_type in ['01','02']:
                scheduler_p.step(loss.item())

        if HP['learning_rates']['params']['scheduler']:
            if inverse_problem:
                if epoch%inverse_iter_rate == 0 and (epoch+1)>inverse_iter_t0:
                    for sch in scheduler_params:
                        sch.step(loss.item(), epoch=epoch)
                
        # saving losses
        loss_track['phys'].append(loss_phys.item())
        loss_track['data'].append(loss_data.item())
        loss_track['bc'].append(loss_bc.item())
        loss_track['wk'].append(loss_wk.item())
        loss_track['pmean'].append(loss_p.item())
        loss_track['gradp'].append(loss_gradp.item())
        loss_track['tot'].append(loss.item())
        
        
        if epoch == 0:
            logging.info('Allocated GPU Memory on run: {} GB'.format(round(torch.cuda.memory_allocated(device)/1e+6,2)))

        # computing new flow
        nn_in = torch.cat((x, y, z), 1)
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
        elif arch_type == '03':
            up = nn_all(nn_in)
            ux = up[:,0]
            uy = up[:,1]
            uz = up[:,2]
            p = up[:,3]

        ux = ux.view(len(ux),-1)
        uy = uy.view(len(uy),-1)
        uz = uz.view(len(uz),-1)
        p = p.view(len(p),-1)


        if not divergence_free:
            ux_np = ux.cpu().data.numpy()
            uy_np = uy.cpu().data.numpy()
            uz_np = uz.cpu().data.numpy()
            del ux, uy, uz
            if NormalizationMode == 'elliptic':
                ux_np = uxfac[0]*ux_np + uxfac[1]
                uy_np = uyfac[0]*ux_np + uyfac[1]
                uz_np = uzfac[0]*ux_np + uzfac[1]
                
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
            del ux, uy, uz

            if NormalizationMode == 'elliptic':
                Ax = uxfac[0]/xfac[0]
                Ay = uyfac[0]/yfac[0]
                Az = uzfac[0]/zfac[0]
                ux_np = uxfac[0]*Ax*ux_np + uxfac[1]
                uy_np = uyfac[0]*Ay*uy_np + uyfac[1]
                uz_np = uzfac[0]*Az*uz_np + uzfac[1]

            utot_vec = ComputeVelocityVector(ux_np, uy_np, uz_np, u_length, Factors)
            del ux_np, uy_np, uz_np

        ptot_vec = ComputePressureVector(p, Factors)
        eps_sol = ComputeSolutionError(utot_vec,DATA['uref'], ptot_vec, DATA['pref'])
        loss_track['eps_sol'].append(eps_sol)
        # removing solution vectors
        u_fem.vector()[:] = utot_vec
        p_fem.vector()[:] = ptot_vec

            
        if save_xdmf and epoch % options['XDMF']['epoch_rate'] == 0:
            xdmf_u.write(u_fem,epoch)
            xdmf_p.write(p_fem,epoch)

        # computing the flow with FEniCS
        for k in windkessel_bnd:
            Q_fenics[k] = dolfin.assemble(dolfin.dot(u_fem, n)*ds(k))
            P_fenics[k] = dolfin.assemble(p_fem*ds(k))/dolfin.assemble(ones*ds(k))/1333.22387415
            logging.info('Bnd: {} \t flow (ml/s): {} \t pressure (mmHg): {}'.format(k,round(Q_fenics[k],2), round(P_fenics[k],2) ))
            # saving the values for postprocessing
            windkessel_track[k]['Q_fenics'].append(Q_fenics[k])
            windkessel_track[k]['P_fenics'].append(P_fenics[k])

        if inverse_problem:
            #pressure_ratio = mean_pressure_value/DATA['pwk_mean'][0]
            eps_params = ComputeParamsError(wk_r,wk_r_ref)
            loss_track['eps_params'].append(eps_params)
            logging.info('Train Epoch: {} \t Total Loss: {:.10f} \u03B5_sol: {} \u03B5_params: {}'.format(epoch + 1, round(loss.item(),5), 
                         round(eps_sol,3) , round(eps_params,3) ))
            for key in wk_r.keys():
                if param_factorization:
                    #r_estim = initial_factors[key]*numpy.power(2,wk_r[key]().item())
                    r_estim = initial_factors[key]*wk_r[key]().item()
                    param_track[key].append(r_fact*r_estim)
                    logging.info('learned resistances {:.4f}/{} '.format(round(r_estim ,3), round(wk_r_ref[key],3)))
                else:
                    param_track[key].append(r_fact*wk_r[key]().item())
                    logging.info('learned resistances {:.4f}/{} '.format(round(wk_r[key]().item(),3), round(wk_r_ref[key],3)))

        else:
            logging.info('Train Epoch: {} \t Total Loss: {:.10f} \u03B5_sol: {}'.format(epoch + 1, round(loss.item(),5), round(eps_sol,2)))



        logging.info('loss_phys: {}'.format(loss_phys.item()))
        logging.info('loss_bc: {}'.format(loss_bc.item()))
        logging.info('loss_data: {}'.format(loss_data.item()))
        logging.info('loss_wk: {}'.format(loss_wk.item()))
        logging.info('loss_pmean: {}'.format(loss_p.item()))
        logging.info('loss_gradp: {}'.format(loss_gradp.item()))
        if inverse_problem:
            if range_distance_reg:
                if loss_params:
                    logging.info('loss_params: {}'.format(loss_params.item()))
                else:
                    logging.info('loss_params: not yet')


        logging.info('---------------------')




        if epoch%250 == 0:
            if arch_type == '01':
                torch.save(nn_ux.state_dict(), options['io']['write_path'] + 'sten_data_ux.pt')
                torch.save(nn_uy.state_dict(), options['io']['write_path'] + 'sten_data_uy.pt')
                torch.save(nn_uz.state_dict(), options['io']['write_path'] + 'sten_data_uz.pt')
                torch.save(nn_p.state_dict(), options['io']['write_path'] + 'sten_data_p.pt')
            elif arch_type == '02':
                torch.save(nn_u.state_dict(), options['io']['write_path'] + 'sten_data_u.pt')
                torch.save(nn_p.state_dict(), options['io']['write_path'] + 'sten_data_p.pt')
            elif arch_type == '03':
                torch.save(nn_all.state_dict(), options['io']['write_path'] + 'sten_data_all.pt')
            
            if inverse_problem:
                for key in wk_r.keys():
                    torch.save(wk_r[key].state_dict(), options['io']['write_path'] + 'sten_data_resistances_' + str(key) + '.pt')




    toc = time.time()
    elapseTime = toc - tic
    logging.info("elapse time in hrs : {}".format(round(elapseTime/60/60,2)))
    # saving the model
    if arch_type == '01':
        torch.save(nn_ux.state_dict(), options['io']['write_path'] + 'sten_data_ux.pt')
        torch.save(nn_uy.state_dict(), options['io']['write_path'] + 'sten_data_uy.pt')
        torch.save(nn_uz.state_dict(), options['io']['write_path'] + 'sten_data_uz.pt')
        torch.save(nn_p.state_dict(), options['io']['write_path'] + 'sten_data_p.pt')
    elif arch_type == '02':
        torch.save(nn_u.state_dict(), options['io']['write_path'] + 'sten_data_u.pt')
        torch.save(nn_p.state_dict(), options['io']['write_path'] + 'sten_data_p.pt')
    elif arch_type == '03':
        torch.save(nn_all.state_dict(), options['io']['write_path'] + 'sten_data_all.pt')

    if inverse_problem:
        for key in wk_r.keys():
            torch.save(wk_r[key].state_dict(), options['io']['write_path'] + 'sten_data_resistances_' + str(key) + '.pt')

    logging.info('model data saved in {}'.format(options['io']['write_path'] + 'results'))
    # saving the losses and parameters
    numpy.savez_compressed(options['io']['write_path'] + 'loss.npz', loss_track)
    if inverse_problem:
        numpy.savez_compressed(options['io']['write_path'] + 'estimation.npz', param_track)
    
    if is_lambda_annealing:
        with open(options['io']['write_path'] + 'lambda_annealing.pickle', 'wb') as handle:
            pickle.dump(lambda_track, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(options['io']['write_path'] + 'annealing_info.pickle', 'wb') as handle:
            pickle.dump(annealing_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(options['io']['write_path'] + 'wk_track.pickle', 'wb') as handle:
        pickle.dump(windkessel_track, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if HP['learning_rates']['params']['scheduler']:
        if inverse_problem:
            scheduler_params[0].save_info(options['io']['write_path'])
        else:
            if arch_type == '01':
                scheduler_ux.save_info(options['io']['write_path'])
            elif arch_type == '02':
                scheduler_u.save_info(options['io']['write_path'])
            elif arch_type == '03':
                scheduler_all.save_info(options['io']['write_path'])


    nn_in = torch.cat((x.requires_grad_(), y.requires_grad_(), z.requires_grad_()),1)

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
    elif arch_type == '03':
        up = nn_all(nn_in)
        ux = up[:,0]
        uy = up[:,1]
        uz = up[:,2]
        p = up[:,3]
        
    ux = ux.view(len(ux),-1)
    uy = uy.view(len(uy),-1)
    uz = uz.view(len(uz),-1)
    p = p.view(len(p),-1)

    if not divergence_free:
        ux_f = ux.cpu().data.numpy()
        uy_f = uy.cpu().data.numpy()
        uz_f = uz.cpu().data.numpy()
        del ux, uy, uz
        if NormalizationMode == 'elliptic':
            ux_f = uxfac[0]*ux_f + uxfac[1]
            uy_f = uyfac[0]*uy_f + uyfac[1]
            uz_f = uzfac[0]*uz_f + uzfac[1]
            p = pfac[0]*p + pfac[1]

    else:
        Fz_y = torch.autograd.grad(uz, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        Fy_z = torch.autograd.grad(uy, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        ux_f = Fz_y - Fy_z
        ux_f = ux_f.cpu().data.numpy()
        del Fz_y, Fy_z
        Fx_z = torch.autograd.grad(ux, z, grad_outputs=torch.ones_like(z), create_graph = True,only_inputs=True)[0]
        Fz_x = torch.autograd.grad(uz, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        uy_f = Fx_z - Fz_x
        uy_f = uy_f.cpu().data.numpy()
        del Fx_z, Fz_x
        Fy_x = torch.autograd.grad(uy, x, grad_outputs=torch.ones_like(x), create_graph = True,only_inputs=True)[0]
        Fx_y = torch.autograd.grad(ux, y, grad_outputs=torch.ones_like(y), create_graph = True,only_inputs=True)[0]
        uz_f = Fy_x - Fx_y
        uz_f = uz_f.cpu().data.numpy()
        del Fy_x, Fx_y

        if NormalizationMode == 'elliptic':
            Ax = uxfac[0]/xfac[0]
            Ay = uyfac[0]/yfac[0]
            Az = uzfac[0]/zfac[0]
            ux_f = uxfac[0]*Ax*ux_f + uxfac[1]
            uy_f = uyfac[0]*Ay*uy_f + uyfac[1]
            uz_f = uzfac[0]*Az*uz_f + uzfac[1]
            p = pfac[0]*p + pfac[1]

    p = p.cpu().data.numpy()
    ux_f = ux_f*Factors['U']
    uy_f = uy_f*Factors['U']
    uz_f = uz_f*Factors['U']
    p = p*(Factors['rho']*Factors['U']**2)

    
    OUTPUT = {
        'ux' : ux_f,
        'uy' : uy_f,
        'uz' : uz_f,
        'p' : p,
        
    }
    
    numpy.savez_compressed(options['io']['write_path'] + 'output.npz', ux = OUTPUT['ux'], uy = OUTPUT['uy'], uz = OUTPUT['uz'], p = OUTPUT['p'])




def ROUTINE(options):

    logging.info(r"""

       .
      ":"
    ___:____     |"\/"|
  ,'        `.    \  /
  |  O        \___/  |
~^~^~^~^~^~^~^~^~^~^~^~^~

    """)

    DATA, CoPoints, Factors = DataReader(options)
    HyperParameters_dict = SetHyperparameters(options)
    FENICS_lst = LoadMesh(options, Factors)


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

    logging.basicConfig(format='%(levelname)s:%(message)s', filename= options['io']['write_path'] + '/run.log', level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    save_inputfile(options, inputfile)
    ROUTINE(options)
    
