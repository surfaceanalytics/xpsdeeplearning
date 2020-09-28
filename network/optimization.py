import numpy as np
import talos
import os
import pandas as pd
import time
import shutil
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import ticker
import seaborn as sns
from tqdm import tqdm



import xpsdeeplearning.network.classifier as classifier

class Hyperoptimization():
    def __init__(self, time, data_name):
        self.time = time
        self.data_name = data_name
        
        root_dir = os.getcwd()
        self.dir_name = self.time + '_' + self.data_name 
        self.test_dir = os.path.join(*[root_dir, 'param_tests', self.dir_name])
        
        self.scans = []
        self.full_data = pd.DataFrame()

        if os.path.isdir(self.test_dir) == False:
            os.makedirs(self.test_dir)
            if os.path.isdir(self.test_dir) == True:
                print('Test folder created at ' +\
                      str(self.test_dir.split(root_dir)[1]))
        else:
            print('Test folder was already at ' +\
                  str(self.test_dir.split(root_dir)[1]))
        
        self.save_full_data()
                  
        
            
    def initialize_clf(self, label_values):
        self.clf = classifier.ClassifierMultiple(time = self.time,
                                                 data_name = self.data_name,
                                                 labels = label_values)
        self.fig_dir = self.clf.fig_dir
        

    def hyper_opt(self, x_train, y_train, x_val, y_val, params):
        model_class = type(self.clf.model)
        model = model_class(self.clf.input_shape,
                            self.clf.num_classes,
                            params)
        model.compile(loss = params['loss_function'](),
                      optimizer = params['optimizer'](params['learning_rate']))

        out = model.fit(x = x_train,
                        y = y_train,
                        validation_data = (x_val, y_val),
                        epochs = params['epochs'],
                        batch_size = params['batch_size'],
                        verbose = 0)
    
        return out, model

    def scan_parameter_space(self, params, **kwargs):
        self.scans = self.restore_previous_scans()
        
        X_train_data = []
        X_val_data = []
        for i in range(self.clf.model.no_of_inputs):
            X_train_data.append(self.clf.X_train)
            X_val_data.append(self.clf.X_val)
            
        filenames = next(os.walk(self.test_dir))[2]
        number = 0
        for filename in filenames:
            if (filename.startswith('test_log') and 
                filename.endswith('.pkl')):
                number += 1
        
        scan = Scan(x = X_train_data,
                    y = self.clf.y_train,
                    params = params,
                    model = self.hyper_opt,
                    experiment_name = self.dir_name,
                    x_val = X_val_data,
                    y_val = self.clf.y_val,
                    val_split = 0,
                    reduction_metric = 'val_loss',
                    minimize_loss = True,
                    number = number,
                    **kwargs)
        scan.prepare(self.test_dir)
        
        self.scans.append(scan)
        
        try:
            scan.run(save_folder = self.test_dir)
            print('\n Parameter space was scanned. Training log was saved.')
            scan.deploy(self.test_dir)

        except KeyboardInterrupt:
            print('\n Scan was interrupted! Training log was saved until round {0}.'.format(self.scan.data.shape[0]))
            scan.deploy(self.test_dir)
            
        self.full_data = self.combine_results()
        self.save_full_data()
        self.all_weights = self._get_all_weights()

    def restore_previous_scans(self):
        scans = []
        filenames = [filename for filename 
                     in next(os.walk(self.test_dir))[2]
                     if (filename.startswith('test_log') 
                         and filename.endswith('pkl'))]
        
        try:
            for filename in filenames:
                scan_number = int(filename.split('.')[0][-1])
                restored_scan = RestoredScan(self.test_dir,
                                             scan_number)
                scans.append(restored_scan)
        except:
            pass
        
        return scans
            
    def combine_results(self):        
        filenames = next(os.walk(self.test_dir))[2]
        full_data = pd.DataFrame()
        for filename in filenames:
          if (filename.startswith('test_log') and filename.endswith('pkl')):
              results_file = os.path.join(self.test_dir, filename) 
              results_file = os.path.join(self.test_dir, filename)  
              new_df = pd.read_pickle(results_file)
              full_data = pd.concat([full_data, new_df], ignore_index = True)

        return full_data
    
    def save_full_data(self):
        new_csv_filepath = os.path.join(self.test_dir, 'all_tests.csv')  
        new_pkl_filepath = os.path.join(self.test_dir, 'all_tests.pkl')  

        np.savetxt(new_csv_filepath,
                   self.full_data,
                   fmt='%s',
                   delimiter=',')
                   
        self.full_data.to_pickle(new_pkl_filepath)

    def _get_all_weights(self):
        weights = [scan.saved_weights for scan in self.scans]

        return np.concatenate(weights)

    def initialize_analyzer(self):
        df = self.full_data.drop(['start', 'end', 'duration'],
                                 axis = 1,
                                 inplace = False)

        self.analyzer = Analysis(df)
        
    def get_best_params(self, metric):
        best_round = self.analyzer._get_best_round(metric)
        self.best_params = self.analyzer._get_params_from_id(best_round)
        
        return self.best_params
    
    def load_model_from_scan(self, best = False, model_id = None, metric = 'val_loss'):
        if (best == True or model_id == None):
            model_id = self.analyzer._get_best_round(metric)
            
        params = self.analyzer._get_params_from_id(model_id)
        initial_model = self.clf.model
        model_class = type(self.clf.model)
        self.clf.model = model_class(self.clf.input_shape,
                                     self.clf.num_classes,
                                     params)
        self.clf.model.compile(
            loss = params['loss_function'](),
            optimizer = params['optimizer'](params['learning_rate']))
        
        self.clf.model._name = 'Model_{0}'.format(model_id)
        self.clf.model.set_weights(self.all_weights[model_id])
        
 
class Scan():
    def __init__(self,
                 x,
                 y,
                 params,
                 model,
                 experiment_name,
                 x_val = None,
                 y_val = None,
                 val_split = .3,
                 random_method = 'uniform_mersenne',
                 seed = None,
                 performance_target = None,
                 fraction_limit = None,
                 round_limit = None,
                 time_limit = None,
                 boolean_limit = None,
                 reduction_method = None,
                 reduction_interval = 50,
                 reduction_window = 20,
                 reduction_threshold = 0.2,
                 reduction_metric = 'val_loss',
                 minimize_loss = False,
                 disable_progress_bar = False,
                 print_params = True,
                 clear_session = True,
                 save_weights = True,
                 number = 0):

        self.x = x
        self.y = y
        self.params = params
        self.model = model
        self.experiment_name = experiment_name
        self.x_val = x_val
        self.y_val = y_val
        self.val_split = val_split

        # randomness
        self.random_method = random_method
        self.seed = seed

        # limiters
        self.performance_target = performance_target
        self.fraction_limit = fraction_limit
        self.round_limit = round_limit
        self.time_limit = time_limit
        self.boolean_limit = boolean_limit

        # optimization
        self.reduction_method = reduction_method
        self.reduction_interval = reduction_interval
        self.reduction_window = reduction_window
        self.reduction_threshold = reduction_threshold
        self.reduction_metric = reduction_metric
        self.minimize_loss = minimize_loss

        # display
        self.disable_progress_bar = disable_progress_bar
        self.print_params = print_params

        # performance
        self.clear_session = clear_session
        self.save_weights = save_weights
        # input parameters section ends
        
        self.number = number
        
    def prepare(self, save_folder):
        '''
        Includes all preparation procedures up until starting the first scan
        through scan_run()
        '''
        
        self._experiment_id = time.strftime('%D%H%M%S').replace('/', '')
        _csv_filename  = 'test_log_{}.csv'.format(self.number)
        self._experiment_log = os.path.join(save_folder, _csv_filename)

        with open(self._experiment_log, 'w') as f:
            f.write('') 

        # for the case where x_val or y_val is missing when other is present
        self.custom_val_split = False
        if (self.x_val is not None and self.y_val is None) or \
           (self.x_val is None and self.y_val is not None):
            raise RuntimeError("If x_val/y_val is inputted, other must as well.")

        elif self.x_val is not None and self.y_val is not None:
            self.custom_val_split = True

        # create reference for parameter keys
        self._param_dict_keys = sorted(list(self.params.keys()))

        # create the parameter object and move to self
        from talos.parameters.ParamSpace import ParamSpace
        self.param_object = ParamSpace(
            params = self.params,
            param_keys = self._param_dict_keys,
            random_method = self.random_method,
            fraction_limit = self.fraction_limit,
            round_limit = self.round_limit,
            time_limit = self.time_limit,
            boolean_limit = self.boolean_limit)

        # mark that it's a first round
        self.first_round = True

        # create various stores
        self.round_history = []
        self.peak_epochs = []
        self.epoch_entropy = []
        self.round_times = []
        self.result = []
        self.saved_models = []
        self.saved_weights = []

        # handle validation split
        from talos.utils.validation_split import validation_split
        self = validation_split(self)

        # set data and len
        self._data_len = len(self.x)

    def run(self, save_folder):
        '''
        The high-level management of the scan procedures
        onwards from preparation. Manages round_run()
        '''
        number = self.number
        # initiate the progress bar
        self.pbar = tqdm(total=len(self.param_object.param_index),
                         disable=self.disable_progress_bar)

        # the main cycle of the experiment
        while True:

            # get the parameters
            self.round_params = self.param_object.round_parameters()

            # break when there is no more permutations left
            if self.round_params is False:
                break
            # otherwise proceed with next permutation
            from talos.scan.scan_round import scan_round
            self = scan_round(self)
            self.pbar.update(1)

        # close progress bar before finishing
        self.pbar.close()

        # finish
        from talos.logging.logging_finish import logging_finish
        self = logging_finish(self)

        from talos.scan.scan_finish import scan_finish
        self = scan_finish(self)
        
        self.number = number
        
        _pkl_filename  = 'test_log_{}.pkl'.format(self.number)
        self._experiment_log_pkl = os.path.join(save_folder, _pkl_filename)
        self.data.to_pickle(self._experiment_log_pkl)

    def deploy(self, test_dir):
        scan_folder = 'scan' + str(self.number)
        scan_dir = os.path.join(test_dir, scan_folder)
        if os.path.isdir(scan_dir) == False:
            os.makedirs(scan_dir)
        
        detail_file = os.path.join(scan_dir, 'details.txt') 
        param_file = os.path.join(scan_dir, 'params')
        weights_file = os.path.join(scan_dir, 'saved_weights')
        round_file = os.path.join(scan_dir, 'round_log.csv')
        model_file = os.path.join(scan_dir, 'saved_models.txt') 
        
        self.details.to_csv(detail_file)
        np.save(param_file, self.params)
        np.save(weights_file, self.saved_weights)
        
        with open(round_file, 'w') as roundfile:
            round_data = pd.DataFrame()
            round_data['loss'] = self.learning_entropy.loss
            round_data['start'] = self.round_times.start
            round_data['end'] = self.round_times.end
            round_data['duration'] = self.round_times.duration
            round_data['round_history'] = self.round_history
            round_data.to_csv(roundfile)    
        
        with open(model_file, 'w') as modelfile:
            for listitem in self.saved_models:
                modelfile.write('%s\n' % listitem)

        print('Data was saved to scan archive.')
        
        
class RestoredScan():
    def __init__(self, folder, number):
        self.number = number
        scan_folder = 'scan' + str(self.number)
        scan_dir = os.path.join(folder, scan_folder)
        detail_file = os.path.join(scan_dir, 'details.txt') 
        weights_file = os.path.join(scan_dir, 'saved_weights.npy')
        round_file = os.path.join(scan_dir, 'round_log.csv')
        model_file = os.path.join(scan_dir, 'saved_models.txt') 
    
        
        self.details = pd.read_csv(detail_file, 
                                    header = None, 
                                    squeeze = True, 
                                    index_col = 0)
        
        
        self.saved_weights = np.load(weights_file,
                                     allow_pickle = True)
        
        round_data = pd.read_csv(round_file)
        self.learning_entropy = pd.DataFrame(round_data['loss'])
            
        self.round_times = round_data[['start',
                                       'end',
                                       'duration']]
            
        self.round_history = list(round_data['round_history'])

        self.saved_models = []
        with open(model_file, 'r') as modelfile:
            for line in modelfile.readlines():
                self.saved_models.append(line[:-1])  
      
        self.reloaded = True
            
        print('Previous data for Scan {} was loaded!'.format(self.number))
        
        
class Analysis():
    def __init__(self, df):
        self.df = df
   
    def _get_params_from_id(self, model_id):
        params = dict(self.df.iloc[model_id])

        return params

    def _get_best_round(self, metric):
        '''
        Returns the number of rounds it took to get to the
        lowest value for a given metric.
        '''
        return self.df[self.df[metric] == self.df[metric].min()].index[0]

    def _minimum_value(self, metric):
        '''
        Returns the minimum value for a given metric'''

        return min(self.df[metric])

    def _exclude_unchanged(self):
        """
        Exlude the parameters that were not changed during the Talos Scan.
        """
        exclude = []
        for key, values in self.df.iteritems():
            if values.nunique() == 1:
                exclude.append(key)

        return exclude

    def _cols(self, metric, exclude):
        '''
        Helper to remove other than desired metric from data table
        '''

        cols = [col for col in self.df.columns if col not in exclude + [metric]]

        #for i, metric in enumerate([metric]):
        #    cols.insert(i, metric)

        # make sure only unique values in col list
        cols = list(set(cols))

        return cols

    def create_line_data(self, metric):
        return self.df[metric]

    def correlate(self, metric):
        exclude = self._exclude_unchanged()
        out = self.df[self._cols(metric, exclude)]

        out.insert(0, metric, self.df[metric])

        corr_data = out.corr(method = 'pearson')        
                
        return corr_data

    def create_hist_data(self, metric):
        return self.df[metric]
        
    def create_kde_data(self, x, y = None):
        if y != None:
            return self.df[[x, y]]
        else:
            return self.df
        
    def create_bar_data(self, x, y, hue, col):
        return self.df[[x, y, hue, col]]
        
        
class Plot():
    def __init__(self, data):
        self.data = data
        self.name = None
        self.fig = None
        
        self.font_dict = {'fontsize': 15,
                          'fontweight': 1,
                          'color': 'black',
                          'verticalalignment': 'bottom',
                          'horizontalalignment': 'center'}
                          
        display_names = {'val_acc' : 'Validation accuracy',
                         'val_loss' : 'Validation loss',
                         'loss' : 'Loss'}
                         
        try:
            self.metric = self.data.name
            self.display_name = display_names[self.metric]
        except:
            pass

    def to_file(self, filepath):
        filename = self.name + '.png'
        self.fig_dir = os.path.join(filepath, self.name)
        self.fig.savefig(self.fig_dir)
        print('File saved to test directory!')

        
        
class LinePlot(Plot):
    def __init__(self, data):
        super(LinePlot, self).__init__(data)
        self.data = data
        self.name = 'line plot_' + self.metric
        
    def plot(self):
        self.x = range(self.data.shape[0])

        self.fig, self.ax = plt.subplots(figsize = (10, 8))
               
        self.ax.plot(self.x,
                     self.data,
                     drawstyle = 'default',
                     linestyle = 'solid',
                     linewidth = 2,
                     marker = "o",
                     markersize = 7,
                     markeredgewidth = 1,
                     mfc = 'blue',
                     rasterized = True,
                     aa = True,
                     alpha = 1)
         
        self.ax.xaxis.set_major_locator(
            ticker.MaxNLocator(nbins = 5, integer = True))

        self.ax.set_xlabel('Round', self.font_dict)
        
        self.ax.set_title(self.display_name + ' across all rounds',
                          self.font_dict)
        self.ax.set_ylabel(self.display_name,
                           self.font_dict)
        

class HistPlot(Plot):
    def __init__(self, data):
        super(HistPlot, self).__init__(data)
        self.name = 'histogram_' + self.metric
    
    def plot(self, bins = 10):
        self.fig, self.ax = plt.subplots(figsize = (10, 8))
        self.ax.hist(self.data, bins = bins)
        
        self.ax.set_title(
            'Histogram of the {0} across all rounds'.format(self.display_name), 
            self.font_dict)
        
        self.ax.set_xlabel(self.display_name, 
                           fontdict = {'fontsize': 15,
                                       'fontweight': 1,
                                       'color': 'black'})
        self.ax.set_ylabel('Counts', self.font_dict)
        
        self.ax.yaxis.set_major_locator(
            ticker.MaxNLocator(integer = True))
        
        
class CorrPlot(Plot):
    def __init__(self, data):
        super(CorrPlot, self).__init__(data)
        self.name = 'correlation_plot'
    
    def plot(self):
        self.fig, self.ax = plt.subplots(figsize = (self.data.shape[0]*2.3,
                                                    self.data.shape[1]*2))
        
        p = sns.heatmap(self.data,
                        ax = self.ax,
                        linewidths= 0.1,
                        linecolor = 'white',
                        cmap = 'YlOrRd')
        self.ax.set_title('Correlation matrix of the scanned parameters', self.font_dict)
        p.set_xticklabels(self.data, rotation = 90)
        p.set_yticklabels(self.data,rotation = 0)
    
        for tick in self.ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(13)
        for tick in self.ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(13)


class KDEPlot(Plot):
    def __init__(self, data, x, y = None):
        super(KDEPlot, self).__init__(data)
        self.x = x
        self.y = y
        self.name = 'kde_plot_' + self.x
        if y != None:
            self.name += ('-' + self.y)

    def plot(self):
        self.fig, self.ax = plt.subplots(figsize = (10,10))
        
        if self.y != None:
            data2 = self.data[self.y]
        else:
            data2 = None

        p = sns.kdeplot(data = self.data[self.x],
                        data2 = data2,
                        ax = self.ax,
                        shade=True,
                        cut = 5,
                        shade_lowest = False,
                        cumulative = False,
                        kernel = 'gau',
                        bw = 'scott')

        self.ax.set_title(
            'Kernel density estimator for {0} and {1}'.format(self.x,self.y),
            self.font_dict)
    
        for tick in self.ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(13)
            tick.label.set_color('black')
        for tick in self.ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(13)
            tick.label.set_color('black')

            
class BarPlot(Plot):
    def __init__(self, data, x, y, hue, col):
        super(BarPlot, self).__init__(data)
        self.x = x
        self.y = y
        self.hue = hue
        self.col = col
        self.name = ('bar_plot-' +
                     self.x + '-' +
                     self.y + '-' +
                     self.hue + '-' +
                     self.col)

    def plot(self):
        #self.fig, self.ax = plt.subplots(figsize = (10,10))
        
        self.fig = sns.catplot(data = self.data,
                               x = self.x,
                               y = self.y,
                               hue = self.hue,
                               col = self.col,
                               col_wrap = 4,
                               legend = True,
                               legend_out = True,
                               size = 4,
                               kind = 'bar')

        self.fig.fig.suptitle(
            'Box plot for {0}, {1}, {2}, and {3}'.format(
                self.x, self.y, self.hue, self.col))