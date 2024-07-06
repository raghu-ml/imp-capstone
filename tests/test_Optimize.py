# https://forms.gle/rUGekP5saJeTM5S89
import unittest
import numpy as np
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from parameterized import parameterized

PATH = '/Users/raghuramnarayanan/Library/CloudStorage/OneDrive-Personal/STUDY/ML/capstone/initial_data/initial_data'
NEW_PATH = '/Users/raghuramnarayanan/Library/CloudStorage/OneDrive-Personal/STUDY/ML/capstone/588_data.csv'

from bayes_opt.simple_opt import SimpleBayesOptimizer
from bayes_opt.util import UtilityFunction

import pandas as pd

import logging
logging.basicConfig()

# By default the root logger is set to WARNING and all loggers you define
# inherit that value. Here we set the root logger to NOTSET. This logging
# level is automatically inherited by all existing and new sub-loggers
# that do not set a less verbose level.
logging.root.setLevel(logging.DEBUG)

# The following line sets the root logger level as well.
# It's equivalent to both previous statements combined:
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

import re
REGEX = '(0\.[\s\d]{5,6}|\d.\d\d\d\d\de-\d\d)+'

F1 = 'function_1'
F2 = 'function_2'
F3 = 'function_3'
F4 = 'function_4'
F5 = 'function_5'
F6 = 'function_6'
F7 = 'function_7'
F8 = 'function_8'

import datetime

class TestCapstone(unittest.TestCase):

    def test_regex(self):
        
        x = self._f_inp_to_array('[0.449672 0.977574]')
        print( x )
        self.assertEqual(x[0], 0.449672)
        self.assertEqual(x[1], 0.977574)

        x = self._f_inp_to_array('[0.        0.977574]')
        print( x )
        self.assertEqual(x[0], 0)
        self.assertEqual(x[1], 0.977574)

        x = self._f_inp_to_array('[0.977574 0.   ]')
        print( x )
        self.assertEqual(x[0], 0.977574)
        self.assertEqual(x[1], 0.0)
        

        x = self._f_inp_to_array('[9.99999e-01 1.00000e-06]')
        print( x )
        self.assertEqual(x[0], 9.99999e-01)
        self.assertEqual(x[1], 1.00000e-06)
        
        x = self._f_inp_to_array('[0.00115 0.0095]')
        print( x )
        self.assertEqual(x[0], 0.00115)
        self.assertEqual(x[1], 0.0095)

    
    def _f_inp_to_array(self, inp_str):
        if isinstance(inp_str, np.ndarray):
            return inp_str
        else:
            inp_str = inp_str[1:] if inp_str[0] == '[' else inp_str
            inp_str = inp_str[:-1] if inp_str[-1] == ']' else inp_str
            parts = inp_str.split(" ")
            final = []
            for _p in parts:
                try:
                    num = float(_p)
                    final.append(num)
                except:
                    pass
                
                    
            return final
            # m = re.findall(REGEX, inp_str)
            # return np.array([float(_) for _ in m])

    # def _ts_to_inp_array(self, inp_str):
    #     if isinstance(inp_str,(pd.Timestamp,)):
    #         return inp_str.to_pydatetime()
    #     else:
    #         return datetime.datetime.strptime(inp_str, '%Y-%m-%d %H:%M')

    def setUp(self) -> None:
        self.NEW_DATA = self.__pop_new_data()    
        

    def __pop_new_data(self):
        logger.info("New data found")

        new_data_df = pd.read_csv(NEW_PATH)

        # duplicate
        new_data_df.drop(new_data_df.index[2], inplace=True)


            
        # new_data_df.timestamp = new_data_df.timestamp.apply(_ts_to_inp_array)
        new_data_df.f1 = new_data_df.f1.apply(self._f_inp_to_array)
        new_data_df.f2 = new_data_df.f2.apply(self._f_inp_to_array)
        new_data_df.f3 = new_data_df.f3.apply(self._f_inp_to_array)
        new_data_df.f4 = new_data_df.f4.apply(self._f_inp_to_array)
        new_data_df.f5 = new_data_df.f5.apply(self._f_inp_to_array)
        new_data_df.f6 = new_data_df.f6.apply(self._f_inp_to_array)
        new_data_df.f7 = new_data_df.f7.apply(self._f_inp_to_array)
        new_data_df.f8 = new_data_df.f8.apply(self._f_inp_to_array)

        new_data_df.sort_values(by='timestamp', inplace=True)

        new_data = {}

        X1 = np.vstack(new_data_df.f1.values)
        Y1 = new_data_df.f1_output.values
        new_data[F1]= (X1,Y1)
        assert(X1.shape[0] == len(Y1))
        assert(X1.shape[1] == 2)
        

        X2 = np.vstack(new_data_df.f2.values)
        Y2 = new_data_df.f2_output.values
        new_data[F2]= (X2,Y2)
        assert(X2.shape[0] == len(Y2))
        assert(X2.shape[1] == 2)

        X3 = np.vstack(new_data_df.f3.values)
        Y3 = new_data_df.f3_output.values
        new_data[F3]= (X3,Y3)
        assert(X3.shape[0] == len(Y3))
        assert(X3.shape[1] == 3)

        X4 = np.vstack(new_data_df.f4.values)
        Y4 = new_data_df.f4_output.values
        new_data[F4]= (X4,Y4)
        assert(X4.shape[0] == len(Y4))
        assert(X4.shape[1] == 4)

        X5 = np.vstack(new_data_df.f5.values)
        Y5 = new_data_df.f5_output.values
        new_data[F5]= (X5,Y5)
        assert(X5.shape[0] == len(Y5))
        assert(X5.shape[1] == 4)

        X6 = np.vstack(new_data_df.f6.values)
        Y6 = new_data_df.f6_output.values
        new_data[F6]= (X6,Y6)
        assert(X6.shape[0] == len(Y6))
        assert(X6.shape[1] == 5)
        
        X7 = np.vstack(new_data_df.f7.values)
        Y7 = new_data_df.f7_output.values
        new_data[F7]= (X7,Y7)
        assert(X7.shape[0] == len(Y7))
        assert(X7.shape[1] == 6)

        X8 = np.vstack(new_data_df.f8.values)
        Y8 = new_data_df.f8_output.values
        new_data[F8]= (X8,Y8)
        assert(X8.shape[0] == len(Y8))
        assert(X8.shape[1] == 8)

        return new_data


    def _get_initial_data(self, functionName):
        X_initial = np.load(PATH+'/'+functionName+'/initial_inputs.npy')
        y_initial = np.load(PATH+'/'+functionName+'/initial_outputs.npy') 
        return X_initial, y_initial
    
    def _get_new_data(self, functionName):
        return self.NEW_DATA.get(functionName, None)
    
    def _get_data(self, functionName):
        X_initial, y_initial = self._get_initial_data(functionName)

        logger.info("{:} : Initial X - {:}".format(functionName, X_initial.shape))
        logger.info("{:} : Initial Y - {:}".format(functionName, y_initial.shape))

        new_data = self._get_new_data(functionName)
        if new_data:
            logger.info("{:} : New X - {:}".format(functionName, new_data[0].shape))
            logger.info("{:} : New Y - {:}".format(functionName, new_data[1].shape))

            X_new = np.vstack([X_initial, new_data[0]])
            Y_new = np.concatenate([y_initial, new_data[1]])
            return ( X_new , Y_new )        
        else:
            logger.info("{:} : No New Data".format(functionName))
            return X_initial, y_initial
        
    def _format_suggestion(self, suggestion):
        return "-".join( ["{:0.6f}".format(s) for s in suggestion] )
        
    @parameterized.expand([ ("function_1", {'kind':'ei'}, (RBF,),  [] ), #np.array( [[0.77,0.8],[0.64,0.75]])
                           ("function_2", {'kind':'ei'},  (), []),
                           ("function_3", {'kind':'ei'},  (), []),
                           ("function_4", {'kind':'ei'},  (), []),
                           ("function_5", {'kind':'poi'}, (RBF,), [] ),
                           ("function_6", {'kind':'ei'},  (), []),
                           ("function_7", {'kind':'ei'},  (), []),
                           ("function_8", {'kind':'ei'},  (), []) ] )
    def test_Functions(self, functionName, acqArgs, kernelArgs, bounds):

        X,y = self._get_data(functionName)
        
        logger.info("{:} : ACQ: {:}".format(functionName, acqArgs))

        # acq = UtilityFunction('ucb') # explore more
        # acq = UtilityFunction('poi', xi = 1.5) # explore more
        acq = UtilityFunction(**acqArgs) 

        if not kernelArgs:
            length_scale_bounds = [(1e-8, 1e7)]
            kernel = Matern(nu=2.5)
            logger.info("{:} : KERNEL DEFAULT : {:}".format(functionName, kernel))
        else:
            kernel = kernelArgs[0](length_scale=1, length_scale_bounds=(1e-10, 1e20))
            logger.info("{:} : KERNEL SPECIFIED : {:}".format(functionName, kernel))

        f1_opt = SimpleBayesOptimizer(acq, kernel)
        # fit the model
        f1_opt.fit(X,y)


        # find the max
        max_idx = np.argmax(y)
        max_y = y[max_idx]
        max_x = X[max_idx]

        logger.info("{:} : max y {:}".format(functionName, max_y))
        logger.info("{:} : max x {:}".format(functionName, max_x))

        # search bounds
        if not len(bounds):
            bounds = np.ndarray((X.shape[1],2))
            bounds[:,0] = 0.0
            bounds[:,1] = 1.0

        logger.info("{:} : bounds {:}".format(functionName, bounds))
        
        # find next
        suggestion = f1_opt.suggest(max_y, max_x, bounds)

        suggestion_str = self._format_suggestion(suggestion)

        logger.info("{:} : suggestion : {:}".format(functionName, suggestion_str))
