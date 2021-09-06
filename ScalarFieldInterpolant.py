import scipy.interpolate as sint
import numpy as np
    
class scalar_field_interpolant():
    def __init__(self, XY, fields):
        self.XY = XY
        self.fields = fields
    
    def __call__(self, x, y):
        shape = x.shape
        x = x.flatten()
        y = y.flatten()
        interpolation = np.zeros((x.size, self.fields.shape[1]))
        
        for i in range(self.fields.shape[1]):
            interpolation[:, i] = sint.griddata(self.XY, self.fields[:, i], (x, y),  method = 'cubic')
            out_of_bounds = interpolation[:, i] != interpolation[:, i]
            if np.sum(out_of_bounds) != 0:
                interpolation[out_of_bounds, i] = sint.griddata(self.XY, self.fields[:, i], (x[out_of_bounds], y[out_of_bounds]),  method = 'nearest')
        
        return interpolation.reshape(shape + (self.fields.shape[1], ))
