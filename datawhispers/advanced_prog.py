import numpy as np
from scipy.optimize import least_squares, minimize, curve_fit
import sympy as sym
from sympy import atan as arctan
from sympy import sqrt, sin, cos, tan, exp, log, ln
import matplotlib.pyplot as plt
a, b, c, x = sym.symbols('a, b, c, x', real=True)

def linReg(x_in,y):
    '''Time series linear regression. Returns coefs in polynomial descending order.
       Coefs computed analytically.
    '''
    
    a = (np.inner(x_in,y) - (len(x_in) * np.mean(x_in) * np.mean(y))) / (np.inner(x_in,x_in) - (len(x_in) * ((np.mean(x_in))**2)))
    b = np.mean(y) - a * np.mean(x_in)
    return [a,b]

def polReg(x_in,y, deg):
    '''Time series polynomial regression. Returns coefs in polynomial descending order.
       Coefs computed numerically.
    '''
    
    coefs = np.polyfit(x_in, y, deg)
    return coefs

def freeReg(x_in, y_out, ansatz):
    '''Regression with user ansatz. The ansatz is expected to depend on three
       parameters, a, b, and c. The ansatz is expected to be a string with a 
       symbolic formulation. for instance: 'a*arctan(b*x_in+c)'.
    '''    
    test_func = sym.lambdify((x, a, b, c), eval(ansatz))
    
    res = curve_fit(test_func, x_in, y_out)

    return res[0]    


def trigReg(x_in, y):
    '''Time seriessine regression. Returns amplitude, frequency and phase
    '''    
    timestep = x_in[1]-x_in[0]
    x_in = np.fft.fftfreq(len(x_in), timestep)
    Y = np.fft.fft(y)

    index = np.argmax(abs(Y))
    
    amplitude = 2*np.absolute(Y[index])/len(x_in)
    frequenz = abs(x_in[index])
    angle = np.angle(Y[index])    
    
    coefs = np.array([amplitude, frequenz, angle])
    
    return coefs

def expReg(x_in,y):
    '''Time series exponential regression. 
    '''
    coef_first_step = linReg(x_in,np.log(y))
    b = coef_first_step[0]
    a = np.exp(coef_first_step[1])
    return [a,b]

def pred(ansatz, coef, x_in, freeRegAnsatz=None):
    '''Computes the predction for input x_in and the computed corresponding
       coefficients
    ''' 
        
    if ansatz == 'linReg' or ansatz == 'polReg':
        values = np.poly1d(coef)(x_in)     
            
    if ansatz == 'trigReg':
        amplitude, frequenz, angle = coef
        values = amplitude*np.cos(2*np.pi*frequenz*x_in+angle)  
        
    if ansatz == 'expReg':
        values = coef[0]*np.exp(coef[1]*x_in) 
        
    if ansatz == 'freeReg':
        #print(eval(freeRegAnsatz))
        f = eval(freeRegAnsatz).subs(a, coef[0]).subs(b, coef[1]).subs(c, coef[2])
        f_num = sym.lambdify(x, f)       
        values = f_num(x_in)
        
        
    return  values 

def leastSquares(func,x):
    return least_squares(func,x)


def r2(y, y_pred):
        '''Coefficient of determination
        '''
        wert = 1-np.sum((y-y_pred)**2)/np.sum((y-np.mean(y))**2)
        return wert
    

def PolyCoefficients(x, coeffs):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x**0`` to ``x**o``).
    """
    o = len(coeffs)
    print(f'# This is a polynomial of order {o}.')
    y = 0
    for i in range(o):
        y += coeffs[i]*x**i
    return y

def make_plot(x,y,y_reg, xticks=[], yticks=[],xlabel="x", ylabel="y", colors=["lightblue", "black"]):
    '''
    Outputs a graph for (x and y) and (x and y_reg) and saves it as fig_reg.png
    x: array with x-values
    y: array with y-values
    y_reg: array with regression y_values
    '''
    plt.plot(x,y_reg,color=colors[1]);
    plt.scatter(x,y, color=colors[0]);
    plt.xlabel(xlabel) 
    plt.ylabel(ylabel)
    if xticks: plt.xticks(xticks);
    if yticks: plt.yticks(yticks);
    plt.show()
    plt.savefig("fig_reg.png");    