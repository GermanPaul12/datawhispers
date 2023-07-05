import numpy as np
from scipy.optimize import least_squares, minimize, curve_fit
import sympy as sym
from sympy import atan as arctan
from sympy import sqrt, sin, cos, tan, exp, log, ln
import matplotlib.pyplot as plt
a, b, c, x = sym.symbols('a, b, c, x', real=True)
# 
def linReg(x_in,y):
    '''Time series linear regression. Returns coefs in polynomial descending order.
       Coefs computed analytically.
       
        Args:
            x_in: Array with x-values
            y: Array with y-values

        Returns:
            coefs in descending order

        Raises:
            None
    '''
    
    a = (np.inner(x_in,y) - (len(x_in) * np.mean(x_in) * np.mean(y))) / (np.inner(x_in,x_in) - (len(x_in) * ((np.mean(x_in))**2)))
    b = np.mean(y) - a * np.mean(x_in)
    return [a,b]

def polReg(x_in,y, deg):
    '''Time series polynomial regression. Returns coefs in polynomial descending order.
       Coefs computed numerically.
       
       Args:
            x_in: Array with x-values
            y: Array with y-values
            deg: the degree of the polynomial

        Returns:
            coefs in descending order

        Raises:
            None
    '''
    
    coefs = np.polyfit(x_in, y, deg)
    return coefs

def freeReg(x_in, y_out, ansatz):
    '''Regression with user ansatz. The ansatz is expected to depend on three
       parameters, a, b, and c. The ansatz is expected to be a string with a 
       symbolic formulation. for instance: 'a*arctan(b*x_in+c)'.
       
       Args:
            x_in: Array with x-values
            y: Array with y-values
            ansatz: "linReg", "polReg", "trigReg" or "expReg"

        Returns:
            coefs

        Raises:
            None
    '''    
    test_func = sym.lambdify((x, a, b, c), eval(ansatz))
    
    res = curve_fit(test_func, x_in, y_out)

    return res[0]    


def trigReg(x_in, y):
    '''Time seriessine regression. Returns amplitude, frequency and phase
    
        Args:
            x_in: Array with x-values
            y: Array with y-values

        Returns:
            amplitude, frequency and phase

        Raises:
            None
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

def predict(ansatz, coef, x_in, freeRegAnsatz=None):
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
    '''Solve a nonlinear least-squares problem with bounds on the variables.
    
        Args:
            func: Function which computes the vector of residuals, 
            with the signature fun(x, *args, **kwargs), i.e., the minimization proceeds with respect to its first argument. 
            The argument x passed to this function is an ndarray of shape (n,) (never a scalar, even for n=1). 
            It must allocate and return a 1-D array_like of shape (m,) or a scalar. 
            If the argument x is complex or the function fun returns complex residuals, 
            it must be wrapped in a real function of real arguments, as shown at the end of the Examples section.
            x: Array with x-values
            

        Returns:
            x: ndarray, shape (n,)
                Solution found.

            cost: float
                Value of the cost function at the solution.

            func: ndarray, shape (m,)
                Vector of residuals at the solution.

        Raises:
            None
            
    '''
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

def make_plot(x,y,y_reg, xticks=[], yticks=[],xlabel="x", ylabel="y", colors=["lightblue", "black"], name="fig_reg.png"):
    '''Outputs a graph for (x and y) and (x and y_reg) and saves it as fig_reg.png
    
    Args:
        x: array with x-values
        y: array with y-values
        y_reg: array with regression y_values
        xticks (optional): list with values to use as x-ticks
        yticks (optional): list with values to use as y-ticks
        xlabel (optional): defualt "x"
        ylabel (optional): defualt "y"
        colors (optional): default ["lightblue", "black"] scatter=lightblue and line=black
        name (optional): default "fig_reg.png"
        
    Returns:
        Outputs the graph and saves it
        
    Raises:
        None
    '''
    plt.plot(x,y_reg,color=colors[1]);
    plt.scatter(x,y, color=colors[0]);
    plt.xlabel(xlabel) 
    plt.ylabel(ylabel)
    if xticks: plt.xticks(xticks);
    if yticks: plt.yticks(yticks);
    plt.savefig(f"{name}");  
    plt.show()


def show_mnist_from_array(arr):    
    """Returns the image of the mnist number and saves it as mnist_num.png
    
    Args:
        arr: of size (784,) or (28,28) with values from 0 to 255
        
    Returns:
        Outputs the image
        
    Raises:
        None
    """
    label = ''
    if arr.shape == (785,): label,arr = arr[0],arr[1:].reshape((28, 28))
    if arr.shape == (784,):   
        arr = arr.reshape((28, 28))
    # Plot
    plt.title(f"MNIST Number {label}")
    plt.imshow(arr, cmap='gray')
    plt.savefig("mnist_num.png", dpi=1200)
    plt.show()
    

def show_mnist_from_file(filepath):    
    """Returns the images of the mnist numbers in the file
    
    Args:
        filepath: csv-filepath with lines consisting of values from 0 to 255 with length of 785 or 784
        
    Returns:
        Outputs the images
        
    Raises:
        None
    """
    with open(filepath) as f: 
        try:
            label = ""
            for i in f:
                if "," in i:   
                    arr = np.array([int(num) for num in i.split(",")])
                    if arr.shape == (785,): label,arr = arr[0],arr[1:].reshape((28, 28))
                    if arr.shape == (784,): arr = arr.reshape((28, 28))
                else:    
                    arr = np.array([int(num) for num in i.split(";")])
                    if arr.shape == (785,): label,arr = arr[0],arr[1:].reshape((28, 28))
                    if arr.shape == (784,): arr = arr.reshape((28, 28))
                plt.title(f"MNIST Number {label}")    
                plt.imshow(arr, cmap="gray")
                plt.show()   
        except Exception as e:
            print("Sorry try the func 'show_mnist_from_array' because your file does not seem to work with this method")         


def add_mnist_num_arrays(num1,num2):
    """Returns the image of the result and saves it as mnist_result.png
    
    Args:
        num1: np.array of length (784,) or (28,28)
        num2: np.array of length (784,) or (28,28)
        
    Returns:
        Outputs the image
        
    Raises:
        None
    """        
    if num1.shape == (784,):
        num1 = num1.reshape((28,28))
    if num2.shape == (784,):
        num2 = num2.reshape((28,28))
    result = num1 + num2
    plt.savefig("mnist_result.png")
    plt.imshow(result) 
    


class Trend:
    """ Trends Class. Trend objects have values and method attributes.
    
    """    
    

    def __init__(self, x,
                       y,
                       ansatz,
                       deg = None,
                       ):
        '''Initialization of Trend with training input, training output,
           ansatz (string) and deg (if polynomial ansatz)
        '''
        self.x = x
        self.y = y
        self.ansatz = ansatz
        self.deg = deg
        self.coef = self.coef()
        self.r2 = self.r2()
        
    
    def coef(self):
        '''Computes coefficients of corresponding ansatz
        '''
    
        if self.ansatz == 'linReg':
            
            coef = linReg(self.x, self.y)
            
        if self.ansatz == 'polReg':
            
            coef = polReg(self.x, self.y, deg = self.deg) 
            
        if self.ansatz == 'trigReg':
            
            coef = trigReg(self.x, self.y)
            
        if self.ansatz == 'expReg':
            
            coef = expReg(self.x, self.y)            
    
        return coef

    
    
    def pred(self, x):
        '''Computes the predction for input x and the computed corresponding
           coefficients
        ''' 
        
        values = predict(self.ansatz, self.coef, x)            
        
    
        return  values      

    
    def r2(self):
        '''Computes the coefficient of determination for the training input
        ''' 
        wert=r2(self.y, self.pred(self.x))
        return round(wert, 4)        
    
    
    def make_easy_plot(self, file_name):
        '''Shows a plot of the data, the regression and saves the plot
        '''
        make_plot(x=self.x, y=self.y, y_reg=predict(self.ansatz, self.coef, self.x), name=file_name)
        print(f"r2: {self.r2}, coefs: {self.coef}")
    
    
def plot_all_regs(x,y, xticks=None, yticks=None):
    """
    Returns the regression of all types and saves them as png
    
    Args:
        x: array with x-values
        y: array with y-values
        xticks (optional): list with values to use as x-ticks
        yticks (optional): list with values to use as y-ticks
        
    Returns:
        Outputs the graphs and saves them
        
    Raises:
        None
    """ 
    model = Trend(x,y,"linReg")
    plt.title("Linear Regression");
    y_reg = model.pred(x)
    make_plot(x,y,y_reg, name="lin_reg_plot.png");
    print(f"Coefs: {model.coef}, r2: {model.r2}")
    
    for i in range(2,10):
        model = Trend(x,y,"polReg", deg=i)
        plt.title(f"Polynomial Regression {i} degree");
        y_reg = model.pred(x)
        make_plot(x,y,y_reg, name=f"pol_reg{i}.png");
        print(f"Coefs: {model.coef}, r2: {model.r2}")
    
    model = Trend(x,y,"trigReg")
    plt.title("Trigonometric Regression");
    y_reg = model.pred(x)
    make_plot(x,y,y_reg, name="trig_reg.png");
    print(f"Coefs: {model.coef}, r2: {model.r2}")   
    
    model = Trend(x,y,"expReg")
    plt.title("Exponential Regression");
    y_reg = model.pred(x)
    make_plot(x,y,y_reg, name="exp_reg.png");
    print(f"Coefs: {model.coef}, r2: {model.r2}")   
        