import scipy.optimize as optimize


def f(x,b):

	b0=b[1]
	b1=b[2]
	b2=b[3]
	b3=b[4]
	b4=b[5]

	#Weights
	w_energy = 0.9
	w_time = 1 - w_energy
	
	x1,x2,x3,x4 = x
	
	return w_energy*(b0+b1*x1+0.05*x1+b2*x2+0.03*x2+b3*x3+b4*x4)+w_time*(x3/x4)

def main_optimize(b,method):
	#print(b)
	v_in = 0.5
	v_ut = 0.5
	s=10
	v_medel = 1

	initial_guess = [v_in, v_ut, s, v_medel]
	result = optimize.minimize(f, initial_guess, args=b ,method='SLSQP', bounds=((0.1,2),(0.1, 2),(s,s),(0, 3)))

	if result.success:
		fitted = result.x
		energyc = b[1]+b[2]*fitted[0]+b[3]*fitted[1]+b[4]*fitted[2]+b[5]*fitted[3]
		timec = fitted[2]/fitted[3]
				
		print("---------------\nEnergy Consumption using {}: {}".format(method, energyc))
		print("Time on the straight road: {}".format(timec))
		#print(fitted_params)
	else:
		raise ValueError(result.message)
	
	return fitted
	