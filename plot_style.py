# This is a library for creating different matplotlib plots
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

# For 2 x 2 subplot grid 

def plot_style22(data,title,xlabel,ylabel):
	plt.rc('axes', titlesize=14)
	plt.rc('axes', labelsize=12)
	colorPlot = np.array([['ob','og'],['or','ok']])

	fig, ax = plt.subplots(2,2)
	for i in range(0,2):
		for j in range(0,2):
			if i == 0 and j == 0:
				data_plot = data[0]
			if i == 0 and j == 1:
				data_plot = data[1]
			if i == 1 and j == 0:
				data_plot = data[2]
			if i == 1 and j == 1:
				data_plot = data[3]

			ax[i][j].plot(data_plot[:,1],data_plot[:,0],colorPlot[i,j],markersize=1)
			ax[i][j].set_xlabel(xlabel)
			ax[i][j].set_xlim(left = 0)
			ax[i][j].set_ylabel(ylabel)
			ax[i][j].set_ylim(bottom = 0)
			ax[i][j].set_title(title[i,j])
			ax[i][j].grid()

	fig.tight_layout(pad=1)

# For single plot

def plot_style1(X,Y,title,xlabel,ylabel):
	plt.rc('axes', titlesize=14)
	plt.rc('axes', labelsize=12)

	fig, ax = plt.subplots()
	ax.plot(X,Y,'-k')
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	#ax.set_xlim(left=0)
	#ax.set_ylim(bottom=0)
	ax.set_title(title)
	ax.grid()

	fig.tight_layout(pad=1)

# For overlaying fitting equation on data 

def plot_style1_curve_fitting(X,Y,label,Xfit1,Yfit1,label_fit1,Xfit2,Yfit2,label_fit2,title,xlabel,ylabel,markersize):
	plt.rc('axes', titlesize=14)
	plt.rc('axes', labelsize=12)

	fig, ax = plt.subplots()
	ax.plot(X,Y,'ok',markersize=markersize,label=label)
	ax.plot(Xfit1,Yfit1,'-b',label=label_fit1)
	ax.plot(Xfit2,Yfit2,'-r',label=label_fit2)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_xlim(left=0)
	ax.set_ylim(bottom=0)
	ax.set_title(title)
	ax.legend()
	ax.grid()

	fig.tight_layout(pad=1)

# Creating 3 dimensional plots using plotly

def plot_style1_3D(zdata,xdata,ydata,zlabel,xlabel,ylabel,title):
    
        fig = go.Figure(data=[go.Surface(z = zdata, x = xdata, y = ydata, showscale = True, colorscale = 'blueRed')])
        fig.update_layout(title={'text':title,'y':0.9,'x':0.5,'xanchor':'center','yanchor':'top'},
                                 scene = dict(xaxis_title=xlabel,yaxis_title=ylabel,zaxis_title=zlabel),
                                 font = dict(size = 12,family = 'Cambria'))
        return fig
