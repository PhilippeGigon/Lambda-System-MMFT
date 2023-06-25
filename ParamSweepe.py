import numpy as np
import matplotlib.pyplot as plt
import math as math
import scipy as sp


class ParamSweepe:
    '''
    Class to store results of parameter sweepe 
    and to visualize the collision angles 
    '''

    def __init__(self, collisionanglesComplete, spectrange, Plot_spectrum, Plot_spectrum_Color,Plot_spectrum_Floquet_Band, Parameter_range, order=-1,fs=18):
        self.Parameter_range = Parameter_range  # Sweeped parameter (array)
        if order == -1:
            # order in perturbation theory
            self.order = collisionanglesComplete.shape[1]-1
        else:
            self.order = order
        self.fs=fs
        # Collision angle at every order
        self.collisionanglesComplete = collisionanglesComplete
        #Spectrum of reduced  FS Hamiltonian
        self.spectrange = spectrange
        #Those variables contain the information to color the spectrum in accordance with the angles
        self.Plot_spectrum = Plot_spectrum
        self.Plot_spectrum_Color = Plot_spectrum_Color
        self.Plot_spectrum_Floquet_Band=Plot_spectrum_Floquet_Band
    ################################################
    def visualize_Fixed_Parameter(self, visualize_parameter):
        """
        Method to do the barplots of collision angle
        vs index of level, for a fixed parameter (for debugging reasons)
        """
        # Find index closest to parameter
        minposition = np.argmin(
            np.abs(self.Parameter_range-visualize_parameter))
        for orde in range(self.order+1):
            plt.figure()
            plt.bar(np.arange(len(
                self.collisionanglesComplete[minposition, orde, :])), self.collisionanglesComplete[minposition, orde, :])
            plt.yticks([0, math.pi/4, math.pi/2],
                       [r'0', r'$\pi/4$', r'$\pi/2$'], fontsize=14)
            plt.tick_params(axis='both', which='major', labelsize=14)
            plt.title("Collision angles at order "+str(orde), fontsize=self.fs)
            plt.xlabel(r"Band", fontsize=self.fs)
            plt.ylabel(r"$\theta$", fontsize=self.fs)

################################################
    def visualize_pretty(self, ymin, ymax,Nmin=0,Nmax=-1, axisLabel=r"Parameter", Savefigure=False):
            '''
            Method to visualize the collision angles at every order
            '''
            
            fstickslabels=self.fs-4
            #The last color is transparent to hide unwanted things in the plot
            colormax = np.array([(51/256, 101/256, 138/256,1), (160/256, 40/256,41/256,1), (134/256, 187/256, 216/256,1),(151/256, 161/256, 105/256,1),(0, 0, 0, 0)])
            if (Nmax-Nmin)>len(colormax)-1: print("Not enough colors!")
            for Perturb_order in range(self.order+1):
                layout = [
                ["a)"],
                ["b)"]]
                fig, axd = plt.subplot_mosaic(layout, figsize=(6, 5))            
                
                axd['a)'].set_xticks([])
                axd['b)'].set_yticks([])
                axd['a)'].set_yticks([0, math.pi/4, math.pi/2],
                     [r'0', r'$\pi/4$', r'$\pi/2$'], fontsize=fstickslabels)
                axd['b)'].set_yticks(np.arange(ymin,ymax),np.arange(ymin,ymax), fontsize=fstickslabels)
                axd['a)'].yaxis.tick_right()
                axd['b)'].yaxis.tick_right()

                axd['a)'].set_ylabel(r"Collision angle", fontsize=self.fs)
                axd['b)'].set_ylabel(r"Spectrum", fontsize=self.fs)
                axd['a)'].set_ylim([0, math.pi/2+0.1])
                axd['b)'].set_xlabel(axisLabel, fontsize=self.fs)
                axd['b)'].tick_params(axis='both', which='major', labelsize=fstickslabels)
                axd['b)'].set_ylim([ymin, ymax])

                # Plot floquet energy bands from exact diagonalization
                for j in range(len(self.spectrange[0])):
                    axd['b)'].plot(self.Parameter_range, self.spectrange[:, j],
                                '-', color='black', alpha=0.1)
                #Plot the band we're interested in
                axd['b)'].plot(self.Parameter_range,np.zeros_like(self.Parameter_range),'-', color='black')

                sortedanges = np.argsort(np.mean(self.collisionanglesComplete[:, -1, :],axis=0))[::-1][Nmin:Nmax]

                # Plot the maximala collision angles
                for indj, j in enumerate(sortedanges[0:Nmax]):
                    axd['a)'].plot(self.Parameter_range, self.collisionanglesComplete[:, Perturb_order,
                                j], '-',color=colormax[indj], linewidth=2, markersize=1.5,alpha=1)

                # Color the Floquet bands which have a large overlap with the colliding non perturbed states
                for band, plot_ in enumerate(self.Plot_spectrum[:, Perturb_order, :].T):
                    C = self.Plot_spectrum_Floquet_Band[plot_, Perturb_order, band]
                    for enum, Fband in enumerate(sortedanges[:Nmax]):
                        C[C == Fband] = enum
                    spectband = self.spectrange[:, band]
                    C[C > np.abs(len(sortedanges[Nmin:Nmax]))] = -1
                    C = np.array(colormax[C], dtype=tuple)
                    axd['b)'].scatter(self.Parameter_range[plot_],spectband[plot_],color=C, s=4)
                axd['a)'].set_title("Order: "+str(Perturb_order),fontsize=self.fs)
                
                fig.subplots_adjust(wspace=0.05, hspace=0)

                #Save figure
                if Savefigure:
                    plt.savefig('Collision_Order='+str(Perturb_order) +
                                ".png", bbox_inches='tight', dpi=600)