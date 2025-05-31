#!/usr/bin/env python3
# Numerical packages
import numpy as np
import sympy as sy
import scipy as sp
import pandas as pd
# Graphical packages
import matplotlib.pyplot as plt
#import seaborn as sns
import matplotlib as mpl 
# Other-tools packages
from itertools import count
from cycler import cycler
import time
import IPython
IPython.display.clear_output
#!/usr/bin/env python
sy.init_printing()
np.set_printoptions(precision=3)
# avoiding innaccurate floating points
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf, linewidth=100)



def main():
    # --------------------------------------------------------------------------------
    # A continuacion se presenta el codigo para resolver el caso de placa plana 2D
    # estacionario, sin fuentes de calor y con TODAS las condiciones tipo Dirichlet
    # Adquisicion de datos de simulacion --> Pandas
    # Ojo, el archivo con los datos de la simulacion es un archivo plano de texto
    # --------------------------------------------------------------------------------
    simulationData = pd.read_csv("thermalDiffusionDynamic2DPlate-data.dat")
    simulationData.to_numpy()
    simulationData.columns = simulationData.columns.str.strip()
    simulationData.set_index('Variable',inplace=True)
    print(simulationData)
    print("-------------- \n ")
    # --------------------------------------------------------------------------------
    nNodesX       = np.int32(simulationData.loc['nNodesX']['Value'])
    nNodesY       = np.int32(simulationData.loc['nNodesY']['Value'])
    lenXPlaca     = np.float64(simulationData.loc['lenXPlaca']['Value'])
    lenYPlaca     = np.float64(simulationData.loc['lenYPlaca']['Value'])
    highTemp      = np.float64(simulationData.loc['highTemp']['Value'])
    lowTemp       = np.float64(simulationData.loc['lowTemp']['Value'])
    simulationTime= np.float64(simulationData.loc['simulationTime']['Value'])
    deltaT        = np.float64(simulationData.loc['deltaT']['Value'])
    nPlots        = np.float64(simulationData.loc['nPlots']['Value'])
    # --------------------------------------------------------------------------------
    nTotalN = np.int32((nNodesX-2)*(nNodesY-2))
    lenX    = lenXPlaca
    lenY    = lenYPlaca
    deltaX  = np.float64(lenX/np.float64(nNodesX-1))
    deltaY  = np.float64(lenY/np.float64(nNodesY-1))
    alpha   = np.float64(0.5*deltaT/(deltaX**2))
    beta    = np.float64(0.5*deltaT/(deltaY**2))
    gamma   = np.float64(2.0*(alpha+beta))
    nTsteps = np.int32(simulationTime/deltaT)
    plotNumber = np.int32(0)
    printEvery = np.int32(np.round(nTsteps/nPlots))
    # --------------------------------------------------------------------------------
    # Construccion del vector de condiciones iniciales... toda la placa en Tlow
    vecTempActual = lowTemp*np.ones(nTotalN,dtype=np.float64)
    # --------------------------------------------------------------------------------
    # Inicializacion de matrices para representacion grafica/visual ------------------
    matTemp = np.zeros((nNodesY,nNodesX),dtype=np.float64) # Matriz de temperaturas
    coordsX = np.zeros((nNodesY,nNodesX),dtype=np.float64) # Matriz de coordsX
    coordsY = np.zeros((nNodesY,nNodesX),dtype=np.float64) # Matriz de coordsY
    # --------------------------------------------------------------------------------
    # Llenado de matrices para representacion grafica/visual ------------------------
    # OJO: Solo valores internos... no los de frontera       ------------------------
    for j in range(nNodesY-2):
        for i in range(nNodesX-2):
            matTemp[-(j+2),i+1] = vecTempActual[i+j*(nNodesX-2)]
            coordsX[-(j+2),i+1] = deltaX*(i+1)
            coordsY[-(j+2),i+1] = deltaY*(j+1)

    # --------------------------------------------------------------------------------
    # Llenado de matrices para representacion grafica/visual -------------------------
    # OJO:  Llenado manual de los valores de frontera --------------------------------
    #       TODAS  las condiciones de frontera son Dirichlet -------------------------
    matTemp[:,0] = lowTemp
    matTemp[0,:] = lowTemp
    matTemp[-1,:] = lowTemp
    matTemp[:,-1] = highTemp

    coordsX[:,0]  = 0.0
    coordsX[:,-1] = deltaX*(nNodesX-1)
    coordsY[-1,:] = 0.0
    coordsY[0,:]  = deltaY*(nNodesY-1)
   
    coordsX[0,:]  = [deltaX*i for i in range(nNodesX)]
    coordsX[-1,:] = [deltaX*i for i in range(nNodesX)]
    coordsY[:,0]  = [deltaY*(nNodesY-1-i) for i in range(nNodesY)]
    coordsY[:,-1] = [deltaY*(nNodesY-1-i) for i in range(nNodesY)]


    # --------------------------------------------------------------------------------
    # Construccion de diagonales con valores diferentes de 0 - Matriz Ixzquierda -----
    diag0  =  np.float64(1.0+gamma)*np.ones(nTotalN)
    diag1U =  np.float64(-1.0*alpha)*np.ones(nTotalN)
    diag1L =  np.float64(-1.0*alpha)*np.ones(nTotalN)
    diag2  =  np.float64(-1.0*beta)*np.ones(nTotalN)

    for i in range(nNodesY-2):
        diag1U[i*(nNodesX-2)] = np.float64(0.0)
        diag1L[i*(nNodesX-2)+(nNodesX-3)] = np.float64(0.0)
        #diag1L[(i+1)*(nNodesX-2)-1] = np.float64(0.0)

    # --------------------------------------------------------------------------------
    # Construccion de matrix sparse usando unicamente las diagonales definidas ↑ -----
    matCoeffA = sp.sparse.spdiags([diag2,diag1L,diag0,diag1U,diag2],
                                 [-(nNodesX-2),-1,0,1,(nNodesX-2)],nTotalN,nTotalN,"csc")

    # --------------------------------------------------------------------------------
    # Construccion de diagonales con valores diferentes de 0 - Matriz Derecha  -------
    diag0  =  np.float64(1.0-gamma)*np.ones(nTotalN)
    diag1U =  np.float64(1.0*alpha)*np.ones(nTotalN)
    diag1L =  np.float64(1.0*alpha)*np.ones(nTotalN)
    diag2  =  np.float64(1.0*beta)*np.ones(nTotalN)

    for i in range(nNodesY-2):
        diag1U[i*(nNodesX-2)] = np.float64(0.0)
        diag1L[i*(nNodesX-2)+(nNodesX-3)] = np.float64(0.0)
        #diag1L[(i+1)*(nNodesX-2)-1] = np.float64(0.0)

    # --------------------------------------------------------------------------------
    # Construccion de matrix sparse usando unicamente las diagonales definidas ↑ -----
    matCoeffB = sp.sparse.spdiags([diag2,diag1L,diag0,diag1U,diag2],
                                 [-(nNodesX-2),-1,0,1,(nNodesX-2)],nTotalN,nTotalN,"csc")

    # --------------------------------------------------------------------------------
    # Print the matrix of coefficients, if SMALL only !!! ----------------------------
    if nTotalN <= 100: print(matCoeffA.toarray())
    if nTotalN <= 100: print(matCoeffB.toarray())


    # --------------------------------------------------------------------------------
    # Construccion de los vectores de condiciones de frontera ------------------------
    vecBC01 = np.zeros(nTotalN,dtype=np.float64) # Vector asociado a alpha (ver presentacion)
    vecBC02 = np.zeros(nTotalN,dtype=np.float64) # Vector asociado a beta (ver presentacion)

    # Llenado de valores de condiciones de frontera izquierda (cold) y derecha (hot) -
    for i in range(nNodesY-2):
        vecBC01[i*(nNodesX-2)] = lowTemp
        vecBC01[i*(nNodesX-2)+(nNodesX-3)] = highTemp

    # Llenado de valores de condiciones de frontera superior e inferior (ambas cold) -
    for i in range(nNodesX-2):
        vecBC02[i] = lowTemp
        vecBC02[(nNodesX-2)*(nNodesY-2)-1-i] = lowTemp

    # --------------------------------------------------------------------------------
    # Construccion del vector del lado derecho (conteniendo BCs)
    vecRHS = np.float64(2.0*alpha)*vecBC01 + np.float64(2.0*beta)*vecBC02

    # --------------------------------------------------------------------------------
    # Estas siguientes lineas son las importante... es donde resolvemos en el tiempo -
    # la distribucion de temperaturas en la placa!!!  --------------------------------
    for t in range(nTsteps):
        vecEvolution = matCoeffB @ vecTempActual + vecRHS
        vecTempFutura = sp.sparse.linalg.spsolve(matCoeffA, vecEvolution, use_umfpack=False)
        
        if ( (t%printEvery)==0 ):
            plotNumber += 1
            # -------------------------------------------------------------------------
            # Llenado de matrices para representacion grafica/visual ------------------
            # OJO: Solo valores internos... no los de frontera       ------------------
            for j in range(nNodesY-2):
                for i in range(nNodesX-2):
                    matTemp[-(j+2),i+1] = vecTempFutura[i+j*(nNodesX-2)]

            # --------------------------------------------------------------------------------
            levels = np.linspace(lowTemp+10,highTemp-10, 7)
            fig, axis = plt.subplots(1, 1, figsize=(10, 8))
            h1 = axis.contourf(coordsX, coordsY, matTemp, 60, 
                               cmap='plasma',vmin=lowTemp,vmax=highTemp)#,levels=levels)
            h2 = axis.contour(h1,levels=levels,colors='w',linewidths=0.5)
            timeStamp=f"t = {np.float64(t)*deltaT:6.3f} s"
            plt.text(0.066*lenX,1.12*lenY, timeStamp, color='black', fontsize=14)
            plt.xlabel('$x$ [m]',fontsize=14)
            plt.ylabel('$y$ [m]',fontsize=14)
            plt.axis('scaled')
            cbar = fig.colorbar(h1,extend='max')
            cbar.ax.set_ylabel('$T$ [K]')
            cbar.add_lines(h2)
            fileName = "./temperaturasPlaca_"+f'{plotNumber:05d}'+".png"
            plt.savefig(fileName)
            #fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap='magma'),
            #             ax=ax, orientation='vertical', label='a colorbar label')
            #plt.clabel(h,h.levels,fontsize=10)
            #plt.colorbar(label='$T$ [K]')
            #plt.show()
            plt.close()
        #endif

        # Reset solution vector as new initial condition
        vecTempActual = vecTempFutura
    #endfor

    # Finalizada la evolucion temporal
    # -----------------------------------------------------------------
    print(f"Hemos acabado!")

# -----------------------------------------------------------------
# Main function call
if __name__ == "__main__":
    main()
# -----------------------------------------------------------------
