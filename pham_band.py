#!/usr/bin/env python

import os, sys, argparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection

# from scipy.sparse import block_diag

mpl.rcParams['axes.unicode_minus'] = False

import os, yaml
try:
    from yaml import CLoader as Loader
except:
    from yaml import Loader

from time import time
############################################################

def phonon_angular_momentum(freq, polar_vec, temp: float=300):
    '''
    Calculate the phonon angular momentum according to the following paper:
    
        "Angular momentum of phonons and the Einstein-de Haas effect",
        PRL, 112, 085503 (2014)

    '''
    # from ase.units import kB
    # from phonopy.units import THzToEv
    kB = 8.617330337217213e-05
    THzToEv = 0.00413566733

    # The Bose distribution, phonon energy given in THz
    if np.isclose(temp, 0.0):
        nbose = 0.5
    else:
        nbose = 0.5 + 1. / (np.exp(freq * THzToEv / (kB * temp)) - 1.0)
    ixyz  = [[1,2], [2, 0], [0, 1]]

    # m = np.array([[0, -1j], [1j, 0]])
    # M = block_diag([m for ii in range(natoms)])

    nqpts, nbnds, natoms, _ = polar_vec.shape

    # phonon angular momentum in hbar
    J0 = np.zeros((3, nqpts, nbnds))
    for ii in range(3):
        # e     = polar_vec[:,:,:,ixyz[ii]].reshape((-1, 2 * natoms))
        # J0[ii] = np.sum((e.conj() @ M) * e, axis=1).reshape((nqpts, # nbnds)).real

        e = polar_vec[...,ixyz[ii]]
        J0[ii] = 2.0 * np.sum(
            e[:,:,:,0].conj() * e[:,:,:,1],
            axis=2
        ).imag

    return J0 * nbose


def read_ph_yaml(filename):
    _, ext = os.path.splitext(filename)
    if ext == '.xz' or ext == '.lzma':
        try:
            import lzma
        except ImportError:
            raise("Reading a lzma compressed file is not supported "
                  "by this python version.")
        with lzma.open(filename) as f:
            data = yaml.load(f, Loader=Loader)
    elif ext == '.gz':
        import gzip
        with gzip.open(filename) as f:
            data = yaml.load(f, Loader=Loader)
    else:
        with open(filename, 'r') as f:
            data = yaml.load(f, Loader=Loader)

    freqs   = []
    dists   = []
    qpoints = []
    labels  = []
    eigvec  = []
    Acell   = np.array(data['lattice'])
    Bcell   = np.array(data['reciprocal_lattice'])

    for j, v in enumerate(data['phonon']):
        if 'label' in v:
            labels.append(v['label'])
        else:
            labels.append(None)
        freqs.append([f['frequency'] for f in v['band']])
        if 'eigenvector' in v['band'][0]:
            eigvec.append([np.array(f['eigenvector']) for f in v['band']])
        qpoints.append(v['q-position'])
        dists.append(v['distance'])

    if all(x is None for x in labels):
        if 'labels' in data:
            ss = np.array(data['labels'])
            labels = list(ss[0])
            for ii, f in enumerate(ss[:-1,1] == ss[1:,0]):
                if not f:
                    labels[-1] += r'|' + ss[ii+1, 0]
                labels.append(ss[ii+1, 1])
        else:
            labels = []

    return (Bcell,
            np.array(dists),
            np.array(freqs),
            np.array(qpoints),
            data['segment_nqpoint'],
            labels, eigvec)


def pam_plot(p):
    '''
    '''
    ############################################################
    # the phonon band
    Bcell, D1, F1, Q1, B1, L1, E1 = read_ph_yaml(p.yaml)

    assert E1, "PHONON EIGENVECTORs MUST NOT BE EMPTY!"
    E1 = np.asarray(E1)
    E1 = E1[...,0] + 1j*E1[...,1]
    nqpts, nbnds, natoms, _ = E1.shape

    Jxyz = phonon_angular_momentum(F1, E1, p.temperature)

    ############################################################
    if p.direction == 'a':
        nsubs = 3
    else:
        nsubs = 1

    if p.figsize is None:
        if nsubs == 1:
            p.figsize = (6.4, 4.8)
        else:
            p.figsize = (6.4, 9.0) if p.layout == 'v' else (12.0, 4.8)
      
    fig = plt.figure(
        figsize=p.figsize,
        dpi=300,
    )

    if p.layout == 'v':
        layout = np.arange(nsubs, dtype=int).reshape((-1, 1))
    else:
        layout = np.arange(nsubs, dtype=int).reshape((1, -1))

    axes   = fig.subplot_mosaic(
        layout,
        empty_sentinel=-1,
        gridspec_kw=dict(
            # height_ratios= [1.0],
            # width_ratios=[1, 1.0],
            # hspace=0.05,
            # wspace=0.06,
        )
    )
    axes = np.array([ax for ax in axes.values()])

    caxs = []
    for ii in range(nsubs):
        ax = axes[ii]
        divider = make_axes_locatable(ax)
        if p.layout == 'v':
            ax_cbar = divider.append_axes('right', size='2%', pad=0.04)
        else:
            ax_cbar = divider.append_axes('top', size='4%', pad=0.03)
        caxs.append(ax_cbar)

    pam_cmaps = [
        "PiYG", 'PuOr', "seismic"
    ]
    ############################################################
    for ii in range(nsubs):
        ax = axes[ii]
        if p.direction == 'a':
            which_j = ii
        else:
            which_j = 'xyz'.index(p.direction)

        ax.axhline(y=0, ls='-', lw=0.5, color='k', alpha=0.6)

        norm = mpl.colors.Normalize(vmin=Jxyz[which_j].min(), vmax=Jxyz[which_j].max())
        s_m = mpl.cm.ScalarMappable(cmap=pam_cmaps[which_j], norm=norm)
        s_m.set_array([Jxyz[which_j]])

        for jj in range(0, F1.shape[1]):
            ik = 0
            for nseg in B1:
                x = D1[ik:ik+nseg]
                y = F1[ik:ik+nseg, jj]
                z = Jxyz[which_j,ik:ik+nseg, jj]

        
                if p.plt_type == 'colormap':
                    ax.plot(
                        x, y,
                        lw=2.0, color='gray',
                    )

                    points = np.array([x, y]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(segments,
                                        colors=[s_m.to_rgba(ww)
                                                for ww in (z[1:] + z[:-1])/2.]
                                        )
                    lc.set_linewidth(2.0)
                    lc.set_alpha(0.8)
                    ax.add_collection(lc)
                else:
                    ax.plot(
                        x, y,
                        lw=0.5, color='gray',
                    )

                ik += nseg

        if p.plt_type == 'scatter':
            sca = ax.scatter(
                np.tile(D1, (nbnds, 1)).T,
                F1,
                s=np.abs(np.abs(Jxyz[which_j])*20),
                c=Jxyz[which_j],
                cmap=pam_cmaps[which_j],
                vmin=Jxyz[which_j].min(),
                vmax=Jxyz[which_j].max(),
            )


        for jj in np.cumsum(B1)[:-1]:
            ax.axvline(
                x=D1[jj], ls='--',
                color='gray', alpha=0.8, lw=0.5
            )

        ax.grid('on', ls='--', lw=0.5, color='gray', alpha=0.5)

        ax.set_xlim(D1.min(), D1.max())
        ax.set_xticks(D1[np.r_[[0], np.cumsum(B1)-1]])
        if L1:
            ax.set_xticklabels(L1)

        if p.layout == 'h' and ii <= 0:
            ax.set_ylabel('Frequency (THz)', labelpad=5)
        elif p.layout == 'v':
            ax.set_ylabel('Frequency (THz)', labelpad=5)
        else:
            pass

        cbar = plt.colorbar(
            s_m, cax=caxs[ii],
            extend='both', shrink=0.5,
            orientation='vertical' if p.layout == 'v' else 'horizontal',
        )
        cbar.ax.tick_params(which='both', labelsize='small')

        if p.layout == 'h':
            cbar.ax.xaxis.set_ticks_position('top') 
            cbar.ax.xaxis.set_label_position('top') 
            cbar.ax.set_xlabel(r'$J_{}\,/\,\hbar$'.format('xyz'[which_j]))
            

        if p.layout == 'v':
            cbar.ax.text(1.60, 1.02, r'$J_{}\,/\,\hbar$'.format('xyz'[which_j]),
                    ha="left",
                    va="bottom",
                    # fontsize='small',
                    # family='monospace',
                    fontweight='bold',
                    transform=cbar.ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, lw=0.5)
            )


    plt.tight_layout(pad=1.0)
    plt.savefig(p.figname)

############################################################

def parse_cml_args(cml):
    '''
    CML parser.
    '''
    arg = argparse.ArgumentParser(add_help=True)

    arg.add_argument('-i', dest='yaml', action='store', type=str,
                     default='band.yaml',
                     help='The yaml file containing phonon eigenvalues and eigenvectors.')

    arg.add_argument('-t', dest='temperature', action='store', type=float,
                     default=0,
                     help='The temperature, default to 0K.')

    arg.add_argument('-d', dest='direction', action='store', type=str,
                     default='a', choices=['x', 'y', 'z', 'a'],
                     help='One or all the components of the PAM.')

    arg.add_argument('-s', '--figsize', dest='figsize', nargs = 2, action='store', type=float,
                     default=None,
                     help='Figure size of the output image.')

    arg.add_argument('-o', dest='figname', action='store', type=str,
                     default='pam.png',
                     help='Output figure name.')

    arg.add_argument('--plt-type', dest='plt_type', action='store', type=str,
                     default='scatter', choices=['scatter', 'colormap'],
                     help='How to render the plot.')

    arg.add_argument('--layout', dest='layout', action='store', type=str,
                     default='v', choices=['h', 'v'],
                     help='Layout of the subfigures.')

    return arg.parse_args(cml)

def main(cml):
    p = parse_cml_args(cml)

    pam_plot(p)


if __name__ == "__main__":
    main(sys.argv[1:])
