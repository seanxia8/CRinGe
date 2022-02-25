import torch
import numpy as np
import string
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import LinearLocator
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
import matplotlib.patches as patches

from scipy.interpolate import griddata
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp2d
import scipy.stats as stats
from scipy import optimize
from scipy.optimize import curve_fit

from copy import copy
from cycler import cycler

def computeDwall_(vertex):
    x = vertex[0]
    y = vertex[1]
    z = vertex[2]

    Rmax = 1690.
    Zmax = 1810.
    rr   = (x*x + y*y)**0.5
    absz = np.abs(z)
    #check if vertex is outside tank
    
    signflg = 1.
    if absz>Zmax or rr > Rmax:
        signflg = -1.
    #find min distance to wall
    distz = np.abs(Zmax-absz)
    distr = np.abs(Rmax-rr)
    wall = signflg*np.minimum(distz,distr)
    return wall

def computeTowall_(vertex, direction):
    x = vertex[0]
    y = vertex[1]
    z = vertex[2]
    dx = direction[0]
    dy = direction[1]
    dz = direction[2]
    R=1690.
    l_b=100000.0
    l_t=100000.0
    H = 0.0
    if dx!=0 or dy!=0:
        A = dx*dx+dy*dy
        B = 2*(x*dx+y*dy)
        C = x*x+y*y-R*R
        RAD = (B*B) - (4*A*C)
        l_b = ((-1*B) + RAD**0.5)/(2*A)
    if dz==0:
        return l_b
    elif dz > 0:
        H=1810
    elif dz < 0:
        H=-1810
    l_t=(H - z)/dz;
    return np.minimum(l_t,l_b)

def _scan_lossvE(net):
    data = torch.as_tensor(net.data, dtype=torch.float, device=net.device)
    origE = data[0][9].item()*net.energy_scale
    orig_loss = net.evaluate(False)['loss']
    
    loss = []
    energy = []
    eSpace = np.linspace(0.2*origE, 1.8*origE, 100).tolist()
    for iE in eSpace:
        net.data[0][9] = float(iE/net.energy_scale)
        energy.append(iE)
        loss.append(net.evaluate(False)['loss'])

    return energy, loss, origE, orig_loss

def _scan_lossvPID(net, energy):
    data = torch.as_tensor(net.data, dtype=torch.float, device=net.device)
    net.data[0][9] = energy/net.energy_scale
    net.data[0][1] = 1
    net.data[0][2] = 0
    LossE =net.evaluate(False)['loss']
    net.data[0][1] = 0
    net.data[0][2] = 1
    LossMu =net.evaluate(False)['loss']

    LossPID = LossE - LossMu
    return LossPID, energy

def quadratic_spline_roots(spl):
    root = []
    knots = spl.get_knots()
    for a, b in zip(knots[:-1], knots[1:]):
        u, v, w = spl(a), spl((a+b)/2), spl(b)
        t = np.roots([u+w-2*v, w-u, 2*v])
        t = t[np.isreal(t) & (np.abs(t) <= 1)]
        root.extend(t*(b-a)/2 + (b+a)/2)
    return np.array(root)

def find_cubicspline_min(spl, root):
    cr_vals = spl(root)
    min_index = np.argmin(cr_vals)
    min_pt = root[min_index]
    return min_pt

def _stack_hit_event_display(net, flip_tb):
    label_top = net.charge_top.detach().cpu().numpy().reshape(48,48)*net.charge_scale
    label_bottom = net.charge_bottom.detach().cpu().numpy().reshape(48,48)*net.charge_scale
    label_barrel = net.charge_barrel.detach().cpu().numpy().reshape(51,150)*net.charge_scale
    
    data = torch.as_tensor(net.data, dtype=torch.float, device=net.device)
    pred_barrel, pred_bottom, pred_top = net(data)
    
    unhit_top = (1/(1+torch.exp(pred_top[:, 0]).detach().cpu().numpy())*net.top_mask).reshape(48,48)
    unhit_bottom = (1/(1+torch.exp(pred_bottom[:, 0]).detach().cpu().numpy())*net.bottom_mask).reshape(48,48)
    unhit_barrel = (1/(1+torch.exp(pred_barrel[:, 0]).detach().cpu().numpy())).reshape(51,150)

    if flip_tb:
        unhit_top, unhit_bottom = unhit_bottom, unhit_top

    label_barrel=np.flipud(label_barrel) # the 1d array starts from bottom?
    label_bottom=np.flipud(label_bottom) 
    
    unhit_barrel = np.flipud(unhit_barrel)
    unhit_bottom = np.flipud(unhit_bottom)
    
    dim_barrel = label_barrel.shape
    dim_cap = label_top.shape #(row, column)
    #make a new array including all 3 regions in a rectangular
    
    new_combined_event_disp = np.zeros((2*dim_cap[0]+dim_barrel[0], dim_barrel[1]))
    new_combined_hitprob = np.zeros((2*dim_cap[0]+dim_barrel[0], dim_barrel[1]))
    #put cap in the center
    cap_start = int(0.5*(dim_barrel[1]-dim_cap[1]))
    new_combined_event_disp[0:dim_cap[0],cap_start:(cap_start+dim_cap[1])] = np.log(label_top+1e-10)
    new_combined_event_disp[dim_cap[0]:(dim_cap[0]+dim_barrel[0]),0:dim_barrel[1]] = np.log(label_barrel+1e-10)
    new_combined_event_disp[(dim_cap[0]+dim_barrel[0]):new_combined_event_disp.shape[0], cap_start:(cap_start+dim_cap[1])] = np.log(label_bottom+1e-10)
    
    new_combined_hitprob[0:dim_cap[0],cap_start:(cap_start+dim_cap[1])] = unhit_top
    new_combined_hitprob[dim_cap[0]:(dim_cap[0]+dim_barrel[0]),0:dim_barrel[1]] = unhit_barrel
    new_combined_hitprob[(dim_cap[0]+dim_barrel[0]):new_combined_event_disp.shape[0], cap_start:(cap_start+dim_cap[1])] = unhit_bottom

    nhit = np.count_nonzero(new_combined_event_disp>0)
    
    return new_combined_event_disp, new_combined_hitprob, nhit

def _stack_event_display(charge, time, mask, tflag):
    label_top = np.where(mask[0], np.log(charge[2]+1.e-10), np.nan).reshape(48,48)
    label_bottom = np.where(mask[1], np.log(charge[1]+1.e-10), np.nan).reshape(48,48)
    label_barrel = np.log(charge[0]+1.e-10).reshape(51,150)

    if tflag:
        time_top = np.where(mask[0], time[2], np.nan).reshape(48,48)
        time_bottom = np.where(mask[1], time[1], np.nan).reshape(48,48)
        time_barrel = time[0].reshape(51,150)
        
    
    label_barrel=np.flipud(label_barrel) # the 1d array starts from bottom?    
    label_bottom=np.flipud(label_bottom) 

    dim_barrel = label_barrel.shape
    dim_cap = label_top.shape #(row, column)
    #make a new array including all 3 regions in a rectangular
    
    new_combined_event_disp = np.empty((2*dim_cap[0]+dim_barrel[0], dim_barrel[1]))
    new_combined_time_disp = np.empty((2*dim_cap[0]+dim_barrel[0], dim_barrel[1]))
    new_combined_event_disp[:] = np.nan
    new_combined_time_disp[:] = np.nan
    #put cap in the center
    cap_start = int(0.5*(dim_barrel[1]-dim_cap[1]))
    new_combined_event_disp[0:dim_cap[0],cap_start:(cap_start+dim_cap[1])] = label_top
    new_combined_event_disp[dim_cap[0]:(dim_cap[0]+dim_barrel[0]),0:dim_barrel[1]] = label_barrel
    new_combined_event_disp[(dim_cap[0]+dim_barrel[0]):new_combined_event_disp.shape[0], cap_start:(cap_start+dim_cap[1])] = label_bottom
    if tflag:
        new_combined_time_disp[0:dim_cap[0],cap_start:(cap_start+dim_cap[1])] = time_top
        new_combined_time_disp[dim_cap[0]:(dim_cap[0]+dim_barrel[0]),0:dim_barrel[1]] = time_barrel
        new_combined_time_disp[(dim_cap[0]+dim_barrel[0]):new_combined_event_disp.shape[0], cap_start:(cap_start+dim_cap[1])] = time_bottom

    
    if not tflag:    
        return new_combined_event_disp
    else:
        return new_combined_event_disp, new_combined_time_disp

def _save_scan_curve(flavor, plot_dict):
    
    loss_range = np.array(plot_dict['loss_scanlist']).max() - np.array(plot_dict['loss_scanlist']).min()    
    rect = copy(plot_dict['rect'])
    circ_top = copy(plot_dict['circ_top'])
    circ_bottom = copy(plot_dict['circ_bottom'])
    rect_cp = copy(rect)
    circt = copy(circ_top)
    circb = copy(circ_bottom)
    
    axscan = plot_dict['figure'].add_subplot(131)
    axevent = plot_dict['figure'].add_subplot(132)
    axhit = plot_dict['figure'].add_subplot(133)

    disp = axevent.imshow(plot_dict['label_stack'], vmin=0, vmax=np.max(plot_dict['label_stack'])+1)
    axevent.set_axis_off()
    axevent.add_patch(rect)
    axevent.add_patch(circ_top)
    axevent.add_patch(circ_bottom)
    plt.colorbar(disp, ax=axevent)

    hit = axhit.imshow(plot_dict['pred_stack'], vmin=0, vmax=1)
    axhit.set_axis_off()
    axhit.add_patch(rect_cp)
    axhit.add_patch(circt)
    axhit.add_patch(circb)
    plt.colorbar(hit, ax=axhit)

    axscan.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    axscan.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

    axscan.plot(plot_dict['energy_scanlist'], plot_dict['loss_scanlist'], color="blue", alpha=0.75)
    axscan.scatter(plot_dict['orig_E'], plot_dict['orig_Loss'], color="red", label="Truth")
    axscan.scatter(plot_dict['crptsELoss'], plot_dict['splELoss_mu'](plot_dict['crptsELoss']), color = "orange", s=10, label="Local min/max")
    axscan.scatter(plot_dict['minLoss'], plot_dict['splELoss'](plot_dict['minLoss']), color="violet", label="Reco", marker="^", s=30)
    axscan.text(0.1, 0.95, "%s: E=%.2f MeV \nWall=%.2f cm | Towall=%.2f cm" % (flavor, plot_dict['orig_E'], plot_dict['wall'], plot_dict['towall']), verticalalignment = 'top', horizontalalignment='left', transform=axscan.transAxes, color='black', fontsize=7, bbox={'facecolor': 'white', 'alpha': 1., 'pad': 10})
    axscan.set_ylim([np.array(plot_dict['loss_scanlist']).min()-0.05*loss_range, np.array(plot_dict['loss_scanlist']).max()+0.05*loss_range])
    axscan.set_ylabel("Loss")
    axscan.set_xlabel("Energy [MeV]")
    axscan.legend(loc='upper right', framealpha=0, prop={'size':8})

    plot_dict['pdfout'].savefig(plot_dict['figure'])

    axevent.cla()
    axhit.cla()
    axscan.cla()
    

def _plot_npeak_comparison(outdir, npeak_total, info_dict, n_scan, tflag, cflag) :
    #info_dict = {npeak{flavor{keys}}}
    fig, (axmu, axe) = plt.subplots(2,2)
    fig.set_size_inches(8,7)
    #for e resolution histogram
    figE, axE = plt.subplots(1,2)    
    figE.set_size_inches(7,3)
    axE[0].set_prop_cycle(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    axE[1].set_prop_cycle(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    axE[0].set_xlabel(r"$\Delta_{E}$", fontsize=10, loc='right')
    axE[1].set_xlabel(r"$\Delta_{E}$", fontsize=10, loc='right')
    axE[0].set_title(r'$\mu^-$ Energy Residual', fontsize=10, y=1)
    axE[1].set_title(r'$e^-$ Energy Residual', fontsize=10, y=1)
    #for PID histogram
    figPID, axPID = plt.subplots(1,2)    
    figPID.set_size_inches(7,3)
    for i in range(len(axPID)):
        axPID[i].set_xlabel(r"$e/\mu$ PID", fontsize=10, loc='center')
        axPID[i].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    
    Ebins = np.linspace(-0.8, 0.8, 320)
    E_centers = (Ebins[:-1] + Ebins[1:])/2
    PIDbins = np.linspace(-5000, 5000, 121)
    
    wallcut = [0, 200, 500, 1700]
    colorid = ['red','green','blue']
    markers = ['o', '^', 's']

    flavor = ['muon', 'electron']
    flavorstyle = ['-', '--']
    residual = [[],[]]
    pid = [[],[]]
    wall = [[],[]]
    towall = [[],[]]
    onbound = [[],[]]
    not_min = [[],[]]

    ngaus = [ i+1 for i in range(npeak_total) ]
    
    for ig in range(npeak_total):
        for j,f in enumerate(flavor):
            residual[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['energy_res'][0:n_scan])
            pid[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['pid'][0:n_scan])
            wall[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['dwall'][0:n_scan])
            towall[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['towall'][0:n_scan]) 
            onbound[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['onbound'][0:n_scan])
            not_min[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['not_local_min'][0:n_scan])

            if ig == 0 or ig == 2 or ig == 4 or ig == 9:
                axE[j].hist(residual[j][ig], Ebins, histtype='step', density=False, linewidth = 0.75, label='{:d}'.format(ig+1))

    axE[0].legend(loc='best', framealpha=1, facecolor='w', edgecolor='inherit', prop={'size': 11})
    axE[1].legend(loc='best', framealpha=1, facecolor='w', edgecolor='inherit', prop={'size': 11})
    axE[0].axvline(x=0, color='grey', linewidth = 0.5, linestyle='--')
    axE[1].axvline(x=0, color='grey', linewidth = 0.5, linestyle='--')    
    axE[0].set_xlim(-0.5, 0.5)
    axE[1].set_xlim(-0.5, 0.5)
    axE[0].set_ylim(0, 800)
    axE[1].set_ylim(0, 800)    

    figE.tight_layout()
    pp = PdfPages(outdir+'/SK_MultiGaus_Eres_NGaus_1_to_'+str(npeak_total)+'_time_'+str(tflag)+'_corr_'+str(cflag)+'_'+str(n_scan)+'_events.pdf')
    pp.savefig(figE)
    pp.close()


    mask = np.array((np.array(onbound)==0)&(np.array(not_min)==0), dtype=bool)
    #if ig == 0 or ig == 4 or ig == 9:
    axPID[0].hist(np.clip(np.array(pid[0][0])[mask[0][0]], PIDbins[0], PIDbins[-1]), PIDbins, histtype='stepfilled', density=False, linewidth = 0.75, linestyle = flavorstyle[0], alpha = 0.75, color = '#1f77b4', label=r'$\mu^-$ 1 Gaussian')
    axPID[0].hist(np.clip(np.array(pid[1][0])[mask[1][0]], PIDbins[0], PIDbins[-1]), PIDbins, histtype='stepfilled', density=False, linewidth = 0.75, linestyle = flavorstyle[1], alpha = 0.75, color = '#ff7f0e', label=r'$e^-$ 1 Gaussian')
    axPID[1].hist(np.clip(np.array(pid[0][npeak_total-1])[mask[0][npeak_total-1]], PIDbins[0], PIDbins[-1]), PIDbins, histtype='stepfilled', density=False, linewidth = 0.75, linestyle = flavorstyle[0], alpha = 0.75, color = '#1f77b4', label=r'$\mu^-$ {:d} Gaussian'.format(npeak_total))
    axPID[1].hist(np.clip(np.array(pid[1][npeak_total-1])[mask[1][npeak_total-1]], PIDbins[0], PIDbins[-1]), PIDbins, histtype='stepfilled', density=False, linewidth = 0.75, linestyle = flavorstyle[1], alpha = 0.75, color = '#ff7f0e', label=r'$e^-$ {:d} Gaussian'.format(npeak_total))

    #handles = [patches.Rectangle((0,0),1,1,color='black',ec='black',fc='w', lw=0.5, linestyle= c ) for c in flavorstyle]
    #legend_flavor = axPID.legend(handles, flavor, loc='upper right', framealpha = 1, facecolor = 'w', edgecolor='inherit', prop={'size':5}, bbox_to_anchor=(0.9, 0.7, 0.1, 0.1))
    axPID[0].legend(loc='upper left', framealpha=1, facecolor='w', edgecolor='inherit', prop={'size': 10})
    #axPID.add_artist(legend_flavor)
    axPID[0].axvline(x=0, color='grey', linewidth = 0.5, linestyle=':')
    axPID[0].set_xlim(-4000, 4000)
    axPID[1].legend(loc='upper left', framealpha=1, facecolor='w', edgecolor='inherit', prop={'size': 10})    
    axPID[0].set_ylim(0, 400)
    axPID[1].axvline(x=0, color='grey', linewidth = 0.5, linestyle=':')
    axPID[1].set_xlim(-4000, 4000)
    axPID[1].set_ylim(0, 400)    

    
    figPID.tight_layout()
    pp = PdfPages(outdir+'/SK_MultiGaus_PID_NGaus_1_to_'+str(npeak_total)+'_time_'+str(tflag)+'_corr_'+str(cflag)+'_'+str(n_scan)+'_events.pdf')
    pp.savefig(figPID)
    pp.close()
    
    #mean and stdev in each tank region for each flavor
    mean_wall = [[[],[]] for i in range(3)]
    stdev_wall = [[[],[]] for i in range(3)]
    mean_towall = [[[],[]] for i in range(3)]
    stdev_towall = [[[],[]] for i in range(3)]
    
    for ip in range(1,4):
        muwall_cut = ((np.array(wall)[0] < wallcut[ip]) & (np.array(wall)[0] >= wallcut[ip-1]))
        mutowall_cut = ((np.array(towall)[0] < wallcut[ip]) & (np.array(towall)[0] >= wallcut[ip-1]))
        ewall_cut = ((np.array(wall)[1] < wallcut[ip]) & (np.array(wall)[1] >= wallcut[ip-1]))
        etowall_cut = ((np.array(towall)[1] < wallcut[ip]) & (np.array(towall)[1] >= wallcut[ip-1]))
        
        mu_wallmasked = np.where( muwall_cut & mask[0], np.array(residual)[0], np.nan )
        mu_towallmasked = np.where( mutowall_cut & mask[0], np.array(residual)[0], np.nan )
        e_wallmasked = np.where( muwall_cut & mask[1], np.array(residual)[1], np.nan )
        e_towallmasked = np.where( mutowall_cut & mask[1], np.array(residual)[1], np.nan )        
        
        dim = mu_wallmasked.shape
        if dim[0] != npeak_total:
            print('Masked event array dimension not correct')
            sys.exit()

        for ig in range(npeak_total):
            axE[0].cla()
            axE[1].cla()            
            subdir = outdir+"/SK_MultiGaus_"+str(ig+1)+"_time_"+str(tflag)+"_corr_"+str(cflag)
            try :
                os.makedirs(subdir)
            except FileExistsError :
                pass
            p0 = [1000, 0, 0.05]
            p1 = [200, 0, 0.1]
            hist_muw, _ = np.histogram(mu_wallmasked[ig][~np.isnan(mu_wallmasked[ig])], bins=Ebins)
            hist_mut, _ = np.histogram(mu_towallmasked[ig][~np.isnan(mu_towallmasked[ig])], bins=Ebins)
            hist_ew, _ = np.histogram(e_wallmasked[ig][~np.isnan(e_wallmasked[ig])], bins=Ebins)
            hist_et, _ = np.histogram(e_towallmasked[ig][~np.isnan(e_towallmasked[ig])], bins=Ebins)
            
            coeff_muw, _ = curve_fit(_single_gaussian, E_centers, hist_muw, p0 = p0, bounds=[[0, -0.8, 0], [n_scan, 0.8, 1/6]], maxfev=2000)
            coeff_mut, _ = curve_fit(_single_gaussian, E_centers, hist_mut, p0 = p1, bounds=[[0, -0.8, 0], [n_scan, 0.8, 1/6]], maxfev=2000)
            coeff_ew, _ = curve_fit(_single_gaussian, E_centers, hist_ew, p0 = p0, bounds=[[0, -0.8, 0], [n_scan, 0.8, 1/6]], maxfev=2000)
            coeff_et, _ = curve_fit(_single_gaussian, E_centers, hist_et, p0 = p1, bounds=[[0, -0.8, 0], [n_scan, 0.8, 1/6]], maxfev=2000)

            if coeff_muw[1] < -0.79 or coeff_muw[1] > 0.79:
                coeff_muw[1] = np.nanmean(mu_wallmasked[ig])
            if coeff_mut[1] < -0.79 or coeff_mut[1] > 0.79:
                coeff_mut[1] = np.nanmean(mu_towallmasked[ig])
            if coeff_ew[1] < -0.79 or coeff_ew[1] > 0.79:
                coeff_ew[1] = np.nanmean(e_wallmasked[ig])
            if coeff_et[1] < -0.79 or coeff_et[1] > 0.79:
                coeff_et[1] = np.nanmean(e_towallmasked[ig])
                
            
            hist_fit_muw = _single_gaussian(E_centers, *coeff_muw)
            hist_fit_mut = _single_gaussian(E_centers, *coeff_mut)
            hist_fit_ew = _single_gaussian(E_centers, *coeff_ew)
            hist_fit_et = _single_gaussian(E_centers, *coeff_et)            

            mean_wall[ip-1][0].append(coeff_muw[1])
            stdev_wall[ip-1][0].append(coeff_muw[2])
            mean_towall[ip-1][0].append(coeff_mut[1])
            stdev_towall[ip-1][0].append(coeff_mut[2])
            mean_wall[ip-1][1].append(coeff_ew[1])
            stdev_wall[ip-1][1].append(coeff_ew[2])
            mean_towall[ip-1][1].append(coeff_et[1])
            stdev_towall[ip-1][1].append(coeff_et[2])
            axE[0].hist(mu_wallmasked[ig][~np.isnan(mu_wallmasked[ig])], Ebins, histtype='step', density=False, linewidth = 0.75, label='{:.1f} $\leq$ Wall < {:.1f} cm'.format(wallcut[ip-1], wallcut[ip]))
            axE[0].hist(mu_towallmasked[ig][~np.isnan(mu_towallmasked[ig])], Ebins, histtype='step', density=False, linewidth = 0.75, label='{:.1f} $\leq$ Towall < {:.1f} cm'.format(wallcut[ip-1], wallcut[ip]))
            axE[0].plot(E_centers, hist_fit_muw, linewidth = 0.75, linestyle='--', label='Fit {:.1f} $\leq$ Wall < {:.1f} cm'.format(wallcut[ip-1], wallcut[ip]))
            axE[0].plot(E_centers, hist_fit_mut, linewidth = 0.75, linestyle='--', label='Fit {:.1f} $\leq$ Towall < {:.1f} cm'.format(wallcut[ip-1], wallcut[ip]))
            
            axE[1].hist(e_wallmasked[ig][~np.isnan(e_wallmasked[ig])], Ebins, histtype='step', density=False, linewidth = 0.75, label='{:.1f} $\leq$ Wall < {:.1f} cm'.format(wallcut[ip-1], wallcut[ip]))
            axE[1].hist(e_towallmasked[ig][~np.isnan(e_towallmasked[ig])], Ebins, histtype='step', density=False, linewidth = 0.75, label='{:.1f} $\leq$ Towall < {:.1f} cm'.format(wallcut[ip-1], wallcut[ip]))
            axE[1].plot(E_centers, hist_fit_ew, linewidth = 0.75, linestyle='--', label='Fit {:.1f} $\leq$ Wall < {:.1f} cm'.format(wallcut[ip-1], wallcut[ip]))
            axE[1].plot(E_centers, hist_fit_et, linewidth = 0.75, linestyle='--', label='Fit {:.1f} $\leq$ Towall < {:.1f} cm'.format(wallcut[ip-1], wallcut[ip]))
            
            axE[0].legend(loc='upper right', prop={'size':5})
            axE[1].legend(loc='upper right', prop={'size':5})
            axE[0].set_xlabel(r"$\Delta_{E}$", fontsize=10, loc='right')
            axE[1].set_xlabel(r"$\Delta_{E}$", fontsize=10, loc='right')
            axE[0].set_title(r'$\mu^-$ Energy Residual', fontsize=10, y=1)
            axE[1].set_title(r'$e^-$ Energy Residual', fontsize=10, y=1)            
            pp = PdfPages(subdir+'/SK_MultiGaus_Eresidual_wallcut_'+str(ip)+'_time_'+str(tflag)+'_corr_'+str(cflag)+'_'+str(n_scan)+'_events.pdf')
            pp.savefig(figE)
            pp.close()

    figE.clf()
            
        
    #############E resolution###############
    for ip in range(1,4):
        axmu[0].errorbar([ng+0.1*(ip-2) for ng in ngaus], np.nanmean(np.array(np.where((np.array(wall)[0] < wallcut[ip]) & (np.array(wall)[0] >= wallcut[ip-1]) & (mask[0]), np.array(residual)[0], np.nan)), axis = 1), yerr = 0.5*np.nanstd(np.array(np.where((np.array(wall)[0] < wallcut[ip]) & (np.array(wall)[0] >= wallcut[ip-1]) & (mask[0]), np.array(residual)[0], np.nan)), axis = 1), marker=markers[ip-1], color=colorid[ip-1], markersize = 4, linestyle = ':', linewidth = 1, capsize=1, elinewidth=1, markeredgewidth=1.5,label="{:.1f} $\leq$ Dwall < {:.1f} cm".format(wallcut[ip-1], wallcut[ip]))
        axmu[1].errorbar([ng+0.1*(ip-2) for ng in ngaus], np.nanmean(np.array(np.where((np.array(towall)[0] < wallcut[ip]) & (np.array(towall)[0] >= wallcut[ip-1]) & (mask[0]), np.array(residual)[0], np.nan)), axis = 1), yerr = 0.5*np.nanstd(np.array(np.where((np.array(towall)[0] < wallcut[ip]) & (np.array(towall)[0] >= wallcut[ip-1]) & (mask[0]), np.array(residual)[0], np.nan)), axis = 1), marker=markers[ip-1], color=colorid[ip-1], markersize = 4, linestyle = ':', linewidth = 1, capsize=1, elinewidth=1, markeredgewidth=1.5,label="{:.1f} $\leq$ Towall < {:.1f} cm".format(wallcut[ip-1], wallcut[ip]))
        axe[0].errorbar([ng+0.1*(ip-2) for ng in ngaus], np.nanmean(np.array(np.where((np.array(wall)[1] < wallcut[ip]) & (np.array(wall)[1] >= wallcut[ip-1]) & (mask[1]), np.array(residual)[1], np.nan)), axis = 1), yerr = 0.5*np.nanstd(np.array(np.where((np.array(wall)[1] < wallcut[ip]) & (np.array(wall)[1] >= wallcut[ip-1]) & (mask[1]), np.array(residual)[1], np.nan)), axis = 1), marker=markers[ip-1], color=colorid[ip-1], markersize = 4, linestyle = ':', linewidth = 1, capsize=1, elinewidth=1, markeredgewidth=1.5,label="{:.1f} $\leq$ Dwall < {:.1f} cm".format(wallcut[ip-1], wallcut[ip]))
        axe[1].errorbar([ng+0.1*(ip-2) for ng in ngaus], np.nanmean(np.array(np.where((np.array(towall)[1] < wallcut[ip]) & (np.array(towall)[1] >= wallcut[ip-1]) & (mask[1]), np.array(residual)[1], np.nan)), axis = 1), yerr = 0.5*np.nanstd(np.array(np.where((np.array(towall)[1] < wallcut[ip]) & (np.array(towall)[1] >= wallcut[ip-1]) & (mask[1]), np.array(residual)[1], np.nan)), axis = 1), marker=markers[ip-1], color=colorid[ip-1], markersize = 4, linestyle = ':', linewidth = 1, capsize=1, elinewidth=1, markeredgewidth=1.5,label="{:.1f} $\leq$ Towall < {:.1f} cm".format(wallcut[ip-1], wallcut[ip]))        
        '''
        axmu[0].errorbar(ngaus, mean_wall[ip-1][0], yerr = stdev_wall[ip-1][0], marker=markers[ip-1], color=colorid[ip-1], markersize = 4, linestyle = ':', linewidth = 1, capsize=1, elinewidth=1, markeredgewidth=1.5,label="{:.1f} $\leq$ Dwall < {:.1f} cm".format(wallcut[ip-1], wallcut[ip]))
        axmu[1].errorbar(ngaus, mean_towall[ip-1][0], yerr = stdev_towall[ip-1][0], marker=markers[ip-1], color=colorid[ip-1], markersize = 4, linestyle = ':', linewidth = 1, capsize=1, elinewidth=1, markeredgewidth=1.5,label="{:.1f} $\leq$ Towall < {:.1f} cm".format(wallcut[ip-1], wallcut[ip]))
        axe[0].errorbar(ngaus, mean_wall[ip-1][1], yerr = stdev_wall[ip-1][1], marker=markers[ip-1], color=colorid[ip-1], markersize = 4, linestyle = ':', linewidth = 1, capsize=1, elinewidth=1, markeredgewidth=1.5,label="{:.1f} $\leq$ Dwall < {:.1f} cm".format(wallcut[ip-1], wallcut[ip]))
        axe[1].errorbar(ngaus, mean_towall[ip-1][1], yerr = stdev_towall[ip-1][1], marker=markers[ip-1], color=colorid[ip-1], markersize = 4, linestyle = ':', linewidth = 1, capsize=1, elinewidth=1, markeredgewidth=1.5,label="{:.1f} $\leq$ Towall < {:.1f} cm".format(wallcut[ip-1], wallcut[ip]))        
        '''

    for iax in range(2):
        axmu[iax].set_ylim(-0.8, 0.6)
        axmu[iax].set_xlim(0, npeak_total+1)
        axmu[iax].set_xlabel(r"$N_{Gaussian}$", fontsize=10, loc='right')
        axmu[iax].set_ylabel(r"$\Delta_{E}$", fontsize=10, loc='top')
        axmu[iax].tick_params(axis='x', labelsize=10)
        axmu[iax].tick_params(axis='y', labelsize=10)
        axmu[iax].axhline(y=0., color='grey', linewidth = 1, linestyle=':', alpha=0.5)
        axe[iax].set_ylim(-0.8, 0.6)
        axe[iax].set_xlim(0, npeak_total+1)
        axe[iax].set_xlabel(r"$N_{Gaussian}$", fontsize=8, loc='right')
        axe[iax].set_ylabel(r"$\Delta_{E}$", fontsize=8, loc='top')
        axe[iax].tick_params(axis='x', labelsize=10)
        axe[iax].tick_params(axis='y', labelsize=10)        
        axe[iax].axhline(y=0., color='grey', linewidth = 1, linestyle=':', alpha=0.5)
        
        axmu[iax].legend(loc='upper left', prop={'size': 8})
        axe[iax].legend(loc='upper left', prop={'size': 8})

    axmu[0].set_title(r'$\mu^-$ divided by Dwall', fontsize=10, y=1)
    axmu[1].set_title(r'$\mu^-$ divided by Towall', fontsize=10, y=1)
    axe[0].set_title(r'$e^-$ divided by Dwall', fontsize=10, y=1)
    axe[1].set_title(r'$e^-$ divided by Towall', fontsize=10, y=1)
    
    fig.tight_layout(pad=0.8)
    pp = PdfPages(outdir+'/SK_MultiGaus_Ereso_vs_Walls_NGaus_1_to_'+str(npeak_total)+'_time_'+str(tflag)+'_corr_'+str(cflag)+'_'+str(n_scan)+'_events.pdf')
    pp.savefig(fig)
    pp.close()

    ################PID####################
    for iax in range(2):
        axmu[iax].cla()
        axe[iax].cla()
        axmu[iax].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axe[iax].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    for ip in range(1,4):
        #offset by 1.e-6 so the 0 points will appear in the log plot
        '''
        axmu[0].plot(ngaus, np.sum(np.array(np.where((np.array(wall)[0] < wallcut[ip]) & (np.array(wall)[0] >= wallcut[ip-1]) & (mask[0]) & (np.array(pid)[0]<=0), 1, 0)), axis = 1)/(1e-10+np.sum(np.array(np.where((np.array(wall)[0] < wallcut[ip]) & (np.array(wall)[0] >= wallcut[ip-1]) & (mask[0]), 1, 0)), axis = 1))+1.e-6,  marker=markers[ip-1], color=colorid[ip-1], markersize = 4, linestyle = ':', linewidth = 1, label="{:.1f} $\leq$ Dwall < {:.1f} cm".format(wallcut[ip-1], wallcut[ip]))
        axmu[1].plot(ngaus, np.sum(np.array(np.where((np.array(towall)[0] < wallcut[ip]) & (np.array(towall)[0] >= wallcut[ip-1]) & (mask[0]) & (np.array(pid)[0]<=0), 1, 0)), axis = 1)/(1e-10+np.sum(np.array(np.where((np.array(towall)[0] < wallcut[ip]) & (np.array(towall)[0] >= wallcut[ip-1]) & (mask[0]), 1, 0)), axis = 1))+1.e-6,  marker=markers[ip-1], color=colorid[ip-1], markersize = 4, linestyle = ':', linewidth = 1, label="{:.1f} $\leq$ Towall < {:.1f} cm".format(wallcut[ip-1], wallcut[ip]))
        axe[0].plot(ngaus, np.sum(np.array(np.where((np.array(wall)[1] < wallcut[ip]) & (np.array(wall)[1] >= wallcut[ip-1]) & (mask[1]) & (np.array(pid)[1]>=0), 1, 0)), axis = 1)/(1e-10+np.sum(np.array(np.where((np.array(wall)[1] < wallcut[ip]) & (np.array(wall)[1] >= wallcut[ip-1]) & (mask[1]), 1, 0)), axis = 1))+1.e-6,  marker=markers[ip-1], color=colorid[ip-1], markersize = 4, linestyle = ':', linewidth = 1, label="{:.1f} $\leq$ Dwall < {:.1f} cm".format(wallcut[ip-1], wallcut[ip]))
        axe[1].plot(ngaus, np.sum(np.array(np.where((np.array(towall)[1] < wallcut[ip]) & (np.array(towall)[1] >= wallcut[ip-1]) & (mask[1]) & (np.array(pid)[1]>=0), 1, 0)), axis = 1)/(1e-10+np.sum(np.array(np.where((np.array(towall)[1] < wallcut[ip]) & (np.array(towall)[1] >= wallcut[ip-1]) & (mask[1]), 1, 0)), axis = 1))+1.e-6,  marker=markers[ip-1], color=colorid[ip-1], markersize = 4, linestyle = ':', linewidth = 1, label="{:.1f} $\leq$ Towall < {:.1f} cm".format(wallcut[ip-1], wallcut[ip]))
        '''
        axmu[0].errorbar([ng+0.1*(ip-2) for ng in ngaus], np.sum(np.array(np.where((np.array(wall)[0] < wallcut[ip]) & (np.array(wall)[0] >= wallcut[ip-1]) & (mask[0]) & (np.array(pid)[0]<=0), 1, 0)), axis = 1)/(1e-10+np.sum(np.array(np.where((np.array(wall)[0] < wallcut[ip]) & (np.array(wall)[0] >= wallcut[ip-1]) & (mask[0]), 1, 0)), axis = 1))+1.e-6,
                         yerr = 0.5*np.sqrt(np.sum(np.array(np.where((np.array(wall)[0] < wallcut[ip]) & (np.array(wall)[0] >= wallcut[ip-1]) & (mask[0]) & (np.array(pid)[0]<=0), 1, 0)), axis = 1))/(1e-10+np.sum(np.array(np.where((np.array(wall)[0] < wallcut[ip]) & (np.array(wall)[0] >= wallcut[ip-1]) & (mask[0]), 1, 0)), axis = 1)), marker=markers[ip-1], color=colorid[ip-1], markersize = 4, linestyle = ':', linewidth = 1,  capsize=1, elinewidth=1, markeredgewidth=1.5, label="{:.1f} $\leq$ Dwall < {:.1f} cm".format(wallcut[ip-1], wallcut[ip]))
        axmu[1].errorbar([ng+0.1*(ip-2) for ng in ngaus], np.sum(np.array(np.where((np.array(towall)[0] < wallcut[ip]) & (np.array(towall)[0] >= wallcut[ip-1]) & (mask[0]) & (np.array(pid)[0]<=0), 1, 0)), axis = 1)/(1e-10+np.sum(np.array(np.where((np.array(towall)[0] < wallcut[ip]) & (np.array(towall)[0] >= wallcut[ip-1]) & (mask[0]), 1, 0)), axis = 1))+1.e-6,
                         yerr=0.5*np.sqrt(np.sum(np.array(np.where((np.array(towall)[0] < wallcut[ip]) & (np.array(towall)[0] >= wallcut[ip-1]) & (mask[0]) & (np.array(pid)[0]<=0), 1, 0)), axis = 1))/(1e-10+np.sum(np.array(np.where((np.array(towall)[0] < wallcut[ip]) & (np.array(towall)[0] >= wallcut[ip-1]) & (mask[0]), 1, 0)), axis = 1)), marker=markers[ip-1], color=colorid[ip-1], markersize = 4, linestyle = ':', linewidth = 1,  capsize=1, elinewidth=1, markeredgewidth=1.5, label="{:.1f} $\leq$ Towall < {:.1f} cm".format(wallcut[ip-1], wallcut[ip]))
        axe[0].errorbar([ng+0.1*(ip-2) for ng in ngaus], np.sum(np.array(np.where((np.array(wall)[1] < wallcut[ip]) & (np.array(wall)[1] >= wallcut[ip-1]) & (mask[1]) & (np.array(pid)[1]>=0), 1, 0)), axis = 1)/(1e-10+np.sum(np.array(np.where((np.array(wall)[1] < wallcut[ip]) & (np.array(wall)[1] >= wallcut[ip-1]) & (mask[1]), 1, 0)), axis = 1))+1.e-6,
                        yerr = 0.5*np.sqrt(np.sum(np.array(np.where((np.array(wall)[1] < wallcut[ip]) & (np.array(wall)[1] >= wallcut[ip-1]) & (mask[1]) & (np.array(pid)[1]>=0), 1, 0)), axis = 1))/(1e-10+np.sum(np.array(np.where((np.array(wall)[1] < wallcut[ip]) & (np.array(wall)[1] >= wallcut[ip-1]) & (mask[1]), 1, 0)), axis = 1)), marker=markers[ip-1], color=colorid[ip-1], markersize = 4, linestyle = ':', linewidth = 1,  capsize=1, elinewidth=1, markeredgewidth=1.5, label="{:.1f} $\leq$ Dwall < {:.1f} cm".format(wallcut[ip-1], wallcut[ip]))
        axe[1].errorbar([ng+0.1*(ip-2) for ng in ngaus], np.sum(np.array(np.where((np.array(towall)[1] < wallcut[ip]) & (np.array(towall)[1] >= wallcut[ip-1]) & (mask[1]) & (np.array(pid)[1]>=0), 1, 0)), axis = 1)/(1e-10+np.sum(np.array(np.where((np.array(towall)[1] < wallcut[ip]) & (np.array(towall)[1] >= wallcut[ip-1]) & (mask[1]), 1, 0)), axis = 1))+1.e-6,
                        yerr = 0.5*np.sqrt(np.sum(np.array(np.where((np.array(towall)[1] < wallcut[ip]) & (np.array(towall)[1] >= wallcut[ip-1]) & (mask[1]) & (np.array(pid)[1]>=0), 1, 0)), axis = 1))/(1e-10+np.sum(np.array(np.where((np.array(towall)[1] < wallcut[ip]) & (np.array(towall)[1] >= wallcut[ip-1]) & (mask[1]), 1, 0)), axis = 1)), marker=markers[ip-1], color=colorid[ip-1], markersize = 4, linestyle = ':',  capsize=1, elinewidth=1, markeredgewidth=1.5, linewidth = 1, label="{:.1f} $\leq$ Towall < {:.1f} cm".format(wallcut[ip-1], wallcut[ip]))

    for iax in range(2):
        axmu[iax].set_ylim(1.e-6, 10)
        axmu[iax].set_yscale('log')
        axmu[iax].set_xlim(0, npeak_total+1)        
        axmu[iax].set_xlabel(r"$N_{Gaussian}$", fontsize=8, loc='right')
        axmu[iax].set_ylabel("Mis-PID Rate", fontsize=8, loc='top')
        axmu[iax].tick_params(axis='x', labelsize=10)
        axmu[iax].tick_params(axis='y', labelsize=10)
        axmu[iax].axhline(y=0., color='grey', linewidth = 1, linestyle=':', alpha=0.5)
        axe[iax].set_ylim(1.e-6, 10)
        axe[iax].set_yscale('log')
        axe[iax].set_xlim(0, npeak_total+1)
        axe[iax].set_xlabel(r"$N_{Gaussian}$", fontsize=8, loc='right')
        axe[iax].set_ylabel("Mis-PID Rate", fontsize=8, loc='top')
        axe[iax].tick_params(axis='x', labelsize=10)
        axe[iax].tick_params(axis='y', labelsize=10)
        axe[iax].axhline(y=0., color='grey', linewidth = 1, linestyle=':', alpha=0.5)
        
        axmu[iax].legend(loc='upper left', prop={'size': 8})
        axe[iax].legend(loc='upper left', prop={'size': 8})

    axmu[0].set_title(r'$\mu^-$ divided by Dwall', y=1, fontsize=10)
    axmu[1].set_title(r'$\mu^-$ divided by Towall', y=1, fontsize=10)
    axe[0].set_title(r'$e^-$ divided by Dwall', y=1, fontsize=10)
    axe[1].set_title(r'$e^-$ divided by Towall', y=1, fontsize=10)

    #fig.tight_layout(pad=0.8)
    pp = PdfPages(outdir+'/SK_MultiGaus_misPID_vs_Walls_NGaus_1_to_'+str(npeak_total)+'_time_'+str(tflag)+'_corr_'+str(cflag)+'_'+str(n_scan)+'_events.pdf')
    pp.savefig(fig)
    pp.close()
    fig.clf()

    #print useful info
    print('$N_{Gaussian}$ & \multicolumn{2}{c}{Wall 1} & \multicolumn{2}{c}{Wall 2} & \multicolumn{2}{c}{Wall 3} & \multicolumn{2}{c}{Towall 1} & \multicolumn{2}{c}{Towall 2} & \multicolumn{2}{c}{Towall 3}\\\\ ')
    print(' & $\mu^-$ & $e^-$ & $\mu^-$ & $e^-$ & $\mu^-$ & $e^-$ & $\mu^-$ & $e^-$ & $\mu^-$ & $e^-$ & $\mu^-$ & $e^-$\\\\ ')
    print('\hline')
    for ig in range(npeak_total):
        print('{:d} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f} & {:.3f}$\pm${:.3f}\\\\ '.format(ig+1, mean_wall[0][0][ig], stdev_wall[0][0][ig], mean_wall[0][1][ig], stdev_wall[0][1][ig], mean_wall[1][0][ig], stdev_wall[1][0][ig], mean_wall[1][1][ig], stdev_wall[1][1][ig], mean_wall[2][0][ig], stdev_wall[2][0][ig], mean_wall[2][1][ig], stdev_wall[2][1][ig], mean_towall[0][0][ig], stdev_towall[0][0][ig], mean_towall[0][1][ig], stdev_towall[0][1][ig], mean_towall[1][0][ig], stdev_towall[1][0][ig], mean_towall[1][1][ig], stdev_towall[1][1][ig], mean_towall[2][0][ig], stdev_towall[2][0][ig], mean_towall[2][1][ig], stdev_towall[2][1][ig]))
    print('\hline \hline')

    
    #print useful info
    print('$N_{Gaussian}$ & \multicolumn{2}{c}{} & \multicolumn{2}{c}{No local min/max} & \multicolumn{2}{c}{Not a local min}\\\\ ')
    print(' & $\mu^-$ & $e^-$ & $\mu^-$ & $e^-$ & $\mu^-$ & $e^-$\\\\ ')
    print('\hline')
    for ig in range(npeak_total):
        print('{:d} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\ '.format(ig+1, 1 - np.sum(mask[0][ig])/n_scan, 1 - np.sum(mask[1][ig])/n_scan, np.sum(onbound[0][ig])/n_scan, np.sum(onbound[1][ig])/n_scan, np.sum(not_min[0][ig])/n_scan, np.sum(not_min[1][ig])/n_scan))
    print('\hline \hline')

    '''
    ##############Nmin################
    figNminAll, (axmu, axe) = plt.subplots(1,2)
    figNminAll.set_size_inches(8,3.5)
    axmu.set_prop_cycle(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    axe.set_prop_cycle(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    nminbins = [1,2,3,4,5,6,7,8,9,10,11]
    for ig in range(npeak_total):
        axmu.hist(np.clip(mu_nmin_all[ig][~np.isnan(mu_nmin_all[ig])], nminbins[0], nminbins[-1]), bins=nminbins, histtype='step', density = True, label=r'$\mu$ {:d} Gaussian'.format(ig+1))
        axe.hist(np.clip(e_nmin_all[ig][~np.isnan(e_nmin_all[ig])], nminbins[0], nminbins[-1]), bins=nminbins, histtype='step', density = True, label=r'e {:d} Gaussian'.format(ig+1))  
        axmu.set_ylabel("Event Rate")
        axe.set_ylabel("Event Rate")
        axmu.set_xlabel("Number of local min/max")
        axe.set_xlabel("Number of local min/max")
        axmu.set_ylim(0, 0.5)
        axmu.set_xlim(1, 11)
        axe.set_ylim(0, 0.5)
        axe.set_xlim(1, 11)
        axmu.legend(loc='upper right', ncol = 2, prop={'size': 5})
        axe.legend(loc='upper right', ncol = 2, prop={'size': 5})
        figNminAll.tight_layout(pad=0.3)
    pp = PdfPages(outdir+'/SK_MultiGaus_Nmin_NGaus_1_to_'+str(npeak_total)+'_time_'+str(tflag)+'_corr_'+str(cflag)+'_'+str(n_scan)+'_events.pdf')
    pp.savefig(figNminAll)
    plt.close(figNminAll)
    pp.close()   
    '''

def _plot_2D_heatmap(outdir, npeak_total, info_dict, n_scan, tflag, cflag) :

    fig =  plt.figure(figsize=(10.5,4.5))
    ax = []
    npmt = 11146
    nlevel = 20

    #some constants and axis range
    nhitax = np.linspace(0, 1, 100)
    wallax = np.linspace(0, 2000, 200)
    towallax = np.linspace(0, 5000, 500)
    dwmesh, twmesh = np.meshgrid(wallax, towallax)
    energyax = np.linspace(0, 7000, 100)
    Emesh, hitmesh = np.meshgrid(energyax, nhitax)
    Ermesh, twEmesh = np.meshgrid(energyax, towallax)

    flavor = ['muon', 'electron']
    flavor_sybname = ['\mu', 'e']
    residual = [[],[]]
    etrue=[[],[]]
    erec=[[],[]]
    nhit=[[],[]]
    wall = [[],[]]
    towall = [[],[]]
    onbound = [[],[]]
    not_min = [[],[]]

    ngaus = [ i+1 for i in range(npeak_total)]

    for ig in range(npeak_total):
        for j,f in enumerate(flavor):
            residual[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['energy_res'][0:n_scan])
            etrue[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['orig_energy'][0:n_scan])
            erec[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['reco_energy'][0:n_scan])
            nhit[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['nhit'][0:n_scan])
            wall[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['dwall'][0:n_scan])
            towall[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['towall'][0:n_scan]) 
            onbound[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['onbound'][0:n_scan])
            not_min[j].append(info_dict['NPeak{:d}'.format(ig+1)][f]['not_local_min'][0:n_scan])
            
    mask = np.array((np.array(onbound)==0)&(np.array(not_min)==0), dtype=bool)

    ####plot muon and electron######
    for idx,f in enumerate(flavor):
        ax.append(fig.add_subplot(1, 2, idx+1))
        ax[idx].set_xlabel('True Energy (MeV)', fontsize=8, loc='right')
        ax[idx].set_ylabel('True Nhit Fraction', fontsize=8, loc='top')
        ax[idx].tick_params(axis='x', labelsize=10)
        ax[idx].tick_params(axis='y', labelsize=10)
        ax[idx].set_xlim(0, np.max(np.array(etrue)[0][idx]))
        ax[idx].set_ylim(0, 1)
        Z = griddata((np.array(etrue)[idx][4], np.array(nhit)[idx][4]/npmt), np.abs(np.array(residual)[idx][4]), (Emesh, hitmesh), method='linear')
        heatmap = ax[idx].contourf(Emesh, hitmesh, Z, nlevel, vmin = -0.8, vmax = 0.8, cmap= 'magma')
        cbar=plt.colorbar(heatmap, ax=ax[idx])
        cbar.ax.set_ylabel(r'|$\Delta_{E}$|', rotation=270)
        ax[idx].set_title(r'${:s}^-$ 5 Gaussian'.format(flavor_sybname[idx]), y=1, fontsize=10)

    fig.tight_layout()
    pp = PdfPages(outdir+'/SK_MultiGaus_Etrue_v_nhit_Heatmap_NGaus_1_to_'+str(npeak_total)+'_time_'+str(tflag)+'_corr_'+str(cflag)+'_'+str(n_scan)+'_events.pdf')
    pp.savefig(fig)
    pp.close()

    for idx,f in enumerate(flavor):
        ax[idx].cla()
        ax[idx].set_xlabel('Dwall (cm)', fontsize=10, loc='right')
        ax[idx].set_ylabel('Towall (cm)', fontsize=10, loc='top')
        ax[idx].tick_params(axis='x', labelsize=10)
        ax[idx].tick_params(axis='y', labelsize=10)
        ax[idx].set_xlim(0, 2000)
        ax[idx].set_ylim(0, 5000)
        Z = griddata((np.array(wall)[idx][4], np.array(towall)[idx][4]), np.abs(np.array(residual)[idx][4]), (dwmesh, twmesh), method='linear')
        heatmap = ax[idx].contourf(dwmesh, twmesh, Z, nlevel, vmin = -0.8, vmax = 0.8, cmap= 'magma')
        ax[idx].set_title(r'${:s}^-$ 5 Gaussian'.format(flavor_sybname[idx]), y=1, fontsize=10)

    fig.tight_layout()
    pp = PdfPages(outdir+'/SK_MultiGaus_wall_v_towall_Heatmap_NGaus_1_to_'+str(npeak_total)+'_time_'+str(tflag)+'_corr_'+str(cflag)+'_'+str(n_scan)+'_events.pdf')
    pp.savefig(fig)
    pp.close()

    for idx,f in enumerate(flavor):
        ax[idx].cla()
        ax[idx].set_xlabel(r'$E_{rec}$ (MeV)', fontsize=10, loc='right')
        ax[idx].set_ylabel('Towall (cm)', fontsize=10, loc='top')
        ax[idx].tick_params(axis='x', labelsize=10)
        ax[idx].tick_params(axis='y', labelsize=10)
        ax[idx].set_xlim(0, 7000)
        ax[idx].set_ylim(0, 5000)
        Z = griddata((np.array(erec)[idx][5], np.array(towall)[idx][5]), np.abs(np.array(residual)[idx][5]), (Ermesh, twEmesh), method='linear')
        heatmap = ax[idx].contourf(Ermesh, twEmesh, Z, nlevel, vmin = -0.8, vmax = 0.8, cmap= 'magma')
        ax[idx].set_title(r'${:s}^-$ 5 Gaussian'.format(flavor_sybname[idx]), y=1, fontsize=10)

    fig.tight_layout()
    pp = PdfPages(outdir+'/SK_MultiGaus_erec_v_towall_Heatmap_NGaus_1_to_'+str(npeak_total)+'_time_'+str(tflag)+'_corr_'+str(cflag)+'_'+str(n_scan)+'_events.pdf')
    pp.savefig(fig)
    pp.close()
    fig.clf()

    '''
    ####plot electron######
    for idx, ig in enumerate([1, int(npeak_total/2), npeak_total]):
        ax[idx].set_xlabel('True Energy (MeV)', fontsize=10, loc='right')
        ax[idx].set_ylabel('True Nhit Fraction', fontsize=10, loc='top')
        ax[idx].tick_params(axis='x', labelsize=10)
        ax[idx].tick_params(axis='y', labelsize=10)
        ax[idx].set_xlim(0, np.max(np.array(etrue)[1][idx]))
        ax[idx].set_ylim(0, 1)
        energyax = np.linspace(0, np.max(np.array(etrue)[1][idx]), 100)
        Emesh, hitmesh = np.meshgrid(energyax, nhitax)
        Z = griddata((np.array(etrue)[1][idx], np.array(nhit)[1][idx]/npmt), np.abs(np.array(residual)[1][idx]), (Emesh, hitmesh), method='linear')
        heatmap = ax[idx].contourf(Emesh, hitmesh, Z, nlevel, vmin = -0.8, vmax = 0.8, cmap= 'magma')
        ax[idx].set_title(r'$e^-$ {:d} Gaussian'.format(ig), y=1, fontsize=10)

    pp = PdfPages(outdir+'/SK_MultiGaus_e_Etrue_v_nhit_Heatmap_NGaus_1_to_'+str(npeak_total)+'_time_'+str(tflag)+'_corr_'+str(cflag)+'_'+str(n_scan)+'_events.pdf')
    pp.savefig(fig)
    pp.close()

    for idx, ig in enumerate([1, int(npeak_total/2), npeak_total]):
        ax[idx].cla()
        ax[idx].set_xlabel('Dwall (cm)', fontsize=10, loc='right')
        ax[idx].set_ylabel('Towall (cm)', fontsize=10, loc='top')
        ax[idx].tick_params(axis='x', labelsize=10)
        ax[idx].tick_params(axis='y', labelsize=10)
        ax[idx].set_xlim(0, 2000)
        ax[idx].set_ylim(0, 5000)
        Z = griddata((np.array(wall)[1][idx], np.array(towall)[1][idx]), np.abs(np.array(residual)[1][idx]), (dwmesh, twmesh), method='linear')
        heatmap = ax[idx].contourf(dwmesh, twmesh, Z, nlevel, vmin = -0.8, vmax = 0.8, cmap= 'magma')
        ax[idx].set_title(r'$e^-$ {:d} Gaussian'.format(ig), y=1, fontsize=10)

    pp = PdfPages(outdir+'/SK_MultiGaus_e_wall_v_towall_Heatmap_NGaus_1_to_'+str(npeak_total)+'_time_'+str(tflag)+'_corr_'+str(cflag)+'_'+str(n_scan)+'_events.pdf')
    pp.savefig(fig)
    pp.close()
    '''

def _stack_scan_curves(fig, ax, info_dict, ig):
    #ax[0]: event display, ax[1]: scan curves
    energies = np.squeeze(np.array([info_dict['scan_energy']], dtype=float))
    losses = np.squeeze(np.array([info_dict['scan_loss']], dtype=float))

    if len(energies) == 0 or len(losses) == 0:
        print("Has no array to interpolate for N_GAUS=", ig+1)
        sys.exit()    
    
    splELoss = InterpolatedUnivariateSpline(energies, losses, k=4)
    crptsELoss = splELoss.derivative().roots()
    if len(crptsELoss) > 0:
        minloss = find_cubicspline_min(splELoss, crptsELoss)
    else:
        minloss = energies[np.argmin(losses)]

    loss_range = losses.max() - losses.min()
    ax[1].plot(energies, splELoss(energies)-splELoss(minloss), linestyle="--", linewidth=0.5, alpha = 1)
    ax[1].scatter(minloss, 0, label='{:d}'.format(ig+1), marker="^", s=8, alpha=0.5)
    ax[1].set_ylim(-0.01*loss_range, 0.02*loss_range)

def _single_gaussian(x, *p):
    amplitude, mean, stddev = p
    return amplitude * np.exp(-(x - mean)**2 / 2 / stddev**2)

def _simple_gaussians(nG, x, coeff, mu, var, charge_scale):
    gauss = [0]*(nG+1)
    tot = 0
    if nG != len(mu):
        print('Gaussian input dimensions do not agree!')
        return gauss
    for i in range(nG):
        tot += coeff[i]*(1./(2*np.pi*var[i])**0.5)*np.exp(-np.power(x/charge_scale - mu[i], 2.) / (2 * var[i]))
        gauss[i] = (coeff[i]*(1./(2*np.pi*var[i])**0.5)*np.exp(-np.power(x/charge_scale - mu[i], 2.) / (2 * var[i])))
    gauss[-1] = tot
    return  gauss

def _user_softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


