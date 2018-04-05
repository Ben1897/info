"""
A set of functions for plotting results from SUR partitioning.

@Author: Peishi Jiang <Ben1897>
@Email:  shixijps@gmail.com

plot_sr_comparison3d()
plot_sr_prop_comparison3d()
plot_sur_prop()
plot_ii()
plot_sr_comparison1d()
plot_sur_1d()
plot_ii1d()

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.axes3d import Axes3D


def plot_sr_comparison3d(xv, yv, rc, r, sc, s, xlabel, ylabel, vmin=0, vmax=1):
    '''Plot the synergistic and redundant information from both II and MIIT in 3d format.'''
    plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=.4, hspace=.4)

    ax = plt.subplot(gs[0,0], projection='3d')
    ax.view_init(20, 225)
    cs = ax.plot_surface(xv, yv, rc, cmap=plt.get_cmap('jet'), vmin=vmin, vmax=vmax,
                         linewidth=0, antialiased=False)
    plt.colorbar(cs, ax=ax)
    ax.set_zlim([-.1, 1.1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('Redundant Info (w/ cond\'t)')

    ax = plt.subplot(gs[0,1], projection='3d')
    ax.view_init(20, 225)
    cs = ax.plot_surface(xv, yv, r, cmap=plt.get_cmap('jet'), vmin=vmin, vmax=vmax,
                    linewidth=0, antialiased=False)
    plt.colorbar(cs, ax=ax)
    ax.set_zlim([-.1, 1.1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('Redundant Info (w/o cond\'t)')

    ax = plt.subplot(gs[1,0], projection='3d')
    ax.view_init(20, 225)
    cs = ax.plot_surface(xv, yv, sc, cmap=plt.get_cmap('jet'), vmin=vmin, vmax=vmax,
                         linewidth=0, antialiased=False)
    plt.colorbar(cs, ax=ax)
    ax.set_zlim([-.1, 1.1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('Synergistic Info (w/ cond\'t)')

    ax = plt.subplot(gs[1,1], projection='3d')
    ax.view_init(20, 225)
    cs = ax.plot_surface(xv, yv, s, cmap=plt.get_cmap('jet'), vmin=vmin, vmax=vmax,
                         linewidth=0, antialiased=False)
    plt.colorbar(cs, ax=ax)
    ax.set_zlim([-.1, 1.1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('Synergistic Info (w/o cond\'t)')


def plot_sr_prop_comparison3d(xv, yv, rc, r, sc, s, xlabel, ylabel, vmin=0, vmax=1):
    '''Plot the proportion of the synergistic and redundant information from both II and MIIT in 3d format.'''
    plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=.4, hspace=.4)

    ax = plt.subplot(gs[0,0], projection='3d')
    ax.view_init(20, 225)
    cs = ax.plot_surface(xv, yv, rc, cmap=plt.get_cmap('jet'), vmin=vmin, vmax=vmax,
                         linewidth=0, antialiased=False)
    plt.colorbar(cs, ax=ax)
    ax.set_zlim([-.1, 1.1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('(Proportion) Redundant Info (w/ cond\'t)')

    ax = plt.subplot(gs[0,1], projection='3d')
    ax.view_init(20, 225)
    cs = ax.plot_surface(xv, yv, r, cmap=plt.get_cmap('jet'), vmin=vmin, vmax=vmax,
                    linewidth=0, antialiased=False)
    plt.colorbar(cs, ax=ax)
    ax.set_zlim([-.1, 1.1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('(Proportion) Redundant Info (w/o cond\'t)')

    ax = plt.subplot(gs[1,0], projection='3d')
    ax.view_init(20, 225)
    cs = ax.plot_surface(xv, yv, sc, cmap=plt.get_cmap('jet'), vmin=vmin, vmax=vmax,
                         linewidth=0, antialiased=False)
    plt.colorbar(cs, ax=ax)
    ax.set_zlim([-.1, 1.1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('(Proportion) Synergistic Info (w/ cond\'t)')

    ax = plt.subplot(gs[1,1], projection='3d')
    ax.view_init(20, 225)
    cs = ax.plot_surface(xv, yv, s, cmap=plt.get_cmap('jet'), vmin=vmin, vmax=vmax,
                         linewidth=0, antialiased=False)
    plt.colorbar(cs, ax=ax)
    ax.set_zlim([-.1, 1.1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('(Proportion) Synergistic Info (w/o cond\'t)')


def plot_sur_prop(xv, yv, rp, sp, uxp, uyp, xlabel, ylabel, titleext='w/ cond\'t'):
    '''Plot the proportions of SUR.'''
    vmin, vmax = 0, 1
    extent = [xv.min(), xv.max(), yv.max(), yv.min()]

    plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=.4, hspace=.4)

    ax = plt.subplot(gs[0,0], projection='3d')
    ax.view_init(20, 225)
    cs = ax.plot_surface(xv, yv, rp, cmap=plt.get_cmap('jet'), vmin=vmin, vmax=vmax,
                         linewidth=0, antialiased=False)
    plt.colorbar(cs, ax=ax)
    ax.set_zlim([-.1, 1.1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('(Proportion) Redundant Info ' + titleext)

    ax = plt.subplot(gs[0,1], projection='3d')
    ax.view_init(20, 225)
    cs = ax.plot_surface(xv, yv, sp, cmap=plt.get_cmap('jet'), vmin=vmin, vmax=vmax,
                    linewidth=0, antialiased=False)
    plt.colorbar(cs, ax=ax)
    ax.set_zlim([-.1, 1.1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('(Proportion) Synergistic Info ' + titleext)

    ax = plt.subplot(gs[1,0], projection='3d')
    ax.view_init(20, 225)
    cs = ax.plot_surface(xv, yv, uxp, cmap=plt.get_cmap('jet'), vmin=vmin, vmax=vmax,
                         linewidth=0, antialiased=False)
    plt.colorbar(cs, ax=ax)
    ax.set_zlim([-.1, 1.1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('(Proportion) Unique Info of X ' + titleext)

    ax = plt.subplot(gs[1,1], projection='3d')
    ax.view_init(20, 225)
    cs = ax.plot_surface(xv, yv, uyp, cmap=plt.get_cmap('jet'), vmin=vmin, vmax=vmax,
                         linewidth=0, antialiased=False)
    plt.colorbar(cs, ax=ax)
    ax.set_zlim([-.1, 1.1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('(Proportion) Unique Info of Y ' + titleext)

def plot_ii(xv, yv, iic, ii, itc, it, xlabel, ylabel):
    '''Plot the interaction information and the total information for comparison.'''
    vmin1, vmax1 = np.min([iic, ii]), np.max([iic, ii])
    vmin2, vmax2 = np.min([itc, it]), np.max([itc, it])
    extent = [xv.min(), xv.max(), yv.min(), yv.max()]

    plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=.4, hspace=.4)

    # ax = plt.subplot(gs[0,0], projection='3d')
    # ax.view_init(20, 225)
    # cs = ax.plot_surface(xv, yv, iic, cmap=plt.get_cmap('jet'), vmin=vmin1, vmax=vmax1,
    #                      linewidth=0, antialiased=False)
    # plt.colorbar(cs, ax=ax)
    # ax.set_zlim([-.1, 1.1])
    ax = plt.subplot(gs[0,0])
    cs=ax.imshow(iic, cmap=plt.get_cmap('jet'), vmin=vmin1, vmax=vmax1, extent=extent,
              interpolation='bilinear')
    plt.colorbar(cs, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('MIIT')

    # ax = plt.subplot(gs[0,1], projection='3d')
    # ax.view_init(20, 225)
    # cs = ax.plot_surface(xv, yv, ii, cmap=plt.get_cmap('jet'), vmin=vmin1, vmax=vmax1,
    #                 linewidth=0, antialiased=False)
    # plt.colorbar(cs, ax=ax)
    # ax.set_zlim([-.1, 1.1])
    ax = plt.subplot(gs[0,1])
    cs=ax.imshow(ii, cmap=plt.get_cmap('jet'), vmin=vmin1, vmax=vmax1, extent=extent,
              interpolation='bilinear')
    plt.colorbar(cs, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('II')

    # ax = plt.subplot(gs[1,0], projection='3d')
    # ax.view_init(20, 225)
    # cs = ax.plot_surface(xv, yv, itc, cmap=plt.get_cmap('jet'), vmin=vmin2, vmax=vmax2,
    #                      linewidth=0, antialiased=False)
    # plt.colorbar(cs, ax=ax)
    # ax.set_zlim([-.1, 1.1])
    ax = plt.subplot(gs[1,0])
    cs=ax.imshow(itc, cmap=plt.get_cmap('jet'), vmin=vmin2, vmax=vmax2, extent=extent,
              interpolation='bilinear')
    plt.colorbar(cs, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('I(X,Y;Z|W)')

    # ax = plt.subplot(gs[1,1], projection='3d')
    # ax.view_init(20, 225)
    # cs = ax.plot_surface(xv, yv, it, cmap=plt.get_cmap('jet'), vmin=vmin2, vmax=vmax2,
    #                      linewidth=0, antialiased=False)
    # plt.colorbar(cs, ax=ax)
    # ax.set_zlim([-.1, 1.1])
    ax = plt.subplot(gs[1,1])
    cs= ax.imshow(it, cmap=plt.get_cmap('jet'), vmin=vmin2, vmax=vmax2, extent=extent,
              interpolation='bilinear')
    plt.colorbar(cs, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('I(X,Y;Z)')


def plot_sr_comparison1d(xset, rc, r, sc, s, xlabel, title, proportion=True):
    '''Plot the proportion of the synergistic and redundant information from both II and MIIT in 1d format.'''
    plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=.4, hspace=.4)

    ax = plt.subplot(gs[0,0])
    ax.plot(xset, rc, 'b', label='R (w/ condition)')
    ax.plot(xset, r, 'r', label='R (w/o condition)')
    ax.plot(xset, sc, 'b-.', label='S (w/ condition)')
    ax.plot(xset, s, 'r-.', label='S (w/o condition)')
    ax.legend(loc='upper right')
    ax.set_xlabel(xlabel)
    if proportion:
        ax.set_ylabel('Proportion')
        # ax.set_ylim([0,1])
    ax.set_ylabel('Information (bit)')
    ax.set_title(title)


def plot_sur_1d(xset, rp, sp, uxp, uyp, xlabel, title, proportion=True):
    '''Bar plot the SUR.'''
    plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=.4, hspace=.4)

    ax = plt.subplot(gs[0,0])
    bar1 = ax.bar(xset, rp, label='Redundant', color='b', edgecolor='white')
    bar2 = ax.bar(xset, sp, label='Synergistic', color='r', edgecolor='white', bottom=rp)
    bar3 = ax.bar(xset, uxp, label='Unique(X)', color='y', edgecolor='white', bottom=rp+sp)
    bar4 = ax.bar(xset, uyp, label='Unique(Y)', color='c', edgecolor='white', bottom=rp+sp+uxp)
    ax.legend(loc='upper right')
    # ax.bar(xset, rp, label='Redundant', color='b', width=bar_width, edgecolor='white')
    # ax.bar(xset, sp, label='Synergistic', color='r', width=bar_width, edgecolor='white')
    # ax.bar(xset, uxp, label='Unique(X)', color='y', width=bar_width, edgecolor='white')
    # ax.bar(xset, uyp, label='Unique(Y)', color='c', width=bar_width, edgecolor='white')
    ax.set_xlabel(xlabel)
    if proportion:
        ax.set_ylabel('Proportion')
        # ax.set_ylim([0,1])
    else:
        ax.set_ylabel('Information (bit)')
    ax.set_title(title)


def plot_ii1d(xset, iic, ii, itc, it, xlabel, ylabel, title):
    '''Plot the interaction information and the total information for comparison.'''
    plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=.4, hspace=.4)

    ax = plt.subplot(gs[0,0])
    ax.plot(xset, iic, 'b', label='MIIT')
    ax.plot(xset, ii, 'r', label='II')
    ax.plot(xset, itc, 'b-.', label='Itotal (w/ condition)')
    ax.plot(xset, it, 'r-.', label='Itotal (w/o condition)')
    ax.legend(loc='upper right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_pid(xv, yv, iic, rp, sp, uxp, uyp, xlabel, ylabel, zlabel='$[nats]$',
             vmin=0, vmax=1, vertical=False, option='MPID', prop=True):
    '''Plot the all the information of PID.'''
    # vmin, vmax = 0, 1
    if vmax < 0.5:
        zlim = [-0.1, 0.5]
        zticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    else:
        zlim = [-0.1, 1.0]
        zticks = [0.0, 0.5, 1.0]
    labelpad = 20
    extent = [xv.min(), xv.max(), yv.max(), yv.min()]

    if vertical:
        plt.figure(figsize=(18, 32))
        gs = gridspec.GridSpec(5, 1)
        gs.update(wspace=.1, hspace=.2)
    else:
        plt.figure(figsize=(40,8))
        gs = gridspec.GridSpec(1, 5)
        gs.update(wspace=.1, hspace=.1)

    if prop:
        proptext = '(proption)'
    else:
        proptext = ''

    # Interaction information
    # ax = plt.subplot(gs[0,0])
    # cs=ax.imshow(iic, cmap=plt.get_cmap('jet'), vmin=vmin1, vmax=vmax1, extent=extent,
    #           interpolation='bilinear')
    # plt.colorbar(cs, ax=ax)
    # ax.set_xlabel(xlabel, labelpad=labelpad)
    # ax.set_ylabel(ylabel, labelpad=labelpad)
    # if option == 'MPID':
    #     ax.set_title(r'$MII-ICP$ %s' % proptext)
    # elif option == 'II':
    #     ax.set_title(r'$II$ %s' % proptext)
    ax = plt.subplot(gs[0], projection='3d')
    ax.view_init(20, 225)
    cs = ax.plot_surface(xv, yv, iic, cmap=plt.get_cmap('jet'), vmin=vmin, vmax=vmax,
                         linewidth=0, antialiased=False)
    plt.colorbar(cs, ax=ax, ticks=zticks)
    ax.set_zlim(zlim)
    ax.set_zticks(zticks)
    ax.set_xlabel(xlabel, labelpad=labelpad)
    ax.set_ylabel(ylabel, labelpad=labelpad)
    if not prop:
        ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax.set_zlabel(zlabel, labelpad=labelpad, rotation=90)
    if option == 'MPID':
        ax.set_title(r'$MII-SCP$ %s' % proptext)
    elif option == 'II':
        ax.set_title(r'$II$ %s' % proptext)

    # Redundant information
    ax = plt.subplot(gs[1], projection='3d')
    ax.view_init(20, 225)
    cs = ax.plot_surface(xv, yv, rp, cmap=plt.get_cmap('jet'), vmin=vmin, vmax=vmax,
                         linewidth=0, antialiased=False)
    plt.colorbar(cs, ax=ax, ticks=zticks)
    ax.set_zlim(zlim)
    ax.set_zticks(zticks)
    ax.set_xlabel(xlabel, labelpad=labelpad)
    ax.set_ylabel(ylabel, labelpad=labelpad)
    if not prop:
        ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax.set_zlabel(zlabel, labelpad=labelpad, rotation=90)
    if option == 'MPID':
        ax.set_title(r'$R_c$ %s' % proptext)
    elif option == 'II':
        ax.set_title(r'$R$ %s' % proptext)

    # Synergistic information
    ax = plt.subplot(gs[2], projection='3d')
    ax.view_init(20, 225)
    cs = ax.plot_surface(xv, yv, sp, cmap=plt.get_cmap('jet'), vmin=vmin, vmax=vmax,
                    linewidth=0, antialiased=False)
    plt.colorbar(cs, ax=ax, ticks=zticks)
    ax.set_zlim(zlim)
    ax.set_zticks(zticks)
    ax.set_xlabel(xlabel, labelpad=labelpad)
    ax.set_ylabel(ylabel, labelpad=labelpad)
    if not prop:
        ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax.set_zlabel(zlabel, labelpad=labelpad, rotation=90)
    if option == 'MPID':
        ax.set_title(r'$S_c$ %s' % proptext)
    elif option == 'II':
        ax.set_title(r'$S$ %s' % proptext)

    # Unique information for x
    ax = plt.subplot(gs[3], projection='3d')
    ax.view_init(20, 225)
    cs = ax.plot_surface(xv, yv, uxp, cmap=plt.get_cmap('jet'), vmin=vmin, vmax=vmax,
                         linewidth=0, antialiased=False)
    plt.colorbar(cs, ax=ax, ticks=zticks)
    ax.set_zlim(zlim)
    ax.set_zticks(zticks)
    ax.set_xlabel(xlabel, labelpad=labelpad)
    ax.set_ylabel(ylabel, labelpad=labelpad)
    if not prop:
        ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax.set_zlabel(zlabel, labelpad=labelpad, rotation=90)
    if option == 'MPID':
        ax.set_title(r'$U_{X,c}$ %s' % proptext)
    elif option == 'II':
        ax.set_title(r'$U_{X}$ %s' % proptext)

    # Unique information for y
    ax = plt.subplot(gs[4], projection='3d')
    ax.view_init(20, 225)
    cs = ax.plot_surface(xv, yv, uyp, cmap=plt.get_cmap('jet'), vmin=vmin, vmax=vmax,
                         linewidth=0, antialiased=False)
    plt.colorbar(cs, ax=ax, ticks=zticks)
    ax.set_zlim(zlim)
    ax.set_zticks(zticks)
    ax.set_xlabel(xlabel, labelpad=labelpad)
    ax.set_ylabel(ylabel, labelpad=labelpad)
    if not prop:
        ax.zaxis.set_rotate_label(False)  # disable automatic rotation
        ax.set_zlabel(zlabel, labelpad=labelpad, rotation=90)
    if option == 'MPID':
        ax.set_title(r'$U_{Y,c}$ %s' % proptext)
    elif option == 'II':
        ax.set_title(r'$U_{Y}$ %s' % proptext)

# Plot state-dependent information metrics
def plot_sdim(results, pdfxy, pdfx_y, pdfy_x, coordx, coordy,
              nx, ny, xlabel='X', ylabel='Y'):
    import seaborn as sns
    c = 'blue'
    xticks, yticks = [0, nx], [0, ny]
    xticklabels = ['%.1f' % coordx[0], '%.1f' % coordx[-1]]
    yticklabels = ['%.1f' % coordy[-1], '%.1f' % coordy[0]]
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(2, 4, height_ratios=[1,1], width_ratios=[1,1,1,1])
    gs.update(wspace=0.4, hspace=0.5)

    # Plot p(X=x;Y=y)
    ax = fig.add_subplot(gs[0, 0])
    ims = sns.heatmap(pdfxy,
    #                   vmin=vmin, vmax=vmax,
                      ax=ax)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation='horizontal')
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, rotation='horizontal')
    ax.set_ylabel(r'$'+ylabel+'$')
    ax.set_xlabel(r'$'+xlabel+'$')
    ax.set_title(r'$p(X=x,Y=y)$')
    ax.grid(False)  # Get rid of the gridlines in seaborn

    # Plot p(Y=y|X=x)
    ax = fig.add_subplot(gs[0, 1])
    ims = sns.heatmap(pdfy_x,
                      ax=ax)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation='horizontal')
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, rotation='horizontal')
    ax.set_ylabel(r'$'+ylabel+'$')
    ax.set_xlabel(r'$'+xlabel+'$')
    ax.set_title(r'$p(Y=y|X=x)$')
    ax.grid(False)  # Get rid of the gridlines in seaborn

    # Plot p(X=x|Y=y)
    ax = fig.add_subplot(gs[0, 2])
    ims = sns.heatmap(pdfx_y,
                      ax=ax)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation='horizontal')
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, rotation='horizontal')
    ax.set_ylabel(r'$'+ylabel+'$')
    ax.set_xlabel(r'$'+xlabel+'$')
    ax.set_title(r'$p(X=x|Y=y)$')
    ax.grid(False)  # Get rid of the gridlines in seaborn

    # Plot I(X=x,Y=y;Z)
    ax = fig.add_subplot(gs[1, 0])
    ims = sns.heatmap(results.itots, cbar_kws={'label':r'$(nat)$'},
                      ax=ax)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation='horizontal')
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, rotation='horizontal')
    ax.set_ylabel(r'$'+ylabel+'$')
    ax.set_xlabel(r'$'+xlabel+'$')
    ax.set_title(r'$I(X=x,Y=y;Z)$')
    ax.grid(False)  # Get rid of the gridlines in seaborn

    # Plot II(X=x;Y=y;Z)
    ax = fig.add_subplot(gs[1, 3])
    ims = sns.heatmap(results.iis, cbar_kws={'label':r'$(nat)$'},
                      ax=ax)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation='horizontal')
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, rotation='horizontal')
    ax.set_ylabel(r'$'+ylabel+'$')
    ax.set_xlabel(r'$'+xlabel+'$')
    ax.set_title(r'$II(X=x;Y=y;Z)$')
    ax.grid(False)  # Get rid of the gridlines in seaborn

    # Plot I(X->Z)
    ax = fig.add_subplot(gs[1, 1])
    ims = sns.heatmap(results.ixsz, cbar_kws={'label':r'$(nat)$'},
                      ax=ax)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation='horizontal')
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, rotation='horizontal')
    ax.set_ylabel(r'$'+ylabel+'$')
    ax.set_xlabel(r'$'+xlabel+'$')
    ax.set_title(r'$I_{X->Z}(x,y)$')
    ax.grid(False)  # Get rid of the gridlines in seaborn

    # Plot I(Y->Z)
    ax = fig.add_subplot(gs[1, 2])
    ims = sns.heatmap(results.iysz, cbar_kws={'label':r'$(nat)$'},
                      ax=ax)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation='horizontal')
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, rotation='horizontal')
    ax.set_ylabel(r'$'+ylabel+'$')
    ax.set_xlabel(r'$'+xlabel+'$')
    ax.set_title(r'$I_{Y->Z}(x,y)$')
    ax.grid(False)  # Get rid of the gridlines in seaborn

    return fig, gs
