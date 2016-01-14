import numpy as np
import pandas as pd
import dicom
import glob
import os
import sys
import math
import timeit

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


# pyscience.wordpress.com/2014/09/08/dicom-in-python-importing-medical-image-data-into-numpy-with-pydicom-and-vtk/
def ReadDicom(dicom_dir='dicom', points_threshold=1100):
    # Get the dicom file list.
    files = sorted(glob.glob(os.path.join(dicom_dir, 'CT*')))

    # Get CT meta data.
    data = dicom.read_file(files[0])
    dims = (int(data.Rows), int(data.Columns), len(files))
    # spacing is in mm.
    spaces = (float(data.PixelSpacing[0]), float(data.PixelSpacing[1]), float(data.SliceThickness))

    # Allocate spaces, loop through all the DICOM files
    v = np.zeros(dims, dtype=data.pixel_array.dtype)
    for f in files:
        data = dicom.read_file(f)
        v[:, :, files.index(f)] = data.pixel_array  

    # Get 3 dimension data. Use t as the filter.
    t = v >= points_threshold
    a = t.nonzero()
    v = v[t]
    v = v.reshape(len(a[0]))
    data = np.array([a[0] * spaces[0], a[1] * spaces[1], a[2] * spaces[2], v])

    # Return a ? * 4 matrix
    data = data.transpose()
    return data


def Cart2Polar(data):
    d = data[:,0:3]
    m = d.mean(axis=0)
    d = d - m
    d = d.transpose()
    r = np.sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2])
    theta = np.arccos(d[2] / r)
    phi = np.arctan2(d[1], d[0])
    p = np.array([theta, phi, r]).transpose()
    return p, m


def Polar2Cart(polar, mean):
    p = polar.transpose()
    x = p[2] * np.sin(p[0])
    y = x * np.sin(p[1])
    x = x * np.cos(p[1])
    z = p[2] * np.cos(p[0])
    d = np.array([x, y, z]).transpose() + mean
    return d


def GetMinMaxPoints(polar, step = 180):
    step_l = np.pi / step
    polar = pd.DataFrame(polar)
    polar.columns=list('tpr')
    polar['ti'] = np.floor(polar['t'] / step_l)
    polar['pi'] = np.floor(polar['p'] / step_l)
    # the index of inner, outer surfaces and 'pi'.
    mM = [[v['r'].argmin(), v['r'].argmax()] + list(k) for k, v in polar.groupby(['ti', 'pi'])]
    # Remove points if min == Max
    mM = [m for m in mM if m[0] != m[1]]
    return np.array(mM, dtype='int64')


start_time = timeit.default_timer()
data = ReadDicom()
polar, mean = Cart2Polar(data)
# data2 = Polar2Cart(polar, mean) # verify the conversion by reverse.
# minMax = GetMinMaxPoints(polar)
minMax = GetMinMaxPoints(polar, 180)

elapsed = timeit.default_timer() - start_time
print('Total running time: %d seconds' % elapsed)

# ------------------------------------------------------------- #
def GetCircle(minMax, i, merged):
    d = np.concatenate((merged[minMax[:,i]], minMax[:,[2,3]]), axis=1)
    c = pd.DataFrame(d, columns=list('xyzvtpr') + ['ti', 'pi'])
    return c



def GetShiftDiffCol(col):
    shifted = col.shift(1, axis=0)
    shifted.iloc[0] = col.iloc[-1]
    return shifted - col

def GetCircleLen(surface):
    surface = surface[list('xy') + ['ti','pi']]
    surface = surface.sort(columns=['ti','pi'])
    size = surface['ti'].unique().shape[0]
    result = pd.DataFrame(columns=['ti','l','count'], data=np.zeros([size, 3]))
    i = 0
    for k, g in surface.groupby('ti'):
        g['xd'] = GetShiftDiffCol(g['x'])
        g['yd'] = GetShiftDiffCol(g['y'])
        d = sum((g['xd'] * g['xd'] + g['yd']*g['yd']).apply(math.sqrt))
        result.loc[i, :] = [k, d, g.shape[0]]
        i = i + 1
    return result

merged = np.concatenate((data, polar), axis=1)
inner_c = GetCircle(minMax, 0, merged)
outer_c = GetCircle(minMax, 1, merged)
outer_c['thick'] = polar[minMax[:,1], 2] - polar[minMax[:,0], 2]

inner_circle = GetCircleLen(inner_c)
outer_circle = GetCircleLen(outer_c)
circle = inner_circle.merge(outer_circle, on=['ti', 'count'], suffixes=['_i', '_o'])

"""
inner_c.to_csv('inner_surface.csv', index=None)
outer_c.to_csv('outer_surface.csv', index=None)
circle.to_csv('circle.csv', index=None)
"""

# ------------------------------------------------------------- #
matplotlib.use("Agg")
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Plot circle with the same p_i.
def PlotCircle(i, inner=None, outer=None):
    fig = plt.figure()
    if not inner.empty:
        idx = inner['ti'] == i
        plt.scatter(inner.loc[idx,'x'], inner.loc[idx, 'y'], c='b')
    if not outer.empty:
        idx = outer['ti'] == i
        plt.scatter(outer.loc[idx,'x'], outer.loc[idx, 'y'], c='r')
    plt.show(block=False)


frames = np.array(circle['ti'].values)
circle.set_index('ti', inplace=True)

def init():
    plot_i.set_offsets([])
    plot_o.set_offsets([])
    idx_txt.set_text('')
    ilen_txt.set_text('')
    olen_txt.set_text('')
    return plot_i, plot_o, idx_txt, ilen_txt, olen_txt

def update(i):
    idx = inner_c['ti'] == frames[i]
    plot_i.set_offsets(np.array(inner_c.loc[idx,list('xy')].values))
    idx = outer_c['ti'] == frames[i]
    plot_o.set_offsets(np.array(outer_c.loc[idx,list('xy')].values))
    idx_txt.set_text('theta = %3d' % frames[i])
    li = circle.loc[frames[i], 'l_i']
    lo = circle.loc[frames[i], 'l_o']
    if li > lo:
        ilen_txt.set_color('r')
        olen_txt.set_color('r')
    else:
        ilen_txt.set_color('k')
        olen_txt.set_color('k')
    ilen_txt.set_text('inner = %.2f' % li)
    olen_txt.set_text('outer = %.2f' % lo)
    return plot_i, plot_o, idx_txt, ilen_txt, olen_txt

t = outer_c[list('xy')]
bd = np.array([t.min().values, t.max().values])
t = (bd[1, :] - bd[0, :]) * 0.1
bd[0, :] -= t
bd[1, :] += t

fig = plt.figure()
# plot_o = plt.scatter([], [], c='r')
# plot_i = plt.scatter([], [], c='b')
ax = fig.add_subplot(111, autoscale_on=False, xlim=tuple(bd[:,0]), ylim=tuple(bd[:,1]))
plot_o = ax.scatter([], [], c='r', edgecolor='r')
plot_i = ax.scatter([], [], c='b', edgecolor='b')
idx_txt = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ilen_txt = ax.text(0.02, 0.90, '', transform=ax.transAxes)
olen_txt = ax.text(0.02, 0.85, '', transform=ax.transAxes)
ax.grid()

action = animation.FuncAnimation(fig, update, len(frames), interval=150, blit=True, init_func=init)
#plt.show()

# Set up formatting for the movie files
Writer = animation.writers['mencoder']
writer = Writer(fps=15, metadata=dict(artist='Me'))
action.save('action.mp4', writer=writer)


sys.exit(0)
"""
idx = circle['ti'] == 39
circle.loc[idx]
circle.head()

idx = circle['circle_i'] > circle['circle_o']
circle[idx].describe()
circle[~idx].describe()


# 39    360  17.519599  18.478689
PlotCircle(164, inner_c, outer_c)
PlotCircle(111, inner_c, outer_c)
PlotCircle(117, inner_c, pd.DataFrame())
PlotCircle(39, pd.DataFrame(), outer_c)
"""

# ------------------------------------------------------------- #
# cmap: http://matplotlib.org/examples/color/colormaps_reference.html
def PlotSkull(data_p, bounds):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    lb = data_p['thick'].quantile(bounds[0])
    ub = data_p['thick'].quantile(bounds[1])
    plt_data = data_p
    color = (plt_data['thick'] - lb) / (ub - lb)
    color[color<0] = 0
    color[color>1] = 1
    ax.scatter(plt_data['x'], plt_data['y'], plt_data['z'], c=color, cmap=plt.cm.get_cmap('rainbow'), marker='.', edgecolor='none')
    plt.show(block=False)


PlotSkull(outer_c, [0.05, 0.95])
# ------------------------------------------------------------- #
