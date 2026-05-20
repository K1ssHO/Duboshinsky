import numpy as np
import matplotlib.pyplot as plt

b = 0.1
w = 0.3
A = 0.5
W = 3
porog = 0.2
al = 1
maxosh = 0.001

def f(x):
    return al * x**3

def eps(x):
    if abs(x) <= porog:
        return 1
    return 0

def fi(x, t):
    return A * eps(x) * np.sin(W * t)

def rg(x0, v0, t0, t1, dt):
    n = int((t1 - t0) / dt) + 1
    t = np.linspace(t0, t1, n)
    x = np.zeros(n)
    v = np.zeros(n)
    x[0] = x0
    v[0] = v0
    
    for i in range(n-1):
        xx = x[i]
        vv = v[i]
        tt = t[i]
        
        k1x = vv
        k1v = -2*b*vv - w*w*xx - f(xx) + fi(xx, tt)
        
        k2x = vv + 0.5*dt*k1v
        k2v = -2*b*(vv+0.5*dt*k1v) - w*w*(xx+0.5*dt*k1x) - f(xx+0.5*dt*k1x) + fi(xx+0.5*dt*k1x, tt+0.5*dt)
        
        k3x = vv + 0.5*dt*k2v
        k3v = -2*b*(vv+0.5*dt*k2v) - w*w*(xx+0.5*dt*k2x) - f(xx+0.5*dt*k2x) + fi(xx+0.5*dt*k2x, tt+0.5*dt)
        
        k4x = vv + dt*k3v
        k4v = -2*b*(vv+dt*k3v) - w*w*(xx+dt*k3x) - f(xx+dt*k3x) + fi(xx+dt*k3x, tt+dt)
        
        x[i+1] = xx + dt/6*(k1x+2*k2x+2*k3x+k4x)
        v[i+1] = vv + dt/6*(k1v+2*k2v+2*k3v+k4v)
    
    return t, x, v

def check(x0, v0, t0, t1, dt):
    dt_cur = dt
    err = 1.0
    
    while err > maxosh:
        t1_arr, x1, v1 = rg(x0, v0, t0, t1, dt_cur)
        t2_arr, x2, v2 = rg(x0, v0, t0, t1, dt_cur/2)
        
        x2_sampled = x2[::2]
        v2_sampled = v2[::2]
        
        err_x = np.abs(x1 - x2_sampled) / 15
        err_v = np.abs(v1 - v2_sampled) / 15
        err = np.maximum(err_x, err_v).max()
        
        if err > maxosh:
            dt_cur = dt_cur * 0.5
        
        if dt_cur < 0.0001:
        	break 
    
    return dt_cur, err, x1, v1, t1_arr

v0 = 0
x0 = 0
t0 = 0
t1 = 100
dt_start = 0.01

dt_final, err, x, v, t = check(x0, v0, t0, t1, dt_start)

plt.figure(figsize=(8,6))
plt.plot(x, v, 'b-', linewidth=0.8)
plt.xlabel('x')
plt.ylabel('v')
plt.grid(True, alpha=0.3)

textstr = f'β={b}\nω₀={w}\nA={A}\nΩ={W}\nα={al}\nx₀={x0}\nv₀={v0}'
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.savefig(r'C:\Users\aleks\Desktop\Новая папка (4)\ИП\phase_portrait1.png', dpi=150, bbox_inches='tight')
plt.show()

