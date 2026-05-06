import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# =========================================================
# Environment
# =========================================================
class Env:
    def __init__(self):
        self.dt = 0.05
        self.state = np.zeros(6)

    def reset(self):
        self.state[:] = 0
        return self.state.copy()

    def step(self, a, t):
        ext = np.zeros(3)

        # 强干扰
        if 10 <= t < 15:
            ext = np.array([5,0,-3])

        # 软干扰
        if 20 <= t < 25:
            ext += 0.8*np.array([
                np.sin(2*t),
                np.cos(1.6*t),
                np.sin(0.8*t)
            ])

        # 遮挡噪声
        if 30 < t < 40:
            noise = np.random.randn(6)*0.8
        else:
            noise = np.random.randn(6)*0.02

        pos = self.state[:3]
        vel = self.state[3:]

        acc = a + ext
        vel = vel + acc*self.dt
        pos = pos + vel*self.dt

        self.state = np.concatenate([pos, vel]) + noise
        return self.state.copy(), ext


# =========================================================
# SCM
# =========================================================
class SCM:
    def __init__(self, enabled=True):
        self.W = np.zeros((6,3))
        self.enabled = enabled

    def predict(self, s):
        if not self.enabled:
            return np.zeros(3)
        return np.tanh(s @ self.W)

    def update(self, s, s2, a, ext):
        if not self.enabled:
            return

        acc = (s2[3:] - s[3:]) / 0.05
        resid = acc - a - ext

        self.W += 5e-5 * np.outer(s/(np.linalg.norm(s)+1e-6), resid)
        self.W *= 0.995
        self.W = np.clip(self.W, -0.1, 0.1)


# =========================================================
# World Model
# =========================================================
class WM:
    def __init__(self, scm):
        self.scm = scm

    def step(self, s, a):
        pos = s[:3]
        vel = s[3:]

        acc = a + 0.1*self.scm.predict(s)
        vel = vel + acc*0.05
        pos = pos + vel*0.05

        return np.concatenate([pos, vel])


# =========================================================
# MPC
# =========================================================
class MPC:
    def __init__(self, wm, use_lyapunov=True, use_vel_penalty=True):
        self.wm = wm
        self.H = 8
        self.N = 80
        self.use_lyapunov = use_lyapunov
        self.use_vel_penalty = use_vel_penalty

    def rollout(self, s, U, target):
        x = s.copy()
        cost = 0

        for u in U:
            x = self.wm.step(x, u)

            e = x[:3] - target
            v = x[3:]

            V = np.sum(e**2)

            Vdot = np.sum(e*v)

            term = V

            if self.use_lyapunov:
                term += 0.6 * Vdot**2

            if self.use_vel_penalty:
                term += 0.2 * np.sum(v**2)

            cost += term

        return cost

    def act(self, s, target):
        mean = np.zeros((self.H,3))
        std = np.ones((self.H,3))

        for _ in range(3):
            samples = []

            for _ in range(self.N):
                U = mean + std*np.random.randn(self.H,3)
                U = np.clip(U, -3, 3)

                c = self.rollout(s, U, target)
                smooth = np.sum((U[1:] - U[:-1])**2)

                samples.append((c + 0.1*smooth, U))

            samples.sort(key=lambda x:x[0])
            elite = np.array([x[1] for x in samples[:10]])

            mean = elite.mean(0)
            std = elite.std(0) + 1e-6

        return mean[0]


# =========================================================
# Controller variants
# =========================================================
class Controller:
    def __init__(self,
                 use_scm=True,
                 use_mpc=True,
                 use_lyapunov=True,
                 use_vel_penalty=True,
                 use_filter=True):

        self.scm = SCM(enabled=use_scm)
        self.wm = WM(self.scm)

        self.use_mpc = use_mpc
        self.use_filter = use_filter

        if use_mpc:
            self.mpc = MPC(self.wm, use_lyapunov, use_vel_penalty)

        self.prev = np.zeros(3)

    def act(self, s, target):

        pos = s[:3]
        vel = s[3:]

        # 基础稳定器
        u = 1.5*(target-pos) - 0.8*vel

        if self.use_mpc:
            u += 0.2*self.mpc.act(s, target)

        if self.use_filter:
            u = 0.8*self.prev + 0.2*u
            self.prev = u

        return u

    def update(self, s, s2, a, ext):
        self.scm.update(s, s2, a, ext)


# =========================================================
# RUN
# =========================================================
def run(ctrl):
    env = Env()
    s = env.reset()
    target = np.array([10,10,5])

    err = []

    for i in range(1500):
        t = i*0.05

        a = ctrl.act(s, target)
        s2, ext = env.step(a, t)

        ctrl.update(s, s2, a, ext)
        s = s2

        err.append(np.linalg.norm(s[:3] - target))

    return np.array(err)


# =========================================================
# Ablation experiments
# =========================================================
experiments = {
    "Full": dict(),
    "No_SCM": dict(use_scm=False),
    "No_Lyapunov": dict(use_lyapunov=False),
    "No_VelPenalty": dict(use_vel_penalty=False),
    "No_MPC": dict(use_mpc=False),
    "No_Filter": dict(use_filter=False)
}

results = {}

for name, cfg in experiments.items():
    ctrl = Controller(**cfg)
    results[name] = run(ctrl)
    print(f"{name} done")

# =========================================================
# Plot
# =========================================================
fig = plt.figure(figsize=(3.3, 2.5))
plt.figure()
for name, e in results.items():
    plt.plot(e, label=name)

plt.legend()
plt.title("Ablation Study - Error")
plt.xlabel("Time step")
plt.ylabel("Error")
plt.savefig('ablation_error.jpeg', dpi=650, bbox_inches='tight')

# =========================================================
# Metrics
# =========================================================
print("\n===== Ablation Results =====")
print("| Method | Final | Mean |")
print("|--------|-------|------|")

for name, e in results.items():
    print(f"| {name} | {e[-1]:.3f} | {e.mean():.3f} |")