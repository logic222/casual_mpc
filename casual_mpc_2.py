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
        if 10 <= t < 15:
            ext = np.array([5, 0, -3])

        if 20 <= t < 25:
            ext += 0.8 * np.array([
                np.sin(2 * t),
                np.cos(1.6 * t),
                np.sin(0.8 * t)
            ])

        occlusion_noise = np.zeros(6)

        if 30 < t < 40:
            occlusion_noise = np.random.rand(6) * 0.8
        else:
            occlusion_noise = np.random.rand(6) * 0.02

        pos = self.state[:3]
        vel = self.state[3:]

        acc = a + ext
        # 去除SCM
        # acc=a
        vel = vel + acc * self.dt
        pos = pos + vel * self.dt

        self.state = np.concatenate([pos, vel]) + occlusion_noise
        return self.state.copy(), ext


# =========================================================
# SCM
# =========================================================
class SCM:
    def __init__(self):
        self.W = np.zeros((6, 3))
        self.W_history = []
        self.residual_history = []
        self.intervention_history = []

    def predict(self, s):
        return np.tanh(s @ self.W)

    def update(self, s, s2, a, ext, dt=0.05):
        acc = (s2[3:] - s[3:]) / 0.05
        resid = acc - a - ext

        lr = 5e-5
        s_norm = s / (np.linalg.norm(s) + 1e-6)

        self.W += lr * np.outer(s_norm, resid)
        self.W *= 0.997
        self.W = np.clip(self.W, -0.1, 0.1)

        self.W_history.append(self.W.copy())
        self.residual_history.append(np.linalg.norm(resid))
        self.intervention_history.append(np.linalg.norm(ext))


# =========================================================
# World Model
# =========================================================
class WM:
    def __init__(self, scm):
        self.scm = scm

    def step(self, s, a):
        pos = s[:3]
        vel = s[3:]

        acc = a + 0.1 * self.scm.predict(s)
        vel = vel + acc * 0.05
        pos = pos + vel * 0.05

        return np.concatenate([pos, vel])


# =========================================================
# MPC (fixed)
# =========================================================
class MPC:
    def __init__(self, wm):
        self.wm = wm
        self.H = 8
        self.N = 80

    def rollout(self, s, U, target):
        x = s.copy()
        cost = 0

        for t, u in enumerate(U):
            x = self.wm.step(x, u)

            e = x[:3] - target
            v = x[3:]


            V = np.sum(e ** 2)

            # strict decay constraint
            Vdot = np.sum(e * v)

            cost += V + 0.6 * Vdot ** 2 + 0.2 * np.sum(v ** 2)

        return cost

    def act(self, s, target):
        mean = np.zeros((self.H, 3))
        std = np.ones((self.H, 3))

        for _ in range(3):
            samples = []

            for _ in range(self.N):
                U = mean + std * np.random.randn(self.H, 3)
                U = np.clip(U, -3, 3)

                c = self.rollout(s, U, target)
                smooth = np.sum((U[1:] - U[:-1]) ** 2)

                samples.append((c + 0.1 * smooth, U))

            samples.sort(key=lambda x: x[0])
            elite = np.array([x[1] for x in samples[:10]])

            mean = elite.mean(0)
            std = elite.std(0) + 1e-6

        return mean[0]


# =========================================================
# Controller
# =========================================================
class Ours:
    def __init__(self):
        self.scm = SCM()
        self.wm = WM(self.scm)
        self.mpc = MPC(self.wm)

        self.prev = np.zeros(3)

    def act(self, s, target):
        pos = s[:3]
        vel = s[3:]

        #exponential stabilizer
        u = 1.5 * (target - pos) - 0.8 * vel

        u_mpc = self.mpc.act(s, target)

        u = u + 0.2 * u_mpc

        #strong damping (关键修复振荡)
        u = 0.8 * self.prev + 0.2 * u
        self.prev = u

        return u

    def update(self, s, s2, a, ext):
        self.scm.update(s, s2, a, ext)


# =========================================================
# PID
# =========================================================
class PID:
    def act(self, s, t):
        return 0.8 * (t - s[:3]) - 0.2 * s[3:]

#MPC
class MPC_Only:
    def __init__(self):
        # ❌ 不使用 SCM
        self.scm = SCM()
        self.scm.predict = lambda s: np.zeros(3)

        self.wm = WM(self.scm)
        self.mpc = MPC(self.wm)

        self.prev = np.zeros(3)

    def act(self, s, target):

        # 完全依赖 MPC
        u = self.mpc.act(s, target)

        # 加一个最小滤波（否则会炸）
        u = 0.7*self.prev + 0.3*u
        self.prev = u

        return u

    def update(self, s, s2, a, ext):
        pass  # ❌ 不学习

# =========================================================
# RUN
# =========================================================
def run(ctrl):
    env = Env()
    s = env.reset()
    target = np.array([10, 10, 5])

    traj = []
    err = []

    for i in range(1500):
        t = i * 0.05

        a = ctrl.act(s, target)
        s2, ext = env.step(a, t)

        if hasattr(ctrl, "update"):
            ctrl.update(s, s2, a, ext)

        s = s2

        traj.append(s[:3])
        err.append(np.linalg.norm(s[:3] - target))

    return np.array(err), np.array(traj)


def plot_scm_W(scm):
    W = np.array(scm.W_history)

    plt.figure()
    plt.plot(np.linalg.norm(W.reshape(len(W), -1), axis=1))
    plt.title("SCM Parameter Norm Evolution")
    plt.xlabel("t")
    plt.ylabel("|W|")
    plt.savefig("scm_W.png")


def plot_residual(scm):
    r = np.array(scm.residual_history)

    plt.figure()
    plt.plot(r)
    plt.title("SCM Residual Convergence")
    plt.xlabel("t")
    plt.ylabel("||residual||")
    plt.savefig("scm_residual.png")


def plot_intervention(scm):
    ext = np.array(scm.intervention_history)

    plt.figure()
    plt.plot(ext)
    plt.title("External Intervention Strength")
    plt.xlabel("t")
    plt.ylabel("||ext||")
    plt.savefig("scm_intervention.png")


ours = Ours()
pid = PID()
mpc_only = MPC_Only()

e1, t1 = run(ours)
e2, t2 = run(pid)
e3, t3 = run(mpc_only)

fig = plt.figure(figsize=(3.3, 2.5))
plt.figure()
plt.plot(e1, label="Full (Ours)")
plt.plot(e2, label="PID")
plt.plot(e3, label="MPC Only")
plt.legend()
plt.title("Error Comparison")
plt.savefig('error-123.jpeg', dpi=650, bbox_inches='tight')

plt.figure()
plt.plot(t1[:,0], label="Full-x")
plt.plot(t1[:,1], label="Full-y")
plt.plot(t1[:,2], label="Full-z")
plt.legend()
plt.title("Trajectory Full")
plt.savefig('traj_full.jpeg', dpi=650, bbox_inches='tight')
plt.close()

plt.figure()
plt.plot(t2[:,0], label="PID-x")
plt.plot(t2[:,1], label="PID-y")
plt.plot(t2[:,2], label="PID-z")
plt.legend()
plt.title("Trajectory PID")
plt.savefig('traj_pid.jpeg', dpi=650, bbox_inches='tight')
plt.close()

plt.figure()
plt.plot(t3[:,0], label="MPC-x")
plt.plot(t3[:,1], label="MPC-y")
plt.plot(t3[:,2], label="MPC-z")
plt.legend()
plt.title("Trajectory MPC")
plt.savefig('traj_mpc.jpeg', dpi=650, bbox_inches='tight')
plt.close()

print("DONE")

print("\n===== FINAL RESULTS =====")
print("| Method | Final Error | Mean Error |")
print("|--------|------------|------------|")

def report(name, e):
    print(f"| {name} | {e[-1]:.3f} | {e.mean():.3f} |")

report("Full", e1)
report("PID", e2)
report("MPC Only", e3)