clc 
clear all

% pwa dynamics
Ax = [-1, 1; -3, -1; 0.2, 1; -1, 0; 1, 0; 0, -1];
bx = [15; 25; 9; 6; 8; 10];

A1 = [1, 0.2; 0, 1];
B1 = [0.1; 1];
c1 = [0; 0];
sys1 = LTISystem('A', A1, 'B', B1, 'f', c1);
R1 = Polyhedron('A', [1, 0; Ax], 'b', [1; bx]);
sys1.setDomain('x', R1);

A2 = [0.5, 0.2; 0, 1];
B2 = [0.1; 1];
c2 = [0.5; 0];
sys2 = LTISystem('A', A2, 'B', B2, 'f', c2);
R2 = Polyhedron('A', [-1, 0; Ax], 'b', [-1; bx]);
sys2.setDomain('x', R2);

pwa_sys = PWASystem([sys1, sys2]);

% constraints
X = Polyhedron('A', Ax, 'b', bx);
pwa_sys.x.with('setConstraint');
pwa_sys.x.setConstraint = X;

u_lim = 3;
pwa_sys.u.min = -u_lim;
pwa_sys.u.max = u_lim;

% cost
pwa_sys.x.penalty = OneNormFunction(diag([1, 1]));
pwa_sys.u.penalty = OneNormFunction(1);

% terminal ingredients
sys1.u.min = -u_lim;
sys1.u.max = u_lim;
sys1.x.penalty = OneNormFunction(diag([1, 1]));
sys1.u.penalty = OneNormFunction(1);
Tset = sys1.LQRSet;
Tcost = sys1.LQRPenalty;
sys1.x.with('setConstraint');
sys1.x.setConstraint = R1;
sys1.x.with('terminalSet');
sys1.x.terminalSet = Tset;
sys1.x.with('terminalPenalty');
sys1.x.terminalPenalty = Tcost;

pwa_sys.x.with('terminalSet');
pwa_sys.x.terminalSet = Tset;
% pwa_sys.x.with('terminalPenalty');
% pwa_sys.x.terminalPenalty = Tcost;

times = [];
num_partitions = [];
for N = 9:12
    mpc = MPCController(pwa_sys, N);
    t0 = tic();
    exp_mpc = mpc.toExplicit();
    runtime = toc(t0);
    nr = exp_mpc.nr;
    num_partitions = [num_partitions, nr];
    times = [times, runtime];
    save("exp_mpc"+string(N)+".mat", "exp_mpc", "runtime", "nr")
end

