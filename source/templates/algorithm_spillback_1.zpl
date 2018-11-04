# Define sets
set V := {read "nodes.txt" as "<1s>" comment "#"};
set V_st := {"s", "t"};
set V_inner := V without V_st;
set V_without_t := V without {"t"};
set E := {read "edges.txt" as "<1s, 2s>" comment "#"};


# Load parameters
param InE_0[E] := read "edges.txt" as "<1s, 2s> 5n" comment "#";
param InE_star[E] := read "edges.txt" as "<1s, 2s> 6n" comment "#";
set E_0 := {<v,w> in E with InE_0[v,w] == 1};
set E_star := {<v,w> in E with InE_star[v,w] == 1};


param nu[E] := read "edges.txt" as "<1s, 2s> 3n" comment "#";
param b[E] := read "edges.txt" as "<1s, 2s> 4n" comment "#";
param r := read "other.txt" as "1n" use 1 comment "#";
param M := read "other.txt" as "1n" skip 1 comment "#";


# Useful sets
defset din(v) := {<i,v> in E};      #d^-(v)
defset dout(v) := {<v,i> in E};     #d^+(v)

# VARIABLES
var l[V] real >= 0 <= M;
var x[E] real >= 0 <= r;
var c[V] real >= 0 <= 1;
var z[V] binary;
var m[E] binary;
var omega[E without E_star] binary;


# OBJECTIVE
maximize labels: sum <v> in V: c[v];

# Direct variable constraints
subto maxOnlyIfThrottle: forall <v,w> in E do m[v,w] <= 1 - z[v];
subto cAndzRelation: forall <v> in V do c[v] >= z[v];
subto atLeastOneMax: forall <v> in V do
                        sum <v,w> in dout(v): m[v,w] >= 1 - z[v];

subto lInit: l['s'] == 1; # TF1
subto cInits: c['s'] == 1;
subto cInitt: c['t'] == 1; #Last node has no outgoing edge, i.e. cant be throttled
                            

# CONSTRAINTS
# Normal Flow Constraints
subto restrictedFlow: forall <v,w> in (E without E_0) do x[v,w] == 0; # Flow only on E_0
# Flow conservation
subto conservation_v: forall <v> in V_inner do
                        sum <i,v> in din(v): x[i,v] - sum <v,i> in dout(v): x[v,i] == 0;
subto conservation_s: sum <i,s> in din('s'): x[i,s] - sum <s,i> in dout('s'): x[s,i] == -1 * r;
subto conservation_t: sum <i,t> in din('t'): x[i,t] - sum <t,i> in dout('t'): x[t,i] == r;

# TF2
subto resettingMinimum: forall <u,v> in E_star without E_0 do c[v]*l[v]*nu[u,v] <= x[u,v];
subto noResettingMinimum_1: forall <u,v> in E without E_star do l[v] <= l[u] + M*(1-omega[u,v]);
subto noResettingMinimum_2: forall <u,v> in E without E_star do c[v]*l[v]*nu[u,v] <= x[u,v] + nu[u,v]*M*omega[u,v];

# TF3
subto resettingActiveMinimum: forall <u,v> in E_star inter E_0 do c[v]*l[v]*nu[u,v] == x[u,v];
subto noResettingActiveMinimum_1: forall <u,v> in E_0 without E_star do l[v] >= l[u];
subto noResettingActiveMinimum_2: forall <u,v> in E_0 without E_star do c[v]*l[v]*nu[u,v] >= x[u,v];

# TF4
subto allEdgeMax: forall <v,w> in E do l[v]*b[v,w] >= x[v,w];

# TF5
subto edgeMax: forall <v,w> in E do l[v] * b[v,w] <= x[v,w] + b[v,w]*M*(1-m[v,w]);

