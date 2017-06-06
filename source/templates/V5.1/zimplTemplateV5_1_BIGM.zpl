#needed sets
set V := {read "nodes.txt" as "<1s>" comment "#"};
set V_st := {"s", "t"};
set V_inner := V without V_st;
set E := {read "edges.txt" as "<1s, 2s>" comment "#"};

param InE_star[E] := read "edges.txt" as "<1s, 2s> 5n" comment "#";
set E_star := {<v,w> in E with InE_star[v,w] == 1};


#needed params
param u[E] := read "edges.txt" as "<1s, 2s> 3n" comment "#";
param u_0 := read "other.txt" as "1n" use 1 comment "#";
param B := read "other.txt" as "1n" skip 1 comment "#";

#useful sets
defset din(v) := {<i,v> in E};      #d^-(v)
defset dout(v) := {<v,i> in E};     #d^+(v)

#VARIABLES
var l[V] real >= 0 <= B;
var x[E] real >= 0 <= u_0;
var a[E] binary;
var z[E without E_star] binary;

#OBJECTIVE
maximize labels: sum <v> in V: l[v];


#CONSTRAINTS
#1: Normal Flow Constraints
subto restrictedFlow: forall <v,w> in E do x[v,w] <= a[v,w]*u_0; #flow only on E_0
#flow conservation
subto conservation_v: forall <v> in V_inner do
                        sum <i,v> in din(v): x[i,v] - sum <v,i> in dout(v): x[v,i] == 0;
subto conservation_s: sum <i,s> in din('s'): x[i,s] - sum <s,i> in dout('s'): x[s,i] == -1 * u_0;
subto conservation_t: sum <i,t> in din('t'): x[i,t] - sum <t,i> in dout('t'): x[t,i] == u_0;

subto lInit: l['s'] == 1;
subto Resetting: forall <v,w> in E_star do
                        u[v,w] * l[w] - x[v,w] == 0;

subto noFlowNoResetting: forall <v,w> in (E without E_star) do
                        l[w] <= l[v] + a[v,w]*B;

subto FlowNoResetting: forall <v,w> in E without E_star do
                        l[w] >= l[v] + (a[v,w] - 1)*B and
                        u[v,w]*l[w] >= x[v,w] + (a[v,w]-1)*B*u[v,w] and
                        l[w] <= l[v] + (1-z[v,w] + 1-a[v,w])*2*B and
                        u[v,w]*l[w] <= x[v,w] + (z[v,w] + 1-a[v,w])*2*B*u[v,w];
                        
subto E_0_chosen_possible_inner: forall <w> in (V_inner) do
                            sum <i,w> in din(w): a[i,w] >= 1;
                            
subto E_0_chosen_possible_t: sum <i,t> in din('t'): a[i,t] >= 1;


