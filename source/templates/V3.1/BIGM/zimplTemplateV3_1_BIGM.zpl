#needed sets
set V := {read "nodes.txt" as "<1s>" comment "#"};
set V_st := {"s", "t"};
set V_inner := V without V_st;
set E := {read "edges.txt" as "<1s, 2s>" comment "#"};

#do print V_st;
#do print V_inner;

param InE_0[E] := read "edges.txt" as "<1s, 2s> 4n" comment "#";
param InE_star[E] := read "edges.txt" as "<1s, 2s> 5n" comment "#";
set E_0 := {<v,w> in E with InE_0[v,w] == 1};
set E_star := {<v,w> in E with InE_star[v,w] == 1};


#needed params
param u[E] := read "edges.txt" as "<1s, 2s> 3n" comment "#";
param u_0 := read "other.txt" as "1n" use 1 comment "#";
param B := read "other.txt" as "1n" skip 1 comment "#";
#do print u_0;
#do print B;

#useful sets
defset din(v) := {<i,v> in E};      #d^-(v)
defset dout(v) := {<v,i> in E};     #d^+(v)

#VARIABLES
var l[V] real >= 0 <= B;
var x[E_0] real >= 0;
var spos[E_0 without E_star] >= 0;
var sneg[E_0 without E_star] >= 0;

#OBJECTIVE
minimize labels: sum <v,w> in (E_0 without E_star): (l[w]);


#CONSTRAINTS
#1: Normal Flow Constraints
subto restrictedFlow: forall <v,w> in (E without E_0) do x[v,w] == 0; #flow only on E_0
#flow conservation
subto conservation_v: forall <v> in V_inner do
                        sum <i,v> in din(v): x[i,v] - sum <v,i> in dout(v): x[v,i] == 0;
subto conservation_s: sum <i,s> in din('s'): x[i,s] - sum <s,i> in dout('s'): x[s,i] == -1 * u_0;
subto conservation_t: sum <i,t> in din('t'): x[i,t] - sum <t,i> in dout('t'): x[t,i] == u_0;

#2: Constraint set 1
subto noFlowNoResetting: forall <v,w> in E without (E_0 union E_star) do
                        l[w] <= l[v];
subto noFlowButResetting: forall <v,w> in E_star without E_0 do
                        l[w] <= 0;
                        
#3: Constraint set 2
subto lInit: l['s'] == 1;
subto FlowButResetting: forall <v,w> in E_0 inter E_star do
                        u[v,w] * l[w] - x[v,w] == 0;
                        

subto FlowNoResetting: forall <v,w> in E_0 without E_star do
                        2*u[v,w]*l[w] == u[v,w]*l[v] + x[v,w] + u[v,w]*(spos[v,w] + sneg[v,w]) and
                        x[v,w] - u[v,w]*l[v] <= u[v,w]*spos[v,w] and
                        u[v,w]*l[v] - x[v,w] <= u[v,w]*sneg[v,w];
                        
                        
