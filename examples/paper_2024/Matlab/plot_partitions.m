clc 
clear all

exp_mpcs = {};
for i=2:10
    o = load("exp_mpc"+string(i)+".mat");
    exp_mpcs{i-1} = o.exp_mpc;
    i
    o.nr
    o.runtime
end

tiledlayout(4, 3);
for i=1:11
    nexttile
    exp_mpcs{i}.partition.plot()
    title("N = "+string(i+1))
end