# This code has been developed by the authors of the paper " Low-rank variance reduction for uncertain radiative transfer with control variates"
# The code is used to reproduce the results of the paper. The code is written in Julia and uses the following packages:
# PyPlot, DelimitedFiles, BenchmarkTools, LaTeXStrings, Distributions, ProgressBars, JLD2, LinearAlgebra, Random, SparseArrays, LinearAlgebra, ProgressBars
#
include("settings.jl")
include("solver.jl")
include("uq.jl")

using PyPlot
using DelimitedFiles
using BenchmarkTools
using LaTeXStrings
using Distributions
using ProgressBars
using JLD2

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 30
rcParams["lines.markersize"] = 10
close("all")

#### Low-rank MC simulation ####
println("Starting computations for low-rank Monte Carlo")

## Define the parameters for the low-rank MC simulation
samples = [400,1600,6400,25600] # Number of samples used for MC 
ranks = [2,5,10,15,20,25,30,35,40] # ranks used for the low-rank approximation
Gridsize = [101,201,401,801] # Grid sizes used for the simulation

samples_plot,ranks_plot = meshgrid(samples, ranks);
  
d = Uniform(0.50,1.5); # The distribution of the random variable
seed = 123;
uncertParam = "1";
FoI = "ScalarFlux";

MC_error_frBUG = zeros(length(Gridsize),length(ranks),length(samples));
bias_frBUG = zeros(length(Gridsize),length(ranks),length(samples));
s_ref = settings(1601,101);
solver_ref = solver(s_ref);
####### To recompute the reference solution uncoment the next 2 lines #######
# mean_ref = Compute_full_samples_for_bias(solver_ref,102400,seed,uncertParam,FoI,d); # Compute the reference solution WARNING!! This computation can take upto 5 days
# save("Results/mean_ref_fine.jld2", "samples", 102400, "mean", mean_ref);

ref = load("Results/mean_ref_fine.jld2"); # Load the reference solution
mean_ref = ref["mean"];

for i = eachindex(Gridsize)
    global d,uncertParam,FoI,MC_error_frBUG,bias_frBUG;
    # local s, solver;
    Nx = Gridsize[i]; # Grid size
    s = settings(Nx,101); # Settings for the simulation
    solver_lr = solver(s); # Solver for the simulation
    c2f_idx = findall(x->x ∈ s.x ,s_ref.x); # Find the indices of the reference grid in the current grid
    for j = eachindex(ranks)
        r = ranks[j]; # Rank of the low-rank approximation
        solver_lr.settings.r = r; # Set the rank of the solver
        for k = eachindex(samples)
            println(i,", ",j,", ",k); 
            N_samples = samples[k]; # Number of samples used for the MC simulation
            MC_frBUG = MC(solver_lr,N_samples,seed,"fixedAugBUG",uncertParam,mean_ref[c2f_idx],d;compute_bias= true, FoI); # Compute the low-rank MC solution
            MC_error_frBUG[i,j,k] = sqrt(MC_frBUG["var"])/sqrt(N_samples); # Compute the MC error
            bias_frBUG[i,j,k] = MC_frBUG["bias"] # Compute the bias
        end
    end
    save("Results/Sol_MC_grid_$Nx.jld2", "MC_error", MC_error_frBUG[i,:,:], "bias", bias_frBUG[i,:,:]) # Save the intermediate results
end

save("Results/Sol_MC_full_run.jld2","samples", samples, "ranks", ranks, "Gridsize", Gridsize, "MC_error", MC_error_frBUG, "bias", bias_frBUG) # Save the final results


# Figure 1a-1b:
s = settings(101,101);
Solver = solver(s);
Random.seed!(123);
fixed_sample = rand(Uniform(0.5,1.5))
solver.alpha = fixed_sample;
println(fixed_sample)

g_full_low = solveFullProblem(Solver);
Solver.settings.r = 2;
g_AugBUG_low2 = solvefixedAugBUG(Solver);
Solver.settings.r = 5;
g_AugBUG_low5 = solvefixedAugBUG(Solver);
Solver.settings.r = 20;
g_AugBUG_low20 = solvefixedAugBUG(Solver);
Solver.settings.r = 30;
g_AugBUG_low30 = solvefixedAugBUG(Solver);

fig,ax = plt.subplots(figsize=(10,10),dpi=100);
ax.plot(s.x, g_full_low[:,1], label="Full", linewidth=5, color="blue",alpha=0.8)
ax.plot(s.x, g_AugBUG_low2[:,1],"--", label=L"r = 2", linewidth=5, color="green", alpha=0.4)
ax.plot(s.x, g_AugBUG_low5[:,1],":", label=L"r = 5", linewidth=5, color="purple", alpha=0.6)
# ax.plot(s.x, g_AugBUG_low20[:,1], label=L"r = 20", linewidth=2, color="purple")
ax.plot(s.x, g_AugBUG_low30[:,1],"-.", label=L"r = 30", linewidth=5, color="red",alpha=0.8)
ax.set_xlabel("x");
ax.set_ylabel("Scalar flux");
ax.set_ylim([0.0,1.0]);
ax.legend(handlelength=2.5, fontsize=30, loc="upper right");
fig.tight_layout();
plt.savefig("Results/BUGvsFull_low.pdf",bbox_inches="tight");


s = settings(401,101);
Solver = solver(s);
Random.seed!(123);
Solver.alpha = fixed_sample;

g_full_high = solveFullProblem(Solver);
Solver.settings.r = 2;
g_AugBUG_high2 = solvefixedAugBUG(Solver);
Solver.settings.r = 5;
g_AugBUG_high5 = solvefixedAugBUG(Solver);
Solver.settings.r = 20;
g_AugBUG_high20 = solvefixedAugBUG(Solver);
Solver.settings.r = 30;
g_AugBUG_high30 = solvefixedAugBUG(Solver);

fig,ax = plt.subplots(figsize=(10,10),dpi=100);
ax.plot(s.x, g_full_high[:,1], label="Full", linewidth=5, color="blue",alpha=0.8)
ax.plot(s.x, g_AugBUG_high2[:,1],"--", label=L"r = 2", linewidth=5, color="green", alpha=0.4)
ax.plot(s.x, g_AugBUG_high5[:,1],":", label=L"r = 5", linewidth=5, color="purple", alpha=0.6)
ax.plot(s.x, g_AugBUG_high30[:,1],"-.", label=L"r = 30", linewidth=5, color="red",alpha=0.8)
ax.set_xlabel("x");
ax.set_ylabel("Scalar flux");
ax.set_ylim([0.0,1.0]);
ax.legend(handlelength=2.5,fontsize=30, loc="upper right");
fig.tight_layout();
plt.savefig("Results/BUGvsFull_high.pdf",bbox_inches="tight");

# Fig 1c
fig,ax = plt.subplots(figsize=(10,10),dpi = 100);
ax.plot(ranks,bias_frBUG[1,:,end], "-o", label = L"m = 101", linewidth=5,markersize=20,color="green", alpha=0.6);
ax.plot(ranks,bias_frBUG[2,:,end], "-v", label = L"m = 201", linewidth=5,markersize=20, color="purple", alpha=0.6);
ax.plot(ranks,bias_frBUG[3,:,end], "-s", label = L"m = 401", linewidth=5,markersize=20,color="blue", alpha = 0.8);
ax.set_xlabel("ranks");
ax.set_ylabel("bias");
ax.set_yscale("log");
fig.tight_layout();
ax.legend();
plt.savefig("Results/BUGvsFull_rank_bias.pdf",bbox_inches="tight");

## Figure 2a-2c
close("all")
fig,axs = plt.subplots(2,length(Gridsize),figsize=(10*(length(Gridsize)),20),dpi=100) # Create the figure
MC_error_frBUG_max = maximum(MC_error_frBUG[:,:,:]); # Maximum MC error
bias_frBUG_max = maximum(bias_frBUG[:,:,:]); # Maximum bias
MC_error_frBUG_min = minimum(MC_error_frBUG[:,:,:]); # Minimum MC error
bias_frBUG_min = minimum(bias_frBUG[:,:,:]); # Minimum bias

for i = eachindex(Gridsize)
    Nx = Gridsize[i] # Grid size
    local im1,im2,color_bar;
    fig,ax = plt.subplots(1,1,figsize=(10,10),dpi=100) # Create the figure
    im1 = ax.pcolormesh(samples_plot,ranks_plot,Transpose(MC_error_frBUG[i,:,:]),shading = "auto",cmap = "plasma",norm=matplotlib.colors.LogNorm( MC_error_frBUG_min,MC_error_frBUG_max)) # Plot the MC error
    color_bar = fig.colorbar(im1, ax=ax, pad=0.03, shrink = 0.71) # Add the color bar
    
    ax.tick_params("both",labelsize=30) 
    ax.set_xlabel("samples", fontsize=30)
    ax.set_xlim([samples[1],samples[end]]);
    ax.set_xscale("log")
    ax.set_ylabel("ranks", fontsize=30)
    plt.savefig("Results/BUGvsFull_rank_samples_MCerror_$Nx.pdf",bbox_inches="tight")
end



#### Low-rank control variates ####

println("Starting computations for low-rank control variates")

s = settings(201,101);
solver_full = solver(s);
time_MCFull = @elapsed begin
    MC_full = MC(solver_full,2000,123,"full","1",Uniform(0.5,1.5);compute_bias= false, FoI="ScalarFlux");
end
var_MCFull = MC_full["MC_error"]
println("Full MC done! The computations took $time_MCFull seconds, variance is $var_MCFull")
save("Results/Results_MCFull.jld2", "mean",MC_full["mean"],"var", MC_full["var"], "time",  time_MCFull, "MCerror", MC_full["MC_error"])

s = settings(201, 101);
solver_lr = solver(s);
ranksCV = [2 5 10 15]';
ranksMC = [20 25 30 35 40]';
numRuns = 20 #for averaging/smoothing out runtime variations
diffCV_MC = zeros(length(ranksCV),length(ranksMC))
diffCV_MC_opt = zeros(length(ranksCV),length(ranksMC))
diffMC_MCfull = zeros(length(ranksMC))
var_CV = zeros(length(ranksCV),length(ranksMC))
var_CV_opt = zeros(length(ranksCV),length(ranksMC))
var_MC = zeros(length(ranksMC))
time_CV = zeros(length(ranksCV),length(ranksMC),numRuns)
time_CV_opt = zeros(length(ranksCV),length(ranksMC),numRuns)
time_MC = zeros(length(ranksMC),numRuns)
optAlpha = zeros(length(ranksCV),length(ranksMC),numRuns)
N_samplesCV = zeros(length(ranksCV),length(ranksMC),numRuns)
N_samples_opt = 0;
thetaOpt = 1;
N_samples_diff = 0;
for n=1:numRuns
    println("Starting run $n")
    for p=eachindex(ranksMC)
            s = settings(201,101);
            s.r = ranksMC[p];
            solver_lr = solver(s);
            time_MC[p,n] = @elapsed begin
            MC_fixedAugBUG = MC(solver_lr,2000,123,"fixedAugBUG","1",Uniform(0.5,1.5);compute_bias= false, FoI="ScalarFlux");
            end
            var_MC[p] = MC_fixedAugBUG["MC_error"]
            diffMC_MCfull[p] = norm(MC_fixedAugBUG["mean"].-MC_full["mean"])
            println("MC done! The computations took $(time_MC[p]) seconds, variance is $(var_MC[p])")
        for r=eachindex(ranksCV)
            N_samples_opt, thetaOpt = getOptParameters(solver_lr,var_MC[p], ranksMC[p],ranksCV[r],Uniform(0.5,1.5),500,"fixedAugBUG";FoI="ScalarFlux");
            time_CV[r,p,n] = @elapsed begin
                CV_fixedAugBUG_tmp = ControlVariate(solver_lr,2000,200,0,123,"1","fixedAugBUG",ranksMC[p],ranksCV[r],1.0,Uniform(0.5,1.5),var_MC[p],true;FoI="ScalarFlux");
            end
            diffCV_MC[r,p] += norm(CV_fixedAugBUG_tmp["mean"].-MC_full["mean"])./numRuns
            var_CV[r,p] += CV_fixedAugBUG_tmp["var"]./numRuns
            time_CV_opt[r,p,n] = @elapsed begin
                CV_fixedAugBUG_tmp = ControlVariate(solver_lr,2000,N_samples_diff,maximum([N_samples_opt 5]),123,"1","fixedAugBUG",ranksMC[p],ranksCV[r],minimum([thetaOpt 1]),Uniform(0.5,1.5),var_MC[p],false;FoI="ScalarFlux");
            end
            diffCV_MC_opt[r,p] += norm(CV_fixedAugBUG_tmp["mean"].-MC_full["mean"])./numRuns
            var_CV_opt[r,p] += CV_fixedAugBUG_tmp["var"]./numRuns
            optAlpha[r,p,n] = CV_fixedAugBUG_tmp["alpha"]
            N_samplesCV[r,p,n] = CV_fixedAugBUG_tmp["N_samples_diff"]
            println("Control variates done! The computations took $(time_CV[r,p,n]) seconds, variance is $(var_CV[r,p])")
        end
    end
end
#save results in jld2 file
save("Results/Results_MC.jld2", "var", var_MC, "time",  time_MC, "diffMC", diffMC_MCfull, "ranks",ranksMC)
save("Results/Results_CV.jld2", "var", var_CV, "time",  time_CV, "diffMC", diffCV_MC, "ranksFine",ranksMC, "ranksCoarse",ranksCV)
save("Results/Results_CVopt.jld2", "var", var_CV_opt, "time",  time_CV_opt, "diffMC", diffCV_MC_opt, "alpha", optAlpha,"ranksFine",ranksMC, "ranksCoarse",ranksCV)

time_MCMin = minimum(time_MC, dims =2)
time_CVMin = minimum(time_CV, dims=3)
time_CV_optMin = minimum(time_CV_opt, dims=3)

PyPlot.close_figs()
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 12
rcParams["figure.autolayout"] = true
ticker = pyimport("matplotlib.ticker")


## Plotting the results
symbsl = ["-o","-x","-*","-+","-8","-v","-D"]
symbs = ["o","v","s","X","*","+","8","D","p"];
prop_cycle = plt.rcParams["axes.prop_cycle"]
mycolors = prop_cycle.by_key()["color"];

fig = figure(dpi=100)
ax=gca()
ax.loglog(ranksMC[1:end],time_MCMin[1:end],symbs[1], label = "MC",markersize=10)
for i = 1:length(ranksCV)
    ax.loglog(ranksMC[1:end],time_CVMin[i,1:end],symbs[i+1], label = string("CV, s = ",ranksCV[i]),markersize=10)
end
ax.set_xlabel("rank ", fontsize=12)
ax.set_ylabel("time",fontsize=12)
ax.legend(loc="upper left", fontsize=12,ncol=1)
ax[:xaxis][:set_major_formatter](ticker[:ScalarFormatter]())
ax[:xaxis][:set_minor_formatter](ticker[:ScalarFormatter]())
ax[:yaxis][:set_major_formatter](ticker[:ScalarFormatter]())
ax.set_xticks([20, 25, 30, 35, 40])
ax.set_yticks([])
ax.set_yticks([100,200,300, 400, 600, 800,1000])
savefig("Results/timeCVMC_minLog.pdf");

fig = figure(dpi=100)
ax=gca()
ax.loglog(ranksMC[1:end],time_MCMin[1:end],symbs[1], label = "MC",markersize=10)
for i = 1:length(ranksCV)
    ax.loglog(ranksMC[1:end],time_CV_optMin[i,1:end],symbs[i+1], label = string("CV, s = ",ranksCV[i]),markersize=10)
end
ax.set_xlabel("rank ", fontsize=12)
ax.set_ylabel("time",fontsize=12)
ax.legend(loc="upper left", fontsize=12,ncol=1)#,frameon=false)
ax.tick_params(axis="both", labelsize=12)
ax[:xaxis][:set_major_formatter](ticker[:ScalarFormatter]())
ax[:xaxis][:set_minor_formatter](ticker[:ScalarFormatter]())
ax[:yaxis][:set_major_formatter](ticker[:ScalarFormatter]())
ax.set_xticks([20, 25, 30, 35, 40])
ax.set_yticks([])
ax.set_yticks([100,200,300, 400, 600, 800,1000])
tight_layout()
savefig("Results/timeCVMC_minOptLog.pdf")
