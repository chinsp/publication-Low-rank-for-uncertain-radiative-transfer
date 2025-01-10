__precompile__
using Random
using Parameters



function MC(obj::solver,N_samples::Int,seed::Int,numerSolver::String,uncertParam::String,mean_ref::Array,d;kwargs...)
    @unpack compute_bias,FoI = kwargs;
    
    samples = rand(d,N_samples);

    if numerSolver == "full"
        NumerSolver = solveFullProblem;
    elseif numerSolver == "frBUG"
        NumerSolver = solvefixedrankBUG;
    elseif numerSolver == "fixedAugBUG"
        NumerSolver = solvefixedAugBUG;
    end
    obj.uncertParam = uncertParam;

    mean = zeros(Float64,obj.settings.Nx);
    var = 0.0;
    bias = 0.0;

    g = zeros(Float64,obj.settings.Nx,obj.settings.nPN+1);

    for n = 1:N_samples
        alpha = samples[n];
        obj.alpha = alpha;
        g .= NumerSolver(obj);

        y = Sol_to_FoI(obj,g,FoI);

        mean .+=  y./N_samples;
        var += y'y;
    end

    var = (var - N_samples*mean'mean)*obj.settings.dx/(N_samples - 1);
    MC_error = sqrt(var)/sqrt(N_samples);
    bias = Transpose((mean_ref - mean))*(mean_ref - mean).*obj.settings.dx;

    output = Dict("N_samples" => N_samples, "mean" => mean, "var" => var, "MC_error" => MC_error,"bias" => bias);

    return output;
end


function MC(obj::solver,N_samples::Int,seed::Int,numerSolver::String,uncertParam::String,d;kwargs...)
    @unpack compute_bias,FoI = kwargs;
    
    samples = rand(d,N_samples);
    if compute_bias
        full_sol_samples = Compute_full_samples_for_bias(obj,N_samples,seed,uncertParam,FoI,d);
    else
        full_sol_samples = zeros(N_samples,obj.settings.Nx);
    end

    if numerSolver == "full"
        NumerSolver = solveFullProblem;
    elseif numerSolver == "frBUG"
        NumerSolver = solvefixedrankBUG;
    elseif numerSolver == "fixedAugBUG"
        NumerSolver = solvefixedAugBUG;
    end
    obj.uncertParam = uncertParam;

    mean = zeros(Float64,obj.settings.Nx);
    var = 0.0;

    Threads.@threads for n = 1:N_samples
        alpha = samples[n];
        obj.alpha = alpha;

        g = NumerSolver(obj);

        y = Sol_to_FoI(obj,g,FoI);

        mean .+=  y./N_samples;
        var += y'y;
    end

    var = (var - N_samples*mean'mean)*obj.settings.dx/(N_samples - 1);
    MC_error = sqrt(var)/sqrt(N_samples);

    output = Dict("N_samples" => N_samples, "mean" => mean, "var" => var, "MC_error" => MC_error);

    return output;
end

function MC(obj::solver,N_samples::Int,numerSolver::String,d;kwargs...)
    @unpack compute_bias,FoI = kwargs;
    seed = 123;
    uncertParam = "1";
    return MC(obj,N_samples,seed,numerSolver,uncertParam,d;compute_bias,FoI)
end

function MC(obj::solver,N_samples::Int,numerSolver::String,full_sol_samples::Array,d;kwargs...)
    @unpack compute_bias,FoI = kwargs;
    seed = 123;
    uncertParam = "1";
    return MC(obj,N_samples,seed,numerSolver,uncertParam,full_sol_samples,d;compute_bias,FoI)
end

function Compute_full_samples_for_bias(obj::solver, N_samples,seed,uncertParam,FoI,d)
    println("------------------------------------------------------------------")
    println("Compute bias flag was set to true")
    println("The reference solution is being computed for $N_samples samples")
    # Random.seed!(seed);
    samples = rand(d,N_samples);
    mean = zeros(Float64,obj.settings.Nx);
    g = zeros(obj.settings.Nx,obj.settings.nPN+1);
    obj.uncertParam = uncertParam;
    for n = ProgressBar(1:N_samples)
        alpha = samples[n];
        obj.alpha = alpha;
        g .= solveFullProblem(obj);
        mean .+= Sol_to_FoI(obj,g,FoI)./N_samples;
    end
    
    println("Reference solution computed")
    println("------------------------------------------------------------------")
    return mean;
end

function Sol_to_FoI(obj::solver,g,FoI::String) #Solution to funciton of interest mapping
    if FoI == "ScalarFlux"
        y = g[:,1];
    end
    return y;
end

function ControlVariate(obj::solver,N_samples_s::Int,N_samples_diff::Int,N_samples_opt::Int,seed::Int,uncertParam::String,numerSolver::String,r::Int,s::Int,theta::Float64,d,epsilon::Float64,optTheta::Bool=false;kwargs...)
    if Threads.nthreads() > 1
        println("Warning: Incremental (co)variance computation might be inaccurate on several threads!")
    end
    @unpack FoI = kwargs

    if r <= s
        println("s must be smaller than r")
        return
    end

    if numerSolver == "full"
        NumerSolver = solveFullProblem;
    elseif numerSolver == "frBUG"
        NumerSolver = solvefixedrankBUG;
    elseif numerSolver == "fixedAugBUG"
        NumerSolver = solvefixedAugBUG;
    end

    samples_diff = rand(d, N_samples_diff)
    samples_s = zeros(N_samples_s)
    samples_s[1:N_samples_diff] = samples_diff
    sol = zeros(Float64, obj.settings.Nx, N_samples_s)

    obj.uncertParam = uncertParam
    # Variables to track number of processed samples
    count_diff = 0
    count_s = 0
    # Initialize means, variances, and covariance 
    mean_s = zeros(Float64, obj.settings.Nx)
    mean_r = zeros(Float64, obj.settings.Nx)
    mean_diff = zeros(Float64,obj.settings.Nx)
    var_s = zeros(Float64, obj.settings.Nx)
    var_r = zeros(Float64, obj.settings.Nx)
    cov_rs = zeros(Float64, obj.settings.Nx)

    yr_n = zeros(N_samples_s, obj.settings.Nx)
    ys_n = zeros(N_samples_s, obj.settings.Nx)    

    if N_samples_opt ==0
        #compute theta_opt and number of samples using N_samples_diff warmup samples
        for n = 1:N_samples_diff
            alpha = samples_diff[n]
            obj.alpha = alpha
            obj.settings.r = r
            _, gr = NumerSolver(obj)

            obj.settings.r = s
            _, gs = NumerSolver(obj)


            yr = Sol_to_FoI(obj, gr, FoI)
            ys = Sol_to_FoI(obj, gs, FoI)

            yr_n[n,:] = yr
            ys_n[n,:] = ys
        end

        # Compute sample variances and covariance
        mean_r = Statistics.mean(yr_n[1:N_samples_diff,:],dims=1)
        mean_s = Statistics.mean(ys_n[1:N_samples_diff,:],dims=1)
        var_r_tmp  = zeros(Float64,obj.settings.Nx);
        var_s_tmp  = zeros(Float64,obj.settings.Nx);
        cov_rs_tmp  = zeros(Float64,obj.settings.Nx);

        factor = obj.settings.dx / (N_samples_diff - 1)

        # Compute deviations from the mean
        dev_r = yr_n[1:N_samples_diff,:] .- mean_r
        dev_s = ys_n[1:N_samples_diff,:] .- mean_s

        # Variance for r
        var_r_tmp = sum(dev_r.^2, dims=1) .* factor

        # Variance for s
        var_s_tmp = sum(dev_s.^2, dims=1) .* factor

        # Covariance between r and s
        cov_rs_tmp = sum(dev_r .* dev_s, dims=1) .* factor
        var_r_tmp = norm(var_r_tmp)
        var_s_tmp = norm(var_s_tmp)
        cov_rs_tmp = norm(cov_rs_tmp)
        rho_rs = cov_rs_tmp/(sqrt(var_r_tmp)*sqrt(var_s_tmp))

        # # Compute the optimal theta
        # theta = cov_rs_tmp/ var_s_tmp

        #compute number samples
        N_diffOpt = maximum([Int(ceil(var_r_tmp * (1-rho_rs^2)/epsilon^2)) 5])
    else
        N_diffOpt =N_samples_opt
    end
    println("Computing $N_diffOpt samples on the differences")

    if N_diffOpt > N_samples_diff
        samples_diff = rand(d, N_diffOpt-N_samples_diff)
        samples_s[N_samples_diff+1:N_diffOpt] = samples_diff
        samples_s[N_diffOpt+1:N_samples_s] = rand(d,N_samples_s-N_diffOpt)

        #compute rest of samples for difference 
        for n = 1:N_diffOpt-N_samples_diff
            alpha = samples_diff[n]
            obj.alpha = alpha
            obj.settings.r = r
            _, gr = NumerSolver(obj)

            obj.settings.r = s
            _, gs = NumerSolver(obj)

            yr = Sol_to_FoI(obj, gr, FoI)
            ys = Sol_to_FoI(obj, gs, FoI)

            yr_n[N_samples_diff+n,:] = yr
            ys_n[N_samples_diff+n,:] = ys
        end
    else
        samples_s[N_samples_diff+1:N_samples_s] = rand(d,N_samples_s-N_samples_diff)
    end

    mean_r = Statistics.mean(yr_n[1:N_diffOpt,:],dims=1)
    mean_s = Statistics.mean(ys_n[1:N_diffOpt,:],dims=1)

    var_r  = zeros(Float64,obj.settings.Nx);
    var_s_tmp  = zeros(Float64,obj.settings.Nx);
    cov_rs  = zeros(Float64,obj.settings.Nx);

    factor = obj.settings.dx / (N_diffOpt - 1)

    # Compute deviations from the mean
    dev_r = yr_n[1:N_diffOpt,:] .- mean_r
    dev_s = ys_n[1:N_diffOpt,:] .- mean_s

    # Variance for r
    var_r = sum(dev_r.^2, dims=1) .* factor

    # Variance for s
    var_s_tmp = sum(dev_s.^2, dims=1) .* factor

    # Covariance between r and s
    cov_rs = sum(dev_r .* dev_s, dims=1) .* factor
    var_r = norm(var_r)
    var_s_tmp = norm(var_s_tmp)
    cov_rs = norm(cov_rs)

    if optTheta
        # Compute the optimal theta
        theta = cov_rs/ var_s_tmp
        if theta > 1
            println("Warning: Optimal theta is larger than 1, most likely due to inaccurate (co)variance estimate. Setting theta to 1 for computations.")
            theta = 1;
        end
    end
    println("Optimal theta: $theta")
    
    mean_diff = mean_r - theta*mean_s

    # Process the additional samples for the low-accuracy solver `s`
    for n = (N_diffOpt + 1):N_samples_s
        alpha = samples_s[n]
        obj.alpha = alpha

        obj.settings.r = s
        _, gs = NumerSolver(obj)

        ys = Sol_to_FoI(obj, gs, FoI)
        ys_n[n,:] = ys
    end
    mean_s = Statistics.mean(ys_n,dims=1)
    var_s  = zeros(Float64,obj.settings.Nx);

    dev_s = ys_n .- mean_s
    factor = obj.settings.dx / (N_samples_s - 1)
    # Variance for s
    var_s = sum(dev_s.^2, dims=1) .* factor
    var_s = norm(var_s)

    # Final variance of the control variate estimator
    var = var_r - 2 * theta * cov_rs + theta^2 * var_s

    # Final estimate of the mean (control variate estimator)
    mean_sol = theta * mean_s + mean_diff
    
    output = Dict("N_samples_s" => N_samples_s, "N_samples_diff" => N_diffOpt, "mean" => mean_sol, "var" => var,"alpha"=>theta);

    return output;
end



function getOptParameters(obj::solver,epsilon,r,s, d,N_samples_diff,numerSolver::String;kwargs...)
    @unpack FoI = kwargs;
     # Variables to track number of processed samples
     count_diff = 0
     count_s = 0

    if numerSolver == "full"
        NumerSolver = solveFullProblem;
    elseif numerSolver == "frBUG"
        NumerSolver = solvefixedrankBUG;
    elseif numerSolver == "fixedAugBUG"
        NumerSolver = solvefixedAugBUG;
    end

     samples_diff = rand(d, N_samples_diff)
     yr_n = zeros(N_samples_diff, obj.settings.Nx)
     ys_n = zeros(N_samples_diff, obj.settings.Nx)
    #compute theta_opt and number of samples using N_samples_diff warmup samples
    for n = 1:N_samples_diff
        alpha = samples_diff[n]
        obj.alpha = alpha
        obj.settings.r = r
        _, gr = NumerSolver(obj)

        obj.settings.r = s
        _, gs = NumerSolver(obj)


        yr = Sol_to_FoI(obj, gr, FoI)
        ys = Sol_to_FoI(obj, gs, FoI)

        yr_n[n,:] = yr
        ys_n[n,:] = ys
    end
    mean_r = mean(yr_n,dims=1)
    mean_s = mean(ys_n,dims=1)

    var_r  = zeros(Float64,obj.settings.Nx);
    var_s  = zeros(Float64,obj.settings.Nx);
    cov_rs  = zeros(Float64,obj.settings.Nx);

    factor = obj.settings.dx / (N_samples_diff - 1)

    # Compute deviations from the mean
    dev_r = yr_n .- mean_r
    dev_s = ys_n .- mean_s

    # Variance for r
    var_r = sum(dev_r.^2, dims=1) .* factor

    # Variance for s
    var_s = sum(dev_s.^2, dims=1) .* factor

    # Covariance between r and s
    cov_rs = sum(dev_r .* dev_s, dims=1) .* factor
    var_r = norm(var_r)
    var_s = norm(var_s)
    cov_rs = norm(cov_rs)
    rho_rs = cov_rs/(sqrt(var_r)*sqrt(var_s))

    # Compute the optimal theta
    theta = cov_rs/ var_s
    println("Theta = $theta")

    #compute number samples
    N_diffOpt = Int(ceil((var_r - 2 * theta * cov_rs + theta^2 * var_s)/epsilon^2))
    println("Number samples = $N_diffOpt")

    return N_diffOpt,theta
end

function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end