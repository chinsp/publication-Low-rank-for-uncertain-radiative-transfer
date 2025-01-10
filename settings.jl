__precompile__

mutable struct settings
        ## Settings of the staggered grids
    # Number of spatial vertices
    Nx::Int64;
    # Number of cell centres
    NxC::Int64;
    # Start and end point of spatial domain
    a::Float64;
    b::Float64;
    # Grid cell width
    dx::Float64;
    
    ## Settings of temporal domain
    # End time
    Tend::Float64;
    # Time increment width
    dt::Float64;
    # CFL number 
    cfl::Float64; # CFL condition

    ## Settings for angular approximation
    # Number of moments
    nPN::Int64;
    
    ## Spatial grid
    x
    xMid

    ## Problem Settings
    problem::String;   

    ## Initial conditions
    ICType::String;
    BCType::String;

    ## Physical parameters
    sigmaS::Float64; ## Try to change this to get a non-constant value for the scattering coefficients
    sigmaA::Float64;

    ## Dynamcial low-rank approximation
    r::Int; # rank of the system
    epsAdapt::Float64; # Tolerance for rank adaptive BUG integrator

    function settings(Nx::Int=1001,nPN::Int=501)
        # Setup spatial grid
        NxC = Nx + 1;
        a = -1.5 # Starting point for the spatial interval
        b = 1.5 # End point for the spatial interval

        x = collect(range(a,stop = b,length = Nx));
        dx = x[2] - x[1];
        xMid = [x[1]-dx;x];
        xMid = xMid .+ dx/2

        # Problem 
        problem = "1DPlanesource" #  1DPlanesource

        # Scattering and absorption coefficients
        sigmaA = 0.0;
        sigmaS = 1.0;

        # Initial and Boundary condition
        ICType = "LS";
        BCType = "exact";

        # Defining the constants related to the simulation
       
        # Setup temporal discretisation
        Tend = 1.0;
        cfl = 1.0; # CFL condition hyperbolic
        
        dt = cfl*dx;
        
        # Settings for BUG integrator
        r = 30;
        epsAdapt = 0.05; # Tolerance for rank adaptive integrator

        new(Nx,NxC,a,b,dx,Tend,dt,cfl,nPN,x,xMid,problem,ICType,BCType,sigmaS,sigmaA,r,epsAdapt);
    end 
end

function IC(obj::settings,alpha::Float64,uncertParam::String)
    x = obj.x;
    y = zeros(size(obj.x));
    if obj.problem == "1DPlanesource"
        floor = 1e-4;
        if uncertParam == "0"
            s1 = 0.03;
            s2 = s1^2;
            for j = 1:length(y);
                y[j] = max(floor,1.0/(sqrt(2*pi)*s1) *exp(-((x[j]-alpha)*(x[j]-alpha))/2.0/s2))
            end
        elseif uncertParam == "1"
            x0 = 0.0
            s1 = 0.03;
            s2 = s1^2;
            for j = 1:length(y);
                y[j] = max(floor,alpha*1.0/(sqrt(2*pi)*s1) *exp(-((x[j]-x0)*(x[j]-x0))/2.0/s2))
            end
        else
            println("Choose between 0 and 1 for the uncertParam as a String")
        end
    else
        println("Initial condition not coded yet")
        
    end
    return y;
end