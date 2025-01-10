__precompile__

using ProgressMeter
using LinearAlgebra
using FastGaussQuadrature, LegendrePolynomials
using PyCall

mutable struct solver
    # Spatial grid of cell vertices
    x::Array{Float64,1};
    xMid::Array{Float64,1};

    # Solver settings
    settings::settings;

    ## Angular discretisation

    # Pn discretisation 
    A::Array{Float64,2};
    absA::Array{Float64,2};
    G::Array{Float64,2};

    # Stencil matrices for spatial discretisation
    Dx::Array{Float64,2};
    Dxx::Array{Float64,2};

    # Physical parameters
    sigmaA::Float64;
    sigmaS::Float64;

    # Uncertainty for initial condition
    alpha::Float64;
    uncertParam::String;
    # Constructor
    function solver(settings)
        x = settings.x;
        xMid = settings.xMid;

        nx = settings.Nx;
        nxC = settings.NxC;

        # Setting up the matrices for the Pn solver
        nPN = settings.nPN; # total number of Legendre polynomials used
        gamma = zeros(nPN+1); # vector with norms of the Legendre polynomials

        for i = 1:nPN+1
            gamma[i] = 2/(2*(i-1) + 1);
        end

        A = zeros(Float64,nPN+1,nPN+1); # reduced Flux matrix for the micro equation
        a_norm = zeros(Float64,nPN+1);

        for i = 1:nPN+1
            a_norm[i] = i/(sqrt((2i-1)*(2i+1)));
        end

        A = Tridiagonal(a_norm[1:end-1],zeros(nPN+1),a_norm[1:end-1]); # Full flux matrix

        S = eigvals(Matrix(A));
        V = eigvecs(Matrix(A));
        absA = V*abs.(diagm(S))*inv(V);

        G = diagm(ones(Float64,nPN+1));
        G[1,1] = 0.0;

        dx = settings.dx;
        

        # Stencil matrices for the Pn solver
        Dx = zeros(Float64,nx,nx);
        Dxx = zeros(Float64,nx,nx);

        Dx = Tridiagonal(-ones(nx-1)./2.0/dx, zeros(nx), ones(nx-1)./2.0/dx);
        Dxx = Tridiagonal(ones(nx-1)./2.0/dx, -ones(nx)./dx, ones(nx-1)./2.0/dx);

        alpha = 0.0;

        uncertParam = "1";

        new(x,xMid,settings,A,absA,G,Dx,Dxx,settings.sigmaA,settings.sigmaS,alpha,uncertParam);
    end
 end


 function setupIC(obj::solver)
    g = zeros(obj.settings.Nx,obj.settings.nPN+1);
    g[:,1] = IC(obj.settings,obj.alpha,obj.uncertParam);
    return g;
 end

 py"""
 import numpy
 def qr(A):
    return numpy.linalg.qr(A)
 """

 function solveFullProblem(obj::solver)
    t = 0.0;
    dt = obj.settings.dt;
    Tend = obj.settings.Tend;

    g = setupIC(obj);
    g0 = zeros(size(g));

    Nt = round(Tend/dt); # Computing the number of steps required 
    dt = Tend/Nt; # Adjusting the step size 
    for k = 1:Nt
        g0 .= g
        g .= g .- dt.*(obj.Dx*g0*Transpose(obj.A));
        g .= g .+ dt.*(obj.Dxx*g0*Transpose(obj.absA));
        g .= g .- dt.*(obj.settings.sigmaS*g0*obj.G);
        g .= g .- dt.*(obj.settings.sigmaA*g0);

        # g .= g .+ dt.*(-obj.Dx*g*Transpose(obj.A) .+ obj.Dxx*g*Transpose(obj.absA) .- obj.settings.sigmaS*g*obj.G .- obj.settings.sigmaA*g);
    end

    return g;

end

function solveFullProblem(alpha::Float64,Nx::Int64,N::Int)
    s = settings1D(Nx,N);
    solver = solver1D(s);

    solver.alpha = alpha;

    g = solveFullProblem(solver);

    return g
end

function solvefixedrankBUG(obj::solver)
    t = 0.0;
    dt = obj.settings.dt;
    Tend = obj.settings.Tend;
    r = obj.settings.r;

    g = setupIC(obj);

    X,s,V = svd(g);
    X = X[:,1:r];
    V = V[:,1:r];
    S = diagm(s[1:r]);

    VGV = zeros(Float64,r,r);
    VAV = zeros(Float64,r,r);
    VabsAV = zeros(Float64,r,r);
    
    XDxX = zeros(Float64,r,r);
    XDxxX = zeros(Float64,r,r);

    K = zeros(size(X));
    K0 = zeros(size(X));
    Lt = zeros(size(transpose(V)));
    Lt0 = zeros(size(transpose(V)));
    S0 = zeros(size(S));

    Nt = round(Tend/dt); # Computing the number of steps required 
    dt = Tend/Nt; # Adjusting the step size 

    r = obj.settings.r;

    for k = 1:Nt

        ## K-step
        K .= X*S;
        K0 .= K;
        VGV .= transpose(V)*obj.G*V;
        VAV .= transpose(V)*transpose(obj.A)*V;
        VabsAV .= transpose(V)*transpose(obj.absA)*V;

        # K .= K .+ dt*(-Dx*K*VAV + Dxx*K*VabsAV - obj.settings.sigmaS*K*VGV - obj.settings.sigmaA*K);
        K .= K .- dt.*obj.Dx*K0*VAV;
        K .= K .+ dt.*obj.Dxx*K0*VabsAV;
        K .= K .- dt.*obj.settings.sigmaS*K0*VGV;
        K .= K .- dt.*obj.settings.sigmaA*K0;

        X1,_ = py"qr"(K);
        M = transpose(X1)*X;
        
        ## L-step
        Lt .= S*transpose(V);
        Lt0 .= Lt;
        XDxX .= transpose(X)*obj.Dx*X;
        XDxxX .= transpose(X)*obj.Dxx*X;
        
        # Lt .= Lt .+ dt*(-XDxX*Lt*Transpose(A) + XDxxX*Lt*Transpose(absA) - obj.settings.sigmaS*Lt*G - obj.settings.sigmaA*Lt);
        Lt .= Lt .- dt.*XDxX*Lt0*Transpose(obj.A);
        Lt .= Lt .+ dt.*XDxxX*Lt0*Transpose(obj.absA);
        Lt .= Lt .- dt.*obj.settings.sigmaS*Lt0*obj.G;
        Lt .= Lt .- dt.*obj.settings.sigmaA*Lt0;

        V1,_ = py"qr"(transpose(Lt)); 
        N = transpose(V1)*V;

        X,V = X1, V1;

        ## S-step
        S .= M*S*transpose(N);
        S0 .= S;
        XDxX .= transpose(X)*obj.Dx*X;
        XDxxX .= transpose(X)*obj.Dxx*X;
        VGV .= transpose(V)*obj.G*V;
        VAV .= transpose(V)*transpose(obj.A)*V;
        VabsAV .= transpose(V)*transpose(obj.absA)*V;
        
        # S .= S .+ dt*(-XDxX*S*VAV + XDxxX*S*VabsAV - obj.settings.sigmaS*S*VGV - obj.settings.sigmaA*S)

        S .= S .- dt.*XDxX*S0*VAV;
        S .= S .+ dt.*XDxxX*S0*VabsAV;
        S .= S .- dt.*obj.settings.sigmaS*S0*VGV;
        S .= S .- dt.*obj.settings.sigmaA*S0;

        t = t+dt;
    end
    return X*S*transpose(V);

end

function solvefixedrankBUG(alpha::Float64,Nx::Int64,N::Int,r::Int)
    s = settings(Nx,N);
    solver = solver(s);
    solver.settings.r = r;

    solver.alpha = alpha;

    g = solvefixedrankBUG(solver);

    return g
end

function solvefixedAugBUG(obj::solver)
    t = 0.0;
    dt = obj.settings.dt;
    Tend = obj.settings.Tend;
    r = obj.settings.r;

    g = setupIC(obj);

    X,s,V = svd(g);
    X = X[:,1:r];
    V = V[:,1:r];
    S = diagm(s[1:r]);

    VGVr = zeros(Float64,r,r);
    VAVr = zeros(Float64,r,r);
    VabsAVr = zeros(Float64,r,r);
    
    XDxXr = zeros(Float64,r,r);
    XDxxXr = zeros(Float64,r,r);

    VGV2r = zeros(Float64,2*r,2*r);
    VAV2r = zeros(Float64,2*r,2*r);
    VabsAV2r = zeros(Float64,2*r,2*r);
    
    XDxX2r = zeros(Float64,2*r,2*r);
    XDxxX2r = zeros(Float64,2*r,2*r);

    K = zeros(size(X));
    K0 = zeros(size(X));
    Lt = zeros(size(transpose(V)));
    Lt0 = zeros(size(transpose(V)));
    S0 = zeros(2*r,2*r);
    Nt = round(Tend/dt); # Computing the number of steps required 
    dt = Tend/Nt; # Adjusting the step size 

    r = obj.settings.r;

    for k = 1:Nt

        ## K-step

        K .= X*S;
        K0 .= K;
        VGVr .= transpose(V)*obj.G*V;
        VAVr .= transpose(V)*transpose(obj.A)*V;
        VabsAVr .= transpose(V)*transpose(obj.absA)*V;

        K .= K .- dt.*obj.Dx*K0*VAVr;
        K .= K .+ dt.*obj.Dxx*K0*VabsAVr;
        K .= K .- dt.*obj.settings.sigmaS*K0*VGVr;
        K .= K .- dt.*obj.settings.sigmaA*K0;

        Xhat,_ = py"qr"([X K]); 
        M = transpose(Xhat)*X;
        
        
        ## L-step
        Lt .= S*transpose(V);
        Lt0 .= Lt;
        XDxXr .= transpose(X)*obj.Dx*X;
        XDxxXr .= transpose(X)*obj.Dxx*X;
        
        Lt .= Lt .- dt.*XDxXr*Lt0*Transpose(obj.A);
        Lt .= Lt .+ dt.*XDxxXr*Lt0*Transpose(obj.absA);
        Lt .= Lt .- dt.*obj.settings.sigmaS*Lt0*obj.G;
        Lt .= Lt .- dt.*obj.settings.sigmaA*Lt0;

        Vhat,_ = py"qr"([transpose(Lt) V]);
        N = transpose(Vhat)*V;

        # X,V = Xhat, Vhat;

        ## S-step
        S = M*S*transpose(N);
        S0 = S;
        XDxX2r .= transpose(Xhat)*obj.Dx*Xhat;
        XDxxX2r .= transpose(Xhat)*obj.Dxx*Xhat;
        VGV2r .= transpose(Vhat)*obj.G*Vhat;
        VAV2r .= transpose(Vhat)*transpose(obj.A)*Vhat;
        VabsAV2r .= transpose(Vhat)*transpose(obj.absA)*Vhat;
        
        S .= S .- dt.*XDxX2r*S0*VAV2r;
        S .= S .+ dt.*XDxxX2r*S0*VabsAV2r;
        S .= S .- dt.*obj.settings.sigmaS*S0*VGV2r;
        S .= S .- dt.*obj.settings.sigmaA*S0;

        P,sig,Q = svd(S);

        X .= Xhat*P[:,1:r];
        V .= Vhat*Q[:,1:r];
        S = diagm(sig[1:r]);

        t = t+dt;
    end
    return X*S*transpose(V);

end

function solvefixedAugBUG(alpha::Float64,Nx::Int64,N::Int,r::Int)
    s = settings(Nx,N);
    solver = solver(s);
    solver.settings.r = r;

    solver.alpha = alpha;

    g = solvefixedAugBUG(solver);

    return g
end