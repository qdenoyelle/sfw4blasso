module certificate

using LinearAlgebra, PyPlot
using blasso,toolbox

include("certificate_plots.jl");

function computeEtaL(u::Array{Float64,1},op::blasso.operator,lambda::Float64)
    Phiu=blasso.computePhiu(u,op);
    etaL(x::Array{Float64,1})=-1/lambda*op.correl(x,Phiu);
    d1etaL(x::Array{Float64,1})=-1/lambda*op.d1correl(x,Phiu);
    d2etaL(x::Array{Float64,1})=-1/lambda*op.d2correl(x,Phiu);

    return etaL,d1etaL,d2etaL
end

function testDegenCert(eta::Array{Float64,1},tol::Float64=1e-3)
  bool=false;
  bool_pos=false;

  for i in 1:length(eta)
    if abs(eta[i])>1.0+tol
      bool=true;
      break;
    end
  end
  if bool
    for i in 1:length(eta)
      if eta[i]>1.0+tol
        bool_pos=true;
        break;
      end
    end
  end

  return(bool,bool_pos);
end

function computeEtaV(x0::Array{Float64,1},s::Array{Float64,1},op::blasso.operator)
    N=length(x0);
    G=zeros(2*N,2*N);

    for i in 1:N
        for j in 1:N
            G[i,j]=op.c(x0[i],x0[j]);
            G[i+N,j]=op.d10c(x0[i],x0[j]);
            G[i,j+N]=op.d01c(x0[i],x0[j]);
            G[i+N,j+N]=op.d11c(x0[i],x0[j]);
        end
    end

    delta=vcat(s,zeros(N));
    a=G\delta;

    function etaV(x::Float64)
        return sum([a[i]*op.c(x,x0[i]) for i in 1:N])+sum([a[i+N]*op.d01c(x,x0[i]) for i in 1:N]);
    end

    return etaV
end

function computeEtaV(x0::Array{Array{Float64,1},1},s::Array{Float64,1},op::blasso.operator)
    N=length(x0);
    G=zeros(N*(1+op.dim),N*(1+op.dim));

    for i in 1:N
        for j in 1:N
            G[i,j]=op.c(x0[i],x0[j]);
            for k in 1:op.dim
                G[i+k*N,j]=op.d10c(k,x0[i],x0[j]);
                G[i,j+k*N]=op.d01c(k,x0[i],x0[j]);
                for l in 1:op.dim
                  if op.dim==3
                    G[i+k*N,j+l*N]=op.d11c(1+(k-1)*(op.dim-1),2+(l-1)*(op.dim-1),x0[i],x0[j]);
                  else
                    G[i+k*N,j+l*N]=dot(op.d1phi(k,x0[i]),op.d1phi(l,x0[j]));#op.d11c(1+(k-1)*(op.dim-1),3+(l-1)*(op.dim-1),x0[i],x0[j]);
                  end
                end
            end
        end
    end

    delta=vcat(s,zeros(op.dim*N));
    a=G\delta;

    v=sum([a[i]*op.phi(x0[i]) for i in 1:N]) + sum([sum([a[i+k*N]*op.d1phi(k,x0[i]) for i in 1:N]) for k in 1:op.dim]);

    function etaV(x::Array{Float64,1})
        return dot(op.phi(x),v);
    end

    return etaV
end

end # module
