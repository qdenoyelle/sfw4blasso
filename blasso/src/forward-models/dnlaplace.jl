# Normalized Discrete Laplace Transform
mutable struct dnlaplace <: dLaplace
  rl_model::Bool
  dim::Int64
  bounds::Array{Float64,1}
  K::Int64
  p::Array{Float64,1}
  nbpointsgrid::Int64
  grid::Array{Float64,1}
end

function setKernel(rl_model::Bool,bounds::Array{Float64,1},p::Array{Float64,1};kwargs...)
  dim=1;
  a,b=bounds[1],bounds[2];

  key_kw=[kwargs[i][1] for i in 1:length(kwargs)];
  if :nbpointsgrid in key_kw
    nbpointsgrid=kwargs[find(key_kw.==:nbpointsgrid)][1][2];
  else
    lSample=3;
    nbpointsgrid=convert(Int64,round(5*lSample*abs(b-a);digits=0));
  end
  g=collect(range(a,stop=b,length=nbpointsgrid));

  return dnlaplace(rl_model,dim,bounds,length(p),p,nbpointsgrid,g)
end

mutable struct discretenormalizedlaplacetransform <: operator
  ker::DataType
  dim::Int64
  rl_model::Bool
  bounds::Array{Float64,1}

  normObs::Float64

  normPhi::Function

  Phi::Array{Array{Float64,1},1}
  PhisY::Array{Float64,1}
  phi::Function
  d1phi::Function
  d2phi::Function
  y::Array{Float64,1}
  c::Function
  ob::Function
  d10c::Function
  d01c::Function
  d11c::Function
  d20c::Function
  d02c::Function
  d1ob::Function
  d2ob::Function
  correl::Function
  d1correl::Function
  d2correl::Function
end

# Function that set the operator for the discrete normalized Laplace Transform
function setoperator(kernel::dnlaplace,a0::Array{Float64,1},x0::Array{Float64,1},w::Array{Float64,1})
  p=kernel.p;
  f(x::Float64,y::Float64)=sum([exp(-pi*(x+y)) for pi in p]);
  d10f(x::Float64,y::Float64)=sum([-pi*exp(-pi*(x+y)) for pi in p]);
  d20f(x::Float64,y::Float64)=sum([pi^2*exp(-pi*(x+y)) for pi in p]);
  d11f(x::Float64,y::Float64)=d20f(x,y);
  g(x::Float64,y::Float64)=sum(sum([[exp(-2*(pi*x+pj*y)) for pi in p] for pj in p]))^(-.5);
  g1(x::Float64,y::Float64)=sum(sum([[pi*exp(-2*(pi*x+pj*y)) for pi in p] for pj in p]));
  g2(x::Float64,y::Float64)=sum(sum([[pi^2*exp(-2*(pi*x+pj*y)) for pi in p] for pj in p]));
  g3(x::Float64,y::Float64)=sum(sum([[pi*pj*exp(-2*(pi*x+pj*y)) for pi in p] for pj in p]));
  d10g(x::Float64,y::Float64)=g1(x,y)*g(x,y)^3;
  d20g(x::Float64,y::Float64)=(-2*g2(x,y)+3*(g1(x,y)*g(x,y))^2)*g(x,y)^3;
  d11g(x::Float64,y::Float64)=(-2*g3(x,y)+3*g1(x,y)*g1(y,x)*g(x,y)^2)*g(x,y)^3;

  c(x::Float64,y::Float64)=f(x,y)*g(x,y);
  d10c(x::Float64,y::Float64)=d10f(x,y)*g(x,y)+f(x,y)*d10g(x,y);
  d01c(x::Float64,y::Float64)=d10c(y,x);
  d11c(x::Float64,y::Float64)=d11f(x,y)*g(x,y)+d10f(x,y)*d10g(y,x)+d10f(y,x)*d10g(x,y)+f(x,y)*d11g(x,y);
  d20c(x::Float64,y::Float64)=d20f(x,y)*g(x,y)+2*d10f(x,y)*d10g(x,y)+f(x,y)*d20g(x,y);
  d02c(x::Float64,y::Float64)=d20c(y,x);

  fob(x::Float64)=[exp(-pi*x) for pi in p];
  d1fob(x::Float64)=[-pi*exp(-pi*x) for pi in p];
  d2fob(x::Float64)=[pi^2*exp(-pi*x) for pi in p];
  gob(x::Float64)=sum([exp(-2*(pi*x)) for pi in p])^(-.5);
  d1gob(x::Float64)=-sum(d1fob(2*x))*gob(x)^3;
  d2gob(x::Float64)=(-2*sum(d2fob(2*x))+3*(sum(d1fob(2*x))*gob(x))^2)*gob(x)^3;

  phiVect(x::Float64)=gob(x).*fob(x);
  d1phiVect(x::Float64)=d1gob(x).*fob(x)+gob(x).*d1fob(x);
  d2phiVect(x::Float64)=d2gob(x).*fob(x)+2*d1gob(x).*d1fob(x)+gob(x).*d2fob(x);

  Phi=phiVect.(kernel.grid);

  normPhi(x::Float64)=norm(fob(x));

  N=length(x0);
  a0_scaled=Array{Float64}(undef,N);
  @simd for i in 1:N
    if kernel.rl_model
      @inbounds a0_scaled[i]=a0[i]*normPhi(x0[i]);
    else
      @inbounds a0_scaled[i]=a0[i];
    end
  end

  y=sum([a0_scaled[i]*phiVect(x0[i]) for i in 1:length(x0)])+w;
  normObs=.5*dot(y,y);

  PhisY=zeros(length(kernel.grid));
  for i in 1:length(kernel.grid)
    PhisY[i]=dot(Phi[i],y);
  end

  function ob(x::Float64,y::Array{Float64,1}=y)
    return dot(phiVect(x),y);
  end
  function d1ob(x::Float64,y::Array{Float64,1}=y)
    return dot(d1phiVect(x),y);
  end
  function d2ob(x::Float64,y::Array{Float64,1}=y)
    return dot(d2phiVect(x),y);
  end

  function correl(x::Array{Float64,1},Phiu::Array{Float64,1})
    return dot(phiVect(x[1]),Phiu-y);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Float64,1})
    return [dot(d1phiVect(x[1]),Phiu-y)];
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Float64,1})
    d2c=zeros(kernel.dim,kernel.dim);
    d2c[1,1]=dot(d2phiVect(x[1]),Phiu-y);
    return d2c;
  end

  discretenormalizedlaplacetransform(typeof(kernel),kernel.dim,kernel.rl_model,kernel.bounds,normObs,normPhi,Phi,PhisY,phiVect,d1phiVect,d2phiVect,y,c,ob,d10c,d01c,d11c,d20c,d02c,d1ob,d2ob,correl,d1correl,d2correl);
end

function computePhiu(u::Array{Float64,1},op::blasso.discretenormalizedlaplacetransform)
  a,X=blasso.decompAmpPos(u,d=op.dim);
  Phiu=sum([a[i]*op.phi(X[i]) for i in 1:length(a)]);
  return Phiu;
end

# Compute the argmin and min of the correl on the grid.
function minCorrelOnGrid(Phiu::Array{Float64,1},kernel::blasso.dnlaplace,op::blasso.operator,positivity::Bool=true)
  correl_min,argmin=Inf,zeros(op.dim);
  for i in 1:length(kernel.grid)
    buffer=dot(op.Phi[i],Phiu)-op.PhisY[i];
    if !positivity
      buffer=-abs(buffer);
    end
    if buffer<correl_min
      correl_min=buffer;
      argmin=[kernel.grid[i]];
    end
  end

  return argmin,correl_min
end

function setbounds(op::blasso.discretenormalizedlaplacetransform,positivity::Bool=true,ampbounds::Bool=true)
  x_low=op.bounds[1];
  x_up=op.bounds[2];

  if ampbounds
    if positivity
      a_low=0.0;
    else
      a_low=-Inf;
    end
    a_up=Inf;
    return a_low,a_up,x_low,x_up
  else
    return x_low,x_up
  end
end
