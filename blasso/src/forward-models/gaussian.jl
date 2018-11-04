mutable struct gaussian <: discrete
  dim::Int64
  p::Array{Float64,1}
  K::Int64

  nbpointsgrid::Array{Int64,1}
  grid::Array{Array{Float64,1},1}

  sigma::Float64
  bounds::Array{Float64,1}
end

function setKernel(K::Int64,sigma::Float64,bounds::Array{Float64,1})
  dim=1;
  a,b=bounds[1],bounds[2];
  p=collect(range(a,stop=b,length=K));

  coeff=.25;
  lSample=[coeff/sigma];
  nbpointsgrid=[convert(Int64,round(5*lSample[i]*abs(b-a);digits=0)) for i in 1:dim];
  g=Array{Array{Float64,1}}(undef,dim);
  for i in 1:dim
    g[i]=collect(range(a,stop=b,length=nbpointsgrid[i]));
  end

  return gaussian(dim,p,K,nbpointsgrid,g,sigma,bounds)
end

mutable struct gaussconv <: operator
  ker::DataType
  dim::Int64
  sigma::Float64
  bounds::Array{Float64,1}

  normObs::Float64

  Phix::Array{Array{Float64,1}}
  PhisY::Array{Float64,1}
  phi::Function
  d1phi::Function
  d2phi::Function
  y::Array{Float64,1}
  c::Function
  d10c::Function
  d01c::Function
  d11c::Function
  d20c::Function
  d02c::Function
  ob::Function
  d1ob::Function
  d2ob::Function
  correl::Function
  d1correl::Function
  d2correl::Function
end

# Functions that set the operator for the gaussian convolution in 2D when the initial measure is given
function setoperator(kernel::gaussian,a0::Array{Float64,1},x0::Array{Float64,1})
  function phiDx(x::Float64,p::Float64)
    return 1/sqrt(2*pi*kernel.sigma^2)*exp(-(p-x)^2/(2*kernel.sigma^2));
  end
  function d1phiDx(x::Float64,p::Float64)
    return (p-x)/(kernel.sigma^2*sqrt(2*pi*kernel.sigma^2))*exp(-(p-x)^2/(2*kernel.sigma^2));
  end
  function d2phiDx(x::Float64,p::Float64)
    return ((p-x)^2/(kernel.sigma^4*sqrt(2*pi*kernel.sigma^2))-1/(kernel.sigma^2*sqrt(2*pi*kernel.sigma^2)))*exp(-(p-x)^2/(2*kernel.sigma^2));
  end

  function phiVect(x::Float64)
    return [phiDx(x,p) for p in kernel.p];
  end
  function d1phiVect(x::Float64)
    return [d1phiDx(x,p) for p in kernel.p];
  end
  function d2phiVect(x::Float64)
    return [d2phiDx(x,p) for p in kernel.p];
  end

  Phix=phiVect.(kernel.grid[1]);

  c(x1::Float64,x2::Float64)=dot(phiVect(x1),phiVect(x2));
  d10c(x1::Float64,x2::Float64)=dot(d1phiVect(x1),phiVect(x2));
  d01c(x1::Float64,x2::Float64)=dot(phiVect(x1),d1phiVect(x2));
  d11c(x1::Float64,x2::Float64)=dot(d1phiVect(x1),d1phiVect(x2));
  d20c(x1::Float64,x2::Float64)=dot(d2phiVect(x1),phiVect(x2));
  d02c(x1::Float64,x2::Float64)=dot(phiVect(x1),d2phiVect(x2));

  N=length(x0);
  y=sum([a0[i]*phiVect(x0[i]) for i in 1:N]);
  normObs=.5*dot(y,y)

  PhisY=zeros(length(kernel.grid[1]));
  for i in 1:length(kernel.grid[1])
    PhisY[i]=dot(Phix[i],y);
  end

  function ob(x::Float64,a0::Array{Float64,1}=a0,x0::Array{Float64,1}=x0)
    return dot(phiVect(x[1]),y);
  end
  function d1ob(x::Float64,a0::Array{Float64,1}=a0,x0::Array{Float64,1}=x0)
    return dot(d1phiVect(x[1]),y);
  end
  function d2ob(x::Float64,a0::Array{Float64,1}=a0,x0::Array{Float64,1}=x0)
    return dot(d2phiVect(x[1]),y);
  end

  function correl(x::Array{Float64,1},Phiu::Array{Float64,1})
    return dot(phiVect(x[1]),Phiu-y);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Float64,1})
    return [dot(d1phiVect(x[1]),Phiu-y)];
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Float64,1})
    d2c=zeros(op.dim,op.dim);
    d2c[1,1]=dot(d2phiVect(x[1]),Phiu-y);
    return d2c;
  end

  gaussconv(typeof(kernel),kernel.dim,kernel.sigma,kernel.bounds,normObs,Phix,PhisY,phiVect,d1phiVect,d2phiVect,y,c,d10c,d01c,d11c,d20c,d02c,ob,d1ob,d2ob,correl,d1correl,d2correl);
end
function setoperator(kernel::gaussian,a0::Array{Float64,1},x0::Array{Float64,1},w::Array{Float64,1})
  function phiDx(x::Float64,p::Float64)
    return 1/sqrt(2*pi*kernel.sigma^2)*exp(-(p-x)^2/(2*kernel.sigma^2));
  end
  function d1phiDx(x::Float64,p::Float64)
    return (p-x)/(kernel.sigma^2*sqrt(2*pi*kernel.sigma^2))*exp(-(p-x)^2/(2*kernel.sigma^2));
  end
  function d2phiDx(x::Float64,p::Float64)
    return ((p-x)^2/(kernel.sigma^4*sqrt(2*pi*kernel.sigma^2))-1/(kernel.sigma^2*sqrt(2*pi*kernel.sigma^2)))*exp(-(p-x)^2/(2*kernel.sigma^2));
  end

  function phiVect(x::Float64)
    return [phiDx(x,p) for p in kernel.p];
  end
  function d1phiVect(x::Float64)
    return [d1phiDx(x,p) for p in kernel.p];
  end
  function d2phiVect(x::Float64)
    return [d2phiDx(x,p) for p in kernel.p];
  end

  Phix=phiVect.(kernel.grid[1]);

  c(x1::Float64,x2::Float64)=dot(phiVect(x1),phiVect(x2));
  d10c(x1::Float64,x2::Float64)=dot(d1phiVect(x1),phiVect(x2));
  d01c(x1::Float64,x2::Float64)=dot(phiVect(x1),d1phiVect(x2));
  d11c(x1::Float64,x2::Float64)=dot(d1phiVect(x1),d1phiVect(x2));
  d20c(x1::Float64,x2::Float64)=dot(d2phiVect(x1),phiVect(x2));
  d02c(x1::Float64,x2::Float64)=dot(phiVect(x1),d2phiVect(x2));

  N=length(x0);
  y=sum([a0[i]*phiVect(x0[i]) for i in 1:N])+w;
  normObs=.5*dot(y,y)

  PhisY=zeros(length(kernel.grid[1]));
  for i in 1:length(kernel.grid[1])
    PhisY[i]=dot(Phix[i],y);
  end

  function ob(x::Float64,a0::Array{Float64,1}=a0,x0::Array{Float64,1}=x0)
    return dot(phiVect(x[1]),y);
  end
  function d1ob(x::Float64,a0::Array{Float64,1}=a0,x0::Array{Float64,1}=x0)
    return dot(d1phiVect(x[1]),y);
  end
  function d2ob(x::Float64,a0::Array{Float64,1}=a0,x0::Array{Float64,1}=x0)
    return dot(d2phiVect(x[1]),y);
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

  gaussconv(typeof(kernel),kernel.dim,kernel.sigma,kernel.bounds,normObs,Phix,PhisY,phiVect,d1phiVect,d2phiVect,y,c,d10c,d01c,d11c,d20c,d02c,ob,d1ob,d2ob,correl,d1correl,d2correl);
end

function computePhiu(u::Array{Float64,1},op::blasso.gaussconv)
  a,X=blasso.decompAmpPos(u,d=op.dim);
  return sum([a[i]*op.phi(X[i]) for i in 1:length(a)]);
end

# Compute the argmin and min of the correl on the grid.
function minCorrelOnGrid(Phiu::Array{Float64,1},kernel::blasso.gaussian,op::blasso.operator,positivity::Bool=true)
  correl_min,argmin=Inf,zeros(op.dim);
  l=1;
  for i in 1:length(kernel.grid[1])
    buffer=dot(op.Phix[i],Phiu)-op.PhisY[i];
    if !positivity
      buffer=-abs(buffer);
    end
    if buffer<correl_min
      correl_min=buffer;
      argmin=[kernel.grid[1][i]];
    end
  end

  return argmin,correl_min
end

function setbounds(op::blasso.gaussconv,positivity::Bool=true,ampbounds::Bool=true)
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
