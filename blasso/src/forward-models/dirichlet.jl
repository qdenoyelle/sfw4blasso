# Fourier coeff kernel
mutable struct dirichlet <: discrete
    fc::Int64
    dim::Int64
    nbpointsgrid::Int64
    grid::Array{Float64,1}
    bounds::Array{Float64,1}
end

function setKernel(fc::Int64)
  dim=1;
  bounds=[0.0,1.0];
  a,b=bounds[1],bounds[2];

  nbpointsgrid=convert(Integer,4*fc);
  gr=collect(range(a,stop=b,length=nbpointsgrid));

  return dirichlet(fc,dim,nbpointsgrid,gr,bounds)
end

# Operator which gives the vector of fourier coefficients
mutable struct fouriercoeff <: operator
    ker::DataType
    dim::Int64
    fc::Integer
    bounds::Array{Float64,1}

    normObs::Float64

    Phi::Array{Array{Float64,1}}
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

# Function that set the operator giving the fourier coefficients
function setoperator(kernel::dirichlet,a0::Array{Float64,1},x0::Array{Float64,1})
  function diriVect(x::Float64)
    v=ones(2kernel.fc+1);
    for i in 2:2kernel.fc+1
      if i<=kernel.fc+1
        v[i]=sqrt(2)*cos(2*pi*(i-1)*x);
      elseif i>kernel.fc+1
        v[i]=sqrt(2)*sin(2*pi*(i-(kernel.fc+1))*x);
      end
    end
    return v
  end
  function d1diriVect(x::Float64)
    v=zeros(2kernel.fc+1);
    for i in 2:2kernel.fc+1
      if i<=kernel.fc+1
        v[i]=-sqrt(2)*2*pi*(i-1)*sin(2*pi*(i-1)*x);
      elseif i>kernel.fc+1
        v[i]=sqrt(2)*2*pi*(i-(kernel.fc+1))*cos(2*pi*(i-(kernel.fc+1))*x);
      end
    end
    return v
  end
  function d2diriVect(x::Float64)
    v=zeros(2kernel.fc+1);
    for i in 2:2kernel.fc+1
      if i<=kernel.fc+1
        v[i]=-sqrt(2)*(2*pi*(i-1))^2*cos(2*pi*(i-1)*x);
      elseif i>kernel.fc+1
        v[i]=-sqrt(2)*(2*pi*(i-(kernel.fc+1)))^2*sin(2*pi*(i-(kernel.fc+1))*x);
      end
    end
    return v
  end

  c(x1::Float64,x2::Float64)=dot(diriVect(x1),diriVect(x2));
  d10c(x1::Float64,x2::Float64)=dot(d1diriVect(x1),diriVect(x2));
  d01c(x1::Float64,x2::Float64)=dot(diriVect(x1),d1diriVect(x2));
  d11c(x1::Float64,x2::Float64)=dot(d1diriVect(x1),d1diriVect(x2));
  d20c(x1::Float64,x2::Float64)=dot(d2diriVect(x1),diriVect(x2));
  d02c(x1::Float64,x2::Float64)=dot(diriVect(x1),d2diriVect(x2));

  y=sum([a0[i]*diriVect(x0[i]) for i in 1:length(x0)]);
  normObs=.5*norm(y)^2;

  Phi=Array{Array{Float64,1}}(undef,kernel.nbpointsgrid);
  for i in 1:kernel.nbpointsgrid
    Phi[i]=diriVect(kernel.grid[i]);
  end

  PhisY=zeros(kernel.nbpointsgrid);
  for i in 1:kernel.nbpointsgrid
    PhisY[i]=dot(Phi[i],y);
  end

  function ob(x::Float64)
    return dot(diriVect(x),y);
  end
  function d1ob(x::Float64)
    return dot(d1diriVect(x),y);
  end
  function d2ob(x::Float64)
    return dot(d2diriVect(x),y);
  end

  function correl(x::Array{Float64,1},Phiu::Array{Float64,1})
    return dot(diriVect(x[1]),Phiu-y);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Float64,1})
    return [dot(d1diriVect(x[1]),Phiu-y)];
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Float64,1})
    d2c=zeros(kernel.dim,kernel.dim);
    d2c[1,1]=dot(d2diriVect(x[1]),Phiu-y);
    return d2c;
  end

  fouriercoeff(typeof(kernel),kernel.dim,kernel.fc,kernel.bounds,normObs,Phi,PhisY,diriVect,d1diriVect,d2diriVect,y,c,ob,d10c,d01c,d11c,d20c,d02c,d1ob,d2ob,correl,d1correl,d2correl);
end

# Function that set the operator giving the fourier coefficients
function setoperator(kernel::dirichlet,a0::Array{Float64,1},x0::Array{Float64,1},w::Array{Float64,1})
  function diriVect(x::Float64)
    v=ones(2kernel.fc+1);
    for i in 2:2kernel.fc+1
      if i<=kernel.fc+1
        v[i]=sqrt(2)*cos(2*pi*(i-1)*x);
      elseif i>kernel.fc+1
        v[i]=sqrt(2)*sin(2*pi*(i-(kernel.fc+1))*x);
      end
    end
    return v
  end
  function d1diriVect(x::Float64)
    v=zeros(2kernel.fc+1);
    for i in 2:2kernel.fc+1
      if i<=kernel.fc+1
        v[i]=-sqrt(2)*2*pi*(i-1)*sin(2*pi*(i-1)*x);
      elseif i>kernel.fc+1
        v[i]=sqrt(2)*2*pi*(i-(kernel.fc+1))*cos(2*pi*(i-(kernel.fc+1))*x);
      end
    end
    return v
  end
  function d2diriVect(x::Float64)
    v=zeros(2kernel.fc+1);
    for i in 2:2kernel.fc+1
      if i<=kernel.fc+1
        v[i]=-sqrt(2)*(2*pi*(i-1))^2*cos(2*pi*(i-1)*x);
      elseif i>kernel.fc+1
        v[i]=-sqrt(2)*(2*pi*(i-(kernel.fc+1)))^2*sin(2*pi*(i-(kernel.fc+1))*x);
      end
    end
    return v
  end

  c(x1::Float64,x2::Float64)=dot(diriVect(x1),diriVect(x2));
  d10c(x1::Float64,x2::Float64)=dot(d1diriVect(x1),diriVect(x2));
  d01c(x1::Float64,x2::Float64)=dot(diriVect(x1),d1diriVect(x2));
  d11c(x1::Float64,x2::Float64)=dot(d1diriVect(x1),d1diriVect(x2));
  d20c(x1::Float64,x2::Float64)=dot(d2diriVect(x1),diriVect(x2));
  d02c(x1::Float64,x2::Float64)=dot(diriVect(x1),d2diriVect(x2));

  y=sum([a0[i]*diriVect(x0[i]) for i in 1:length(x0)])+w;
  normObs=.5*norm(y)^2;

  Phi=Array{Array{Float64,1}}(undef,kernel.nbpointsgrid);
  for i in 1:kernel.nbpointsgrid
    Phi[i]=diriVect(kernel.grid[i]);
  end

  PhisY=zeros(kernel.nbpointsgrid);
  for i in 1:kernel.nbpointsgrid
    PhisY[i]=dot(Phi[i],y);
  end

  function ob(x::Float64)
    return dot(diriVect(x),y);
  end
  function d1ob(x::Float64)
    return dot(d1diriVect(x),y);
  end
  function d2ob(x::Float64)
    return dot(d2diriVect(x),y);
  end

  function correl(x::Array{Float64,1},Phiu::Array{Float64,1})
    return dot(diriVect(x[1]),Phiu-y);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Float64,1})
    return [dot(d1diriVect(x[1]),Phiu-y)];
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Float64,1})
    d2c=zeros(kernel.dim,kernel.dim);
    d2c[1,1]=dot(d2diriVect(x[1]),Phiu-y);
    return d2c;
  end

  fouriercoeff(typeof(kernel),kernel.dim,kernel.fc,kernel.bounds,normObs,Phi,PhisY,diriVect,d1diriVect,d2diriVect,y,c,ob,d10c,d01c,d11c,d20c,d02c,d1ob,d2ob,correl,d1correl,d2correl);
end

# Function that set the operator giving the fourier coefficients
function setoperator(kernel::dirichlet,y::Array{Float64,1})
  function diriVect(x::Float64)
    v=ones(2kernel.fc+1);
    for i in 2:2kernel.fc+1
      if i<=kernel.fc+1
        v[i]=sqrt(2)*cos(2*pi*(i-1)*x);
      elseif i>kernel.fc+1
        v[i]=sqrt(2)*sin(2*pi*(i-(kernel.fc+1))*x);
      end
    end
    return v
  end
  function d1diriVect(x::Float64)
    v=zeros(2kernel.fc+1);
    for i in 2:2kernel.fc+1
      if i<=kernel.fc+1
        v[i]=-sqrt(2)*2*pi*(i-1)*sin(2*pi*(i-1)*x);
      elseif i>kernel.fc+1
        v[i]=sqrt(2)*2*pi*(i-(kernel.fc+1))*cos(2*pi*(i-(kernel.fc+1))*x);
      end
    end
    return v
  end
  function d2diriVect(x::Float64)
    v=zeros(2kernel.fc+1);
    for i in 2:2kernel.fc+1
      if i<=kernel.fc+1
        v[i]=-sqrt(2)*(2*pi*(i-1))^2*cos(2*pi*(i-1)*x);
      elseif i>kernel.fc+1
        v[i]=-sqrt(2)*(2*pi*(i-(kernel.fc+1)))^2*sin(2*pi*(i-(kernel.fc+1))*x);
      end
    end
    return v
  end

  c(x1::Float64,x2::Float64)=dot(diriVect(x1),diriVect(x2));
  d10c(x1::Float64,x2::Float64)=dot(d1diriVect(x1),diriVect(x2));
  d01c(x1::Float64,x2::Float64)=dot(diriVect(x1),d1diriVect(x2));
  d11c(x1::Float64,x2::Float64)=dot(d1diriVect(x1),d1diriVect(x2));
  d20c(x1::Float64,x2::Float64)=dot(d2diriVect(x1),diriVect(x2));
  d02c(x1::Float64,x2::Float64)=dot(diriVect(x1),d2diriVect(x2));

  normObs=.5*norm(y)^2;

  Phi=Array{Array{Float64,1}}(undef,kernel.nbpointsgrid);
  for i in 1:kernel.nbpointsgrid
    Phi[i]=diriVect(kernel.grid[i]);
  end

  PhisY=zeros(kernel.nbpointsgrid);
  for i in 1:kernel.nbpointsgrid
    PhisY[i]=dot(Phi[i],y);
  end

  function ob(x::Float64)
    return dot(diriVect(x),y);
  end
  function d1ob(x::Float64)
    return dot(d1diriVect(x),y);
  end
  function d2ob(x::Float64)
    return dot(d2diriVect(x),y);
  end

  function correl(x::Array{Float64,1},Phiu::Array{Float64,1})
    return dot(diriVect(x[1]),Phiu-y);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Float64,1})
    return [dot(d1diriVect(x[1]),Phiu-y)];
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Float64,1})
    d2c=zeros(kernel.dim,kernel.dim);
    d2c[1,1]=dot(d2diriVect(x[1]),Phiu-y);
    return d2c;
  end

  fouriercoeff(typeof(kernel),kernel.dim,kernel.fc,kernel.bounds,normObs,Phi,PhisY,diriVect,d1diriVect,d2diriVect,y,c,ob,d10c,d01c,d11c,d20c,d02c,d1ob,d2ob,correl,d1correl,d2correl);
end


function computePhiu(u::Array{Float64,1},op::blasso.fouriercoeff)
  a,X=blasso.decompAmpPos(u,d=op.dim);
  Phiu=sum([a[i]*op.phi(X[i]) for i in 1:length(a)]);
  return Phiu;
end

# Compute the argmin and min of the correl on the grid.
function minCorrelOnGrid(Phiu::Array{Float64,1},kernel::blasso.dirichlet,op::blasso.operator,positivity::Bool=true)
  correl_min,argmin=Inf,zeros(op.dim);
  l=1;
  for i in 1:kernel.nbpointsgrid
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

function setbounds(op::blasso.fouriercoeff,positivity::Bool=true,ampbounds::Bool=true)
  x_low=-Inf*ones(op.dim);
  x_up=Inf*ones(op.dim);

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
