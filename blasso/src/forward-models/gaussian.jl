type gaussian <: discrete
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
  p=collect(linspace(a,b,K));

  coeff=.25;
  lSample=[coeff/sigma];
  nbpointsgrid=[convert(Int64,round(5*lSample[i]*abs(b-a),0)) for i in 1:dim];
  g=Array{Array{Float64,1}}(dim);
  for i in 1:dim
    g[i]=collect(linspace(a,b,nbpointsgrid[i]));
  end

  return gaussian(dim,p,K,nbpointsgrid,g,sigma,bounds)
end

type gaussconv <: operator
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

  c(x1::Float64,x2::Float64)=vecdot(phiVect(x1),phiVect(x2));
  d10c(x1::Float64,x2::Float64)=vecdot(d1phiVect(x1),phiVect(x2));
  d01c(x1::Float64,x2::Float64)=vecdot(phiVect(x1),d1phiVect(x2));
  d11c(x1::Float64,x2::Float64)=vecdot(d1phiVect(x1),d1phiVect(x2));
  d20c(x1::Float64,x2::Float64)=vecdot(d2phiVect(x1),phiVect(x2));
  d02c(x1::Float64,x2::Float64)=vecdot(phiVect(x1),d2phiVect(x2));

  N=length(x0);
  y=sum([a0[i]*phiVect(x0[i]) for i in 1:N]);
  normObs=.5*vecdot(y,y)

  PhisY=zeros(length(kernel.grid[1]));
  for i in 1:length(kernel.grid[1])
    PhisY[i]=vecdot(Phix[i],y);
  end

  function ob(x::Float64,a0::Array{Float64,1}=a0,x0::Array{Float64,1}=x0)
    return vecdot(phiVect(x[1]),y);
  end
  function d1ob(x::Float64,a0::Array{Float64,1}=a0,x0::Array{Float64,1}=x0)
    return vecdot(d1phiVect(x[1]),y);
  end
  function d2ob(x::Float64,a0::Array{Float64,1}=a0,x0::Array{Float64,1}=x0)
    return vecdot(d2phiVect(x[1]),y);
  end

  function correl(x::Array{Float64,1},Phiu::Array{Float64,1})
    return vecdot(phiVect(x[1]),Phiu-y);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Float64,1})
    return vecdot(d1phiVect(x[1]),Phiu-y);
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Float64,1})
    return vecdot(d2phiVect(x[1]),Phiu-y);
  end

  # function correl(x::Array{Float64,1},Phiu::Array{Float64,1})
  #   return vecdot(phiVect(x[1]),Phiu)-ob(x[1]);
  # end
  # function d1correl(x::Array{Float64,1},Phiu::Array{Float64,1})
  #   return vecdot(d1phiVect(x[1]),Phiu)-d1ob(x[1]);
  # end
  # function d2correl(x::Array{Float64,1},Phiu::Array{Float64,1})
  #   return vecdot(d2phiVect(x[1]),Phiu)-d2ob(x[1]);
  # end

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

  c(x1::Float64,x2::Float64)=vecdot(phiVect(x1),phiVect(x2));
  d10c(x1::Float64,x2::Float64)=vecdot(d1phiVect(x1),phiVect(x2));
  d01c(x1::Float64,x2::Float64)=vecdot(phiVect(x1),d1phiVect(x2));
  d11c(x1::Float64,x2::Float64)=vecdot(d1phiVect(x1),d1phiVect(x2));
  d20c(x1::Float64,x2::Float64)=vecdot(d2phiVect(x1),phiVect(x2));
  d02c(x1::Float64,x2::Float64)=vecdot(phiVect(x1),d2phiVect(x2));

  N=length(x0);
  y=sum([a0[i]*phiVect(x0[i]) for i in 1:N])+w;
  normObs=.5*vecdot(y,y)

  PhisY=zeros(length(kernel.grid[1]));
  for i in 1:length(kernel.grid[1])
    PhisY[i]=vecdot(Phix[i],y);
  end

  function ob(x::Float64,a0::Array{Float64,1}=a0,x0::Array{Float64,1}=x0)
    return vecdot(phiVect(x[1]),y);
  end
  function d1ob(x::Float64,a0::Array{Float64,1}=a0,x0::Array{Float64,1}=x0)
    return vecdot(d1phiVect(x[1]),y);
  end
  function d2ob(x::Float64,a0::Array{Float64,1}=a0,x0::Array{Float64,1}=x0)
    return vecdot(d2phiVect(x[1]),y);
  end

  function correl(x::Array{Float64,1},Phiu::Array{Float64,1})
    return vecdot(phiVect(x[1]),Phiu-y);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Float64,1})
    return vecdot(d1phiVect(x[1]),Phiu-y);
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Float64,1})
    return vecdot(d2phiVect(x[1]),Phiu-y);
  end

  # function correl(x::Array{Float64,1},Phiu::Array{Float64,1})
  #   return vecdot(phiVect(x[1]),Phiu)-ob(x[1]);
  # end
  # function d1correl(x::Array{Float64,1},Phiu::Array{Float64,1})
  #   return vecdot(d1phiVect(x[1]),Phiu)-d1ob(x[1]);
  # end
  # function d2correl(x::Array{Float64,1},Phiu::Array{Float64,1})
  #   return vecdot(d2phiVect(x[1]),Phiu)-d2ob(x[1]);
  # end

  gaussconv(typeof(kernel),kernel.dim,kernel.sigma,kernel.bounds,normObs,Phix,PhisY,phiVect,d1phiVect,d2phiVect,y,c,d10c,d01c,d11c,d20c,d02c,ob,d1ob,d2ob,correl,d1correl,d2correl);
end

# gaussian convolution in 2D when only y is given
function setoperator(kernel::gaussian,y::Array{Float64,1})
  function phiDx(x::Float64,xi::Float64)
    return .5*(erf( (xi+kernel.Dp/2-x)/(sqrt(2)*kernel.sigma) )-erf( (xi-kernel.Dp/2-x)/(sqrt(2)*kernel.sigma) ));
  end
  function phiDy(y::Float64,yi::Float64)
    return .5*(erf( (yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay) )-erf( (yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay) ));
  end
  function d1phiDx(x::Float64,xi::Float64)
    return -1/(sqrt(2*pi)*kernel.sigma)*(exp(-((xi+kernel.Dp/2-x)/(sqrt(2)*kernel.sigma))^2) - exp(-((xi-kernel.Dp/2-x)/(sqrt(2)*kernel.sigma))^2));
  end
  function d1phiDy(y::Float64,yi::Float64)
    return -1/(sqrt(2*pi)*kernel.sigmay)*(exp(-((yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay))^2) - exp(-((yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay))^2));
  end
  function d2phiDx(x::Float64,xi::Float64)
    return -1/(sqrt(pi)*kernel.sigma^2)*( ((xi+kernel.Dp/2-x)/(sqrt(2)*kernel.sigma))*exp(-((xi+kernel.Dp/2-x)/(sqrt(2)*kernel.sigma))^2) - ((xi-kernel.Dp/2-x)/(sqrt(2)*kernel.sigma))*exp(-((xi-kernel.Dp/2-x)/(sqrt(2)*kernel.sigma))^2) );
  end
  function d2phiDy(y::Float64,yi::Float64)
    return -1/(sqrt(pi)*kernel.sigmay^2)*( ((yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay))*exp(-((yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay))^2) - ((yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay))*exp(-((yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay))^2) );
  end

  function phix(x::Float64)
    phipx=zeros(kernel.Np);
    for i in 1:kernel.Np
      phipx[i]=phiDx(x,kernel.px[i]);
    end
    return phipx;
  end
  function phiy(y::Float64)
    phipy=zeros(kernel.Npy);
    for i in 1:kernel.Np
      phipy[i]=phiDy(y,kernel.py[i]);
    end
    return phipy;
  end
  Phix=phix.(kernel.grid[1]);
  Phiy=phiy.(kernel.grid[2]);

  #
  function d1phix(x::Float64)
    d1phipx=zeros(kernel.Np);
    for i in 1:kernel.Np
      d1phipx[i]=d1phiDx(x,kernel.px[i]);
    end
    return d1phipx;
  end
  function d1phiy(y::Float64)
    d1phipy=zeros(kernel.Npy);
    for i in 1:kernel.Np
      d1phipy[i]=d1phiDy(y,kernel.py[i]);
    end
    return d1phipy;
  end
  #

  #
  function d2phix(x::Float64)
    d2phipx=zeros(kernel.Np);
    for i in 1:kernel.Np
      d2phipx[i]=d2phiDx(x,kernel.px[i]);
    end
    return d2phipx;
  end
  function d2phiy(y::Float64)
    d2phipy=zeros(kernel.Npy);
    for i in 1:kernel.Np
      d2phipy[i]=d2phiDy(y,kernel.py[i]);
    end
    return d2phipy;
  end

  function phiVect(x::Array{Float64,1})
    phipx=phix(x[1]);
    phipy=phiy(x[2]);
    v=zeros(kernel.Np*kernel.Npy);
    l=1;
    for j in 1:kernel.Npy
      for i in 1:kernel.Np
        v[l]=phipx[i]*phipy[j];
        l+=1;
      end
    end
    return v;
  end
  function d1phiVect(m::Int64,x::Array{Float64,1})
    #
    phipx=phix(x[1]);
    phipy=phiy(x[2]);
    #
    d1phipx=d1phix(x[1]);
    d1phipy=d1phiy(x[2]);
    #
    if m==1
      v=zeros(kernel.Np*kernel.Npy);
      l=1;
      for j in 1:kernel.Npy
        for i in 1:kernel.Np
          v[l]=d1phipx[i]*phipy[j];
          l+=1;
        end
      end
      return v;
    else
      v=zeros(kernel.Np*kernel.Npy);
      l=1;
      for j in 1:kernel.Npy
        for i in 1:kernel.Np
          v[l]=phipx[i]*d1phipy[j];
          l+=1;
        end
      end
      return v;
    end
  end
  function d11phiVect(i::Int64,j::Int64,x::Array{Float64,1})
    #
    d1phipx=d1phix(x[1]);
    d1phipy=d1phiy(x[2]);
    #
    v=zeros(kernel.Np*kernel.Npy);
    l=1;
    for j in 1:kernel.Npy
      for i in 1:kernel.Np
        v[l]=d1phipx[i]*d1phipy[j];
        l+=1;
      end
    end
    return v;
  end
  function d2phiVect(m::Int64,x::Array{Float64,1})
    #
    phipx=phix(x[1]);
    phipy=phiy(x[2]);
    #
    d2phipx=d2phix(x[1]);
    d2phipy=d2phiy(x[2]);
    #
    if m==1
      v=zeros(kernel.Np*kernel.Npy);
      l=1;
      for j in 1:kernel.Npy
        for i in 1:kernel.Np
          v[l]=d2phipx[i]*phipy[j];
          l+=1;
        end
      end
      return v;
    else
      v=zeros(kernel.Np*kernel.Npy);
      l=1;
      for j in 1:kernel.Npy
        for i in 1:kernel.Np
          v[l]=phipx[i]*d2phipy[j];
          l+=1;
        end
      end
      return v;
    end
  end

  c(x1::Array{Float64,1},x2::Array{Float64,1})=vecdot(phiVect(x1),phiVect(x2));
  function d10c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return vecdot(d1phiVect(i,x1),phiVect(x2));
  end
  function d01c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return vecdot(phiVect(x1),d1phiVect(i,x2));
  end
  function d11c(i::Int64,j::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    if i==1 && j==2 || i==2 && j==1
      return vecdot(d1phiVect(1,x1),d1phiVect(1,x2));
    end
    if i==1 && j==3 || i==3 && j==1
      return vecdot(d11phiVect(1,2,x1),phiVect(x2));
    end
    if i==1 && j==4 || i==4 && j==1
      return vecdot(d1phiVect(1,x1),d1phiVect(2,x2));
    end

    if i==2 && j==3 || i==3 && j==2
      return vecdot(d1phiVect(2,x1),d1phiVect(1,x2));
    end
    if i==2 && j==4 || i==4 && j==2
      return vecdot(phiVect(x1),d11phiVect(1,2,x2));
    end

    if i==3 && j==4 || i==4 && j==3
      return vecdot(d1phiVect(2,x1),d1phiVect(2,x2));
    end
  end
  function d20c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return vecdot(d2phiVect(i,x1),phiVect(x2));
  end
  function d02c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return vecdot(phiVect(x1),d2phiVect(i,x2));
  end

  normObs=.5*norm(y)^2;

  PhisY=zeros(prod([length(kernel.grid[i]) for i in 1:2]));
  l=1;
  for i in 1:length(kernel.grid[1])
    for j in 1:length(kernel.grid[2])
      v=zeros(kernel.Np*kernel.Npy);
      lp=1;
      for jp in 1:kernel.Npy
        for ip in 1:kernel.Np
          v[lp]=Phix[i][ip]*Phiy[j][jp];
          lp+=1;
        end
      end
      PhisY[l]=vecdot(v,y);
      l+=1;
    end
  end

  function ob(x::Array{Float64,1},y::Array{Float64,1}=y)
    return vecdot(phiVect(x),y);
  end
  function d1ob(k::Int64,x::Array{Float64,1},y::Array{Float64,1}=y)
    return vecdot(d1phiVect(k,x),y);
  end
  function d11ob(k::Int64,l::Int64,x::Array{Float64,1},y::Array{Float64,1}=y)
    return vecdot(d11phiVect(k,l,x),y);
  end
  function d2ob(k::Int64,x::Array{Float64,1},y::Array{Float64,1}=y)
    return vecdot(d2phiVect(k,x),y);
  end

  function correl(x::Array{Float64,1},Phiu::Array{Array{Array{Float64,1},1},1})
    a=[vecdot(phix(x[1]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    b=[vecdot(phiy(x[2]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    return vecdot(a,b)-ob(x);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Array{Array{Float64,1},1},1})
    d1c=zeros(kernel.dim);
    a=[vecdot(phix(x[1]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    b=[vecdot(phiy(x[2]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    da=[vecdot(d1phix(x[1]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    db=[vecdot(d1phiy(x[2]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    d1c[1]=vecdot(da,b)-d1ob(1,x);
    d1c[2]=vecdot(a,db)-d1ob(2,x);
    return d1c
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Array{Array{Float64,1},1},1})
    d2c=zeros(kernel.dim,kernel.dim);
    a=[vecdot(phix(x[1]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    b=[vecdot(phiy(x[2]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    da=[vecdot(d1phix(x[1]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    db=[vecdot(d1phiy(x[2]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    dda=[vecdot(d2phix(x[1]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    ddb=[vecdot(d2phiy(x[2]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    d2c[1,2]=vecdot(da,db)-vecdot(d11phiVect(1,2,x),y);
    d2c=d2c+d2c';
    d2c[1,1]=vecdot(dda,b)-d2ob(1,x);
    d2c[2,2]=vecdot(a,ddb)-d2ob(2,x);
    return(d2c)
  end

  gaussconv(typeof(kernel),kernel.dim,kernel.sigma,kernel.sigmay,kernel.bounds,normObs,Phix,Phiy,PhisY,phix,phiy,phiVect,d1phiVect,d11phiVect,d2phiVect,y,c,d10c,d01c,d11c,d20c,d02c,ob,d1ob,d11ob,d2ob,correl,d1correl,d2correl);
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
    buffer=vecdot(op.Phix[i],Phiu)-op.PhisY[i];
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
