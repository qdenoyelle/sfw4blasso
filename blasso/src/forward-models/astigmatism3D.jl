mutable struct astigmatism3D <: discrete
  dim::Int64
  px::Array{Float64,1}
  py::Array{Float64,1}
  p::Array{Array{Float64,1},1}
  Npx::Int64
  Npy::Int64
  Dpx::Float64
  Dpy::Float64

  nbpointsgrid::Array{Int64,1}
  grid::Array{Array{Float64,1},1}

  sigmax::Function
  sigmay::Function
  d1sigmax::Function
  d1sigmay::Function
  d2sigmax::Function
  d2sigmay::Function
  bounds::Array{Array{Float64,1},1}
end

function setKernel(Npx::Int64,Npy::Int64,Dpx::Float64,Dpy::Float64,lamb::Float64,NA::Float64,fp::Float64,
  ni::Float64,bounds::Array{Array{Float64,1},1})

  px=collect(Dpx/2+range(0,stop=(Npx-1)*Dpx,length=Npx));
  py=collect(Dpy/2+range(0,stop=(Npy-1)*Dpy,length=Npy));

  dim=3;
  a,b=bounds[1],bounds[2];
  p=Array{Array{Float64,1}}(undef,0);
  for pyi in py
    for pxi in px
      append!(p,[[pxi,pyi]]);
    end
  end

  sigma0=2*.21*lamb/NA;
  alpha=-.79;
  beta=.2;
  d=(.5*lamb*ni)/NA^2;

  sigmax(z::Float64)=sigma0*sqrt(1+(alpha*(z-fp)-beta)^2/d^2);
  sigmay(z::Float64)=sigmax(-z+2*fp);
  d1sigmax(z::Float64)=sigma0/d^2*alpha*(alpha*(z-fp)-beta)*(1+(alpha*(z-fp)-beta)^2/d^2)^(-.5);
  d1sigmay(z::Float64)=-d1sigmax(-z+2*fp);
  d2sigmax(z::Float64)=alpha^2*sigma0/d^2*( (1+(alpha*(z-fp)-beta)^2/d^2)^(-.5) - 1/d^2*(alpha*(z-fp)-beta)^2*(1+(alpha*(z-fp)-beta)^2/d^2)^(-1.5) );
  d2sigmay(z::Float64)=d2sigmax(-z+2*fp);

  coeff=.25;
  lSample=[coeff/sigmax(fp),coeff/sigmay(fp),12];
  nbpointsgrid=[convert(Int64,round(5*lSample[i]*abs(b[i]-a[i]);digits=0)) for i in 1:3];
  g=Array{Array{Float64,1}}(undef,dim);
  for i in 1:3
    g[i]=collect(range(a[i],stop=b[i],length=nbpointsgrid[i]));
  end

  return astigmatism3D(dim,px,py,p,Npx,Npy,Dpx,Dpy,nbpointsgrid,g,sigmax,sigmay,d1sigmax,d1sigmay,d2sigmax,d2sigmay,bounds)
end

mutable struct gaussconv2DAstigmatism <: operator
    ker::DataType
    dim::Int64
    bounds::Array{Array{Float64,1},1}

    normObs::Float64

    Phix::Array{Array{Array{Float64,1},1},1}
    Phiy::Array{Array{Array{Float64,1},1},1}
    PhisY::Array{Float64,1}
    phix::Function
    phiy::Function
    phi::Function
    d1phi::Function
    d11phi::Function
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
    d11ob::Function
    d2ob::Function
    correl::Function
    d1correl::Function
    d2correl::Function
end

# Functions that set the operator for the gaussian convolution in 2D when the initial measure is given
function setoperator(kernel::astigmatism3D,a0::Array{Float64,1},x0::Array{Array{Float64,1},1})
  function phiDx(x::Float64,z::Float64,xi::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return .5*(erf(alphap/kernel.sigmax(z))-erf(alpham/kernel.sigmax(z)));
  end
  function phiDy(y::Float64,z::Float64,yi::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return .5*(erf(alphap/kernel.sigmay(z))-erf(alpham/kernel.sigmay(z)));
  end
  function d1xphiDx(x::Float64,z::Float64,xi::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmax(z))*(exp(-(alphap/kernel.sigmax(z))^2) - exp(-(alpham/kernel.sigmax(z))^2));
  end


  function d1yphiDy(y::Float64,z::Float64,yi::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmay(z))*(exp(-(alphap/kernel.sigmay(z))^2) - exp(-(alpham/kernel.sigmay(z))^2));
  end
  function d11xzphiDx(x::Float64,z::Float64,xi::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return kernel.d1sigmax(z)/(sqrt(2*pi)*kernel.sigmax(z)^2)*( (exp(-(alphap/kernel.sigmax(z))^2) - exp(-(alpham/kernel.sigmax(z))^2)) - 2/kernel.sigmax(z)^2*(alphap^2*exp(-(alphap/kernel.sigmax(z))^2)-alpham^2*exp(-(alpham/kernel.sigmax(z))^2)) );
  end
  function d11yzphiDy(y::Float64,z::Float64,yi::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return kernel.d1sigmay(z)/(sqrt(2*pi)*kernel.sigmay(z)^2)*( (exp(-(alphap/kernel.sigmay(z))^2) - exp(-(alpham/kernel.sigmay(z))^2)) - 2/kernel.sigmay(z)^2*(alphap^2*exp(-(alphap/kernel.sigmay(z))^2)-alpham^2*exp(-(alpham/kernel.sigmay(z))^2)) );
  end
  function d2xphiDx(x::Float64,z::Float64,xi::Float64)
    return -1/(sqrt(pi)*kernel.sigmax(z)^2)*( ((xi+kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))*exp(-((xi+kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))^2) - ((xi-kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))*exp(-((xi-kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))^2) );
  end
  function d2yphiDy(y::Float64,z::Float64,yi::Float64)
    return -1/(sqrt(pi)*kernel.sigmay(z)^2)*( ((yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))*exp(-((yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))^2) - ((yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))*exp(-((yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))^2) );
  end

  function d1zphiDx(x::Float64,z::Float64,xi::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return -kernel.d1sigmax(z)/(sqrt(pi)*kernel.sigmax(z)^2)*( alphap*exp(-(alphap/kernel.sigmax(z))^2) - alpham*exp(-(alpham/kernel.sigmax(z))^2) );
  end
  function d1zphiDy(y::Float64,z::Float64,yi::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return -kernel.d1sigmay(z)/(sqrt(pi)*kernel.sigmay(z)^2)*( alphap*exp(-(alphap/kernel.sigmay(z))^2) - alpham*exp(-(alpham/kernel.sigmay(z))^2) );
  end

  function d2zphiDx(x::Float64,z::Float64,xi::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return -1/(sqrt(pi)*kernel.sigmax(z)^2)*( alphap*exp(-(alphap/kernel.sigmax(z))^2) - alpham*exp(-(alpham/kernel.sigmax(z))^2) )*(kernel.d2sigmax(z)-2*kernel.d1sigmax(z)^2/kernel.sigmax(z)) - 2*kernel.d1sigmax(z)^2/(sqrt(pi)*kernel.sigmax(z)^5)*(alphap^3*exp(-(alphap/kernel.sigmax(z))^2)-alpham^3*exp(-(alpham/kernel.sigmax(z))^2));
  end
  function d2zphiDy(y::Float64,z::Float64,yi::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return -1/(sqrt(pi)*kernel.sigmay(z)^2)*( alphap*exp(-(alphap/kernel.sigmay(z))^2) - alpham*exp(-(alpham/kernel.sigmay(z))^2) )*(kernel.d2sigmay(z)-2*kernel.d1sigmay(z)^2/kernel.sigmay(z)) - 2*kernel.d1sigmay(z)^2/(sqrt(pi)*kernel.sigmay(z)^5)*(alphap^3*exp(-(alphap/kernel.sigmay(z))^2)-alpham^3*exp(-(alpham/kernel.sigmay(z))^2));
  end

  ##
  function phix(x::Float64,z::Float64)
    phipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      phipx[i]=phiDx(x,z,kernel.px[i]);
    end
    return phipx;
  end
  function phiy(y::Float64,z::Float64)
    phipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      phipy[i]=phiDy(y,z,kernel.py[i]);
    end
    return phipy;
  end

  Phix=Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3]);
  Phiy=Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3]);
  for k in 1:kernel.nbpointsgrid[3]
    Phix[k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[1]);
    Phiy[k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[2]);
    for i in 1:kernel.nbpointsgrid[1]
      Phix[k][i]=phix(kernel.grid[1][i],kernel.grid[3][k]);
    end
    for j in 1:kernel.nbpointsgrid[2]
      Phiy[k][j]=phiy(kernel.grid[2][j],kernel.grid[3][k]);
    end
  end

  #
  function d1xphix(x::Float64,z::Float64)
    d1xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1xphipx[i]=d1xphiDx(x,z,kernel.px[i]);
    end
    return d1xphipx;
  end
  function d1yphiy(y::Float64,z::Float64)
    d1yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1yphipy[i]=d1yphiDy(y,z,kernel.py[i]);
    end
    return d1yphipy;
  end
  function d1zphix(x::Float64,z::Float64)
    d1zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1zphipx[i]=d1zphiDx(x,z,kernel.px[i]);
    end
    return d1zphipx;
  end
  function d1zphiy(y::Float64,z::Float64)
    d1zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1zphipy[i]=d1zphiDy(y,z,kernel.py[i]);
    end
    return d1zphipy;
  end
  #

  #
  function d11xzphix(x::Float64,z::Float64)
    d11xzphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d11xzphipx[i]=d11xzphiDx(x,z,kernel.px[i]);
    end
    return d11xzphipx
  end
  function d11yzphiy(y::Float64,z::Float64)
    d11yzphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d11yzphipy[i]=d11yzphiDy(y,z,kernel.py[i]);
    end
    return d11yzphipy
  end
  #

  #
  function d2xphix(x::Float64,z::Float64)
    d2xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2xphipx[i]=d2xphiDx(x,z,kernel.px[i]);
    end
    return d2xphipx;
  end
  function d2yphiy(y::Float64,z::Float64)
    d2yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2yphipy[i]=d2yphiDy(y,z,kernel.py[i]);
    end
    return d2yphipy;
  end
  #

  #
  function d2zphix(x::Float64,z::Float64)
    d2zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2zphipx[i]=d2zphiDx(x,z,kernel.px[i]);
    end
    return d2zphipx;
  end
  function d2zphiy(y::Float64,z::Float64)
    d2zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2zphipy[i]=d2zphiDy(y,z,kernel.py[i]);
    end
    return d2zphipy;
  end
  #
  ##

  ##
  function phiVect(x::Array{Float64,1})
    phipx=phix(x[1],x[3]);
    phipy=phiy(x[2],x[3]);
    v=zeros(kernel.Npx*kernel.Npy);
    l=1;
    for j in 1:kernel.Npy
      for i in 1:kernel.Npx
        v[l]=phipx[i]*phipy[j];
        l+=1;
      end
    end
    return v;
  end
  function d1phiVect(m::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy);
    if m==1
      l=1;
      #
      phipy=phiy(x[2],x[3]);
      #
      d1xphipx=d1xphix(x[1],x[3]);
      #
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d1xphipx[i]*phipy[j];
          l+=1;
        end
      end
      return v;
    elseif m==2
      l=1;
      #
      phipx=phix(x[1],x[3]);
      #
      d1yphipy=d1yphiy(x[2],x[3]);
      #
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=phipx[i]*d1yphipy[j];
          l+=1;
        end
      end
      return v;
    else
      l=1;
      #
      phipx=phix(x[1],x[3]);
      phipy=phiy(x[2],x[3]);
      #
      d1zphipx=d1zphix(x[1],x[3]);
      d1zphipy=d1zphiy(x[2],x[3]);
      #
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d1zphipx[i]*phipy[j]+phipx[i]*d1zphipy[j];
          l+=1;
        end
      end
      return v;
    end
  end
  function d11phiVect(m::Int64,n::Int64,x::Array{Float64,1})
    #
    phipx=phix(x[1],x[3]);
    phipy=phiy(x[2],x[3]);
    #
    d1xphipx=d1xphix(x[1],x[3]);
    d1yphipy=d1yphiy(x[2],x[3]);
    d1zphipx=d1zphix(x[1],x[3]);
    d1zphipy=d1zphiy(x[2],x[3]);
    #
    d11xzphipx=d11xzphix(x[1],x[3]);
    d11yzphipy=d11yzphiy(x[2],x[3]);
    #
    v=zeros(kernel.Npx*kernel.Npy);
    l=1;
    if m==1 && n==2 || m==2 && n==1
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d1xphipx[i]*d1yphipy[j];
          l+=1;
        end
      end
      return v;
    elseif m==1 && n==3 || m==3 && n==1
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d11xzphipx[i]*phipy[j]+d1xphipx[i]*d1zphipy[j];
          l+=1;
        end
      end
      return v;
    elseif m==2 && n==3 || m==3 && n==2
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=phipx[i]*d11yzphipy[j]+d1zphipx[i]*d1yphipy[j];
          l+=1;
        end
      end
      return v;
    end
  end
  function d2phiVect(m::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy);
    l=1;
    if m==1
      #
      phipy=phiy(x[2],x[3]);
      #
      d2xphipx=d2xphix(x[1],x[3]);
      #
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d2xphipx[i]*phipy[j];
          l+=1;
        end
      end
      return v;
    elseif m==2
      #
      phipx=phix(x[1],x[3]);
      #
      d2yphipy=d2yphiy(x[2],x[3]);
      #
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=phipx[i]*d2yphipy[j];
          l+=1;
        end
      end
      return v;
    else
      #
      phipx=phix(x[1],x[3]);
      phipy=phiy(x[2],x[3]);
      #
      d1zphipx=d1zphix(x[1],x[3]);
      d1zphipy=d1zphiy(x[2],x[3]);
      #
      d2zphipx=d2zphix(x[1],x[3]);
      d2zphipy=d2zphiy(x[2],x[3]);
      #
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d2zphipx[i]*phipy[j]+phipx[i]*d2zphipy[j]+2*d1zphipx[i]*d1zphipy[j];
          l+=1;
        end
      end
      return v;
    end
  end

  c(x1::Array{Float64,1},x2::Array{Float64,1})=dot(phiVect(x1),phiVect(x2));
  function d10c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return dot(d1phiVect(i,x1),phiVect(x2));
  end
  function d01c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return dot(phiVect(x1),d1phiVect(i,x2));
  end
  function d11c(i::Int64,j::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    if i==1 && j==2 || i==2 && j==1
      return dot(d1phiVect(1,x1),d1phiVect(1,x2));
    end
    if i==1 && j==3 || i==3 && j==1
      return dot(d11phiVect(1,2,x1),phiVect(x2));
    end
    if i==1 && j==4 || i==4 && j==1
      return dot(d1phiVect(1,x1),d1phiVect(2,x2));
    end
    if i==1 && j==5 || i==5 && j==1
      return dot(d11phiVect(1,3,x1),phiVect(x2));
    end
    if i==1 && j==6 || i==6 && j==1
      return dot(d1phiVect(1,x1),d1phiVect(3,x2));
    end

    if i==2 && j==3 || i==3 && j==2
      return dot(d1phiVect(2,x1),d1phiVect(1,x2));
    end
    if i==2 && j==4 || i==4 && j==2
      return dot(phiVect(x1),d11phiVect(1,2,x2));
    end
    if i==2 && j==5 || i==5 && j==2
      return dot(d1phiVect(3,x1),d1phiVect(1,x2));
    end
    if i==2 && j==6 || i==6 && j==2
      return dot(phiVect(x1),d11phiVect(1,3,x2));
    end

    if i==3 && j==4 || i==4 && j==3
      return dot(d1phiVect(2,x1),d1phiVect(2,x2));
    end
    if i==3 && j==5 || i==5 && j==3
      return dot(d11phiVect(2,3,x1),phiVect(x2));
    end
    if i==3 && j==6 || i==6 && j==3
      return dot(d1phiVect(2,x1),d1phiVect(3,x2));
    end

    if i==4 && j==5 || i==5 && j==4
      return dot(d1phiVect(3,x1),d1phiVect(2,x2));
    end
    if i==4 && j==6 || i==6 && j==4
      return dot(phiVect(x1),d11phiVect(2,3,x2));
    end

    if i==5 && j==6 || i==6 && j==5
      return dot(d1phiVect(3,x1),d1phiVect(3,x2));
    end
  end
  function d20c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return dot(d2phiVect(i,x1),phiVect(x2));
  end
  function d02c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return dot(phiVect(x1),d2phiVect(i,x2));
  end

  y=sum([a0[i]*phiVect(x0[i]) for i in 1:length(x0)]);
  normObs=.5*norm(y)^2;

  PhisY=zeros(prod(kernel.nbpointsgrid));
  l=1;
  for k in 1:kernel.nbpointsgrid[3]
    for i in 1:kernel.nbpointsgrid[1]
      for j in 1:kernel.nbpointsgrid[2]
        v=zeros(kernel.Npx*kernel.Npy);
        lp=1;
        for jp in 1:kernel.Npy
          for ip in 1:kernel.Npx
            v[lp]=Phix[k][i][ip]*Phiy[k][j][jp];
            lp+=1;
          end
        end
        PhisY[l]=dot(v,y);
        l+=1;
      end
    end
  end

  function ob(x::Array{Float64,1},y::Array{Float64,1}=y)
    return dot(phiVect(x),y);
  end
  function d1ob(k::Int64,x::Array{Float64,1},y::Array{Float64,1}=y)
    return dot(d1phiVect(k,x),y);
  end
  function d11ob(k::Int64,l::Int64,x::Array{Float64,1},y::Array{Float64,1}=y)
    o=1;
    if k>=3 && k<=4
      o=2;
    end
    if k>=5
      o=3;
    end
    p=1;
    if l>=3 && l<=4
      p=2;
    end
    if l>=5
      p=3;
    end
    return dot(d11phiVect(o,p,x),y);
  end
  function d2ob(k::Int64,x::Array{Float64,1},y::Array{Float64,1}=y)
    return dot(d2phiVect(k,x),y);
  end


  function correl(x::Array{Float64,1},Phiu::Array{Array{Array{Float64,1},1},1})
    a=[dot(phix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    b=[dot(phiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    return dot(a,b)-ob(x);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Array{Array{Float64,1},1},1})
    d1c=zeros(kernel.dim);
    a=[dot(phix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    b=[dot(phiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    da=[dot(d1xphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    db=[dot(d1yphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    dc=[dot(d1zphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    dd=[dot(d1zphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    d1c[1]=dot(da,b)-d1ob(1,x);
    d1c[2]=dot(a,db)-d1ob(2,x);
    d1c[3]=dot(dc,b)+dot(a,dd)-d1ob(3,x);
    return d1c
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Array{Array{Float64,1},1},1})
    d2c=zeros(kernel.dim,kernel.dim);
    a=[dot(phix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    b=[dot(phiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    da=[dot(d1xphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    db=[dot(d1yphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    dc=[dot(d1zphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    dd=[dot(d1zphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    dac=[dot(d11xzphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    dbd=[dot(d11yzphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    dda=[dot(d2xphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    ddb=[dot(d2yphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    ddc=[dot(d2zphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    ddd=[dot(d2zphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];

    d2c[1,2]=dot(da,db)-dot(d11phiVect(1,2,x),y);
    d2c[1,3]=dot(da,dd)+dot(dac,b)-dot(d11phiVect(1,3,x),y);
    d2c[2,3]=dot(dc,db)+dot(a,dbd)-dot(d11phiVect(2,3,x),y);
    d2c=d2c+d2c';
    d2c[1,1]=dot(dda,b)-d2ob(1,x);
    d2c[2,2]=dot(a,ddb)-d2ob(2,x);
    d2c[3,3]=dot(ddc,b)+dot(a,ddd)+2*dot(dc,dd)-d2ob(3,x);
    return(d2c)
  end

  gaussconv2DAstigmatism(typeof(kernel),kernel.dim,kernel.bounds,normObs,Phix,Phiy,PhisY,phix,phiy,phiVect,d1phiVect,d11phiVect,d2phiVect,y,c,d10c,d01c,d11c,d20c,d02c,ob,d1ob,d11ob,d2ob,correl,d1correl,d2correl);
end

function setoperator(kernel::astigmatism3D,a0::Array{Float64,1},x0::Array{Array{Float64,1},1},w::Array{Float64,1})
  function phiDx(x::Float64,z::Float64,xi::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return .5*(erf(alphap/kernel.sigmax(z))-erf(alpham/kernel.sigmax(z)));
  end
  function phiDy(y::Float64,z::Float64,yi::Float64)
    return .5*(erf( (yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)) )-erf( (yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)) ));
  end
  function d1xphiDx(x::Float64,z::Float64,xi::Float64)
    return -1/(sqrt(2*pi)*kernel.sigmax(z))*(exp(-((xi+kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))^2) - exp(-((xi-kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))^2));
  end
  function d1yphiDy(y::Float64,z::Float64,yi::Float64)
    return -1/(sqrt(2*pi)*kernel.sigmay(z))*(exp(-((yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))^2) - exp(-((yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))^2));
  end
  function d11xzphiDx(x::Float64,z::Float64,xi::Float64)
    return -1/(sqrt(2*pi))*(kernel.d1sigmax(z)/kernel.sigmax(z)^2)*((1+((xi+kernel.Dpx/2-x)/kernel.sigmax(z))^2)*exp(-((xi+kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))^2) - (1+((xi-kernel.Dpx/2-x)/kernel.sigmax(z))^2)*exp(-((xi-kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))^2));
  end
  function d11yzphiDy(y::Float64,z::Float64,yi::Float64)
    return -1/(sqrt(2*pi))*(kernel.d1sigmay(z)/kernel.sigmay(z)^2)*((1+((yi+kernel.Dpy/2-y)/kernel.sigmay(z))^2)*exp(-((yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))^2) - (1+((yi-kernel.Dpy/2-y)/kernel.sigmay(z))^2)*exp(-((yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))^2));
  end
  function d2xphiDx(x::Float64,z::Float64,xi::Float64)
    return -1/(sqrt(pi)*kernel.sigmax(z)^2)*( ((xi+kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))*exp(-((xi+kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))^2) - ((xi-kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))*exp(-((xi-kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))^2) );
  end
  function d2yphiDy(y::Float64,z::Float64,yi::Float64)
    return -1/(sqrt(pi)*kernel.sigmay(z)^2)*( ((yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))*exp(-((yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))^2) - ((yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))*exp(-((yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))^2) );
  end

  function d1zphiDx(x::Float64,z::Float64,xi::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return -kernel.d1sigmax(z)/(sqrt(pi)*kernel.sigmax(z)^2)*( alphap*exp(-(alphap/kernel.sigmax(z))^2) - alpham*exp(-(alpham/kernel.sigmax(z))^2) );
  end
  function d1zphiDy(y::Float64,z::Float64,yi::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return -kernel.d1sigmay(z)/(sqrt(pi)*kernel.sigmay(z)^2)*( alphap*exp(-(alphap/kernel.sigmay(z))^2) - alpham*exp(-(alpham/kernel.sigmay(z))^2) );
  end

  function d2zphiDx(x::Float64,z::Float64,xi::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return -1/(sqrt(pi)*kernel.sigmax(z)^2)*( alphap*exp(-(alphap/kernel.sigmax(z))^2) - alpham*exp(-(alpham/kernel.sigmax(z))^2) )*(kernel.d2sigmax(z)-2*kernel.d1sigmax(z)^2/kernel.sigmax(z)) - 2*kernel.d1sigmax(z)^2/(sqrt(pi)*kernel.sigmax(z)^5)*(alphap^3*exp(-(alphap/kernel.sigmax(z))^2)-alpham^3*exp(-(alpham/kernel.sigmax(z))^2));
  end
  function d2zphiDy(y::Float64,z::Float64,yi::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return -1/(sqrt(pi)*kernel.sigmay(z)^2)*( alphap*exp(-(alphap/kernel.sigmay(z))^2) - alpham*exp(-(alpham/kernel.sigmay(z))^2) )*(kernel.d2sigmay(z)-2*kernel.d1sigmay(z)^2/kernel.sigmay(z)) - 2*kernel.d1sigmay(z)^2/(sqrt(pi)*kernel.sigmay(z)^5)*(alphap^3*exp(-(alphap/kernel.sigmay(z))^2)-alpham^3*exp(-(alpham/kernel.sigmay(z))^2));
  end

  ##
  function phix(x::Float64,z::Float64)
    phipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      phipx[i]=phiDx(x,z,kernel.px[i]);
    end
    return phipx;
  end
  function phiy(y::Float64,z::Float64)
    phipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      phipy[i]=phiDy(y,z,kernel.py[i]);
    end
    return phipy;
  end

  Phix=Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3]);
  Phiy=Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3]);
  for k in 1:kernel.nbpointsgrid[3]
    Phix[k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[1]);
    Phiy[k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[2]);
    for i in 1:kernel.nbpointsgrid[1]
      Phix[k][i]=phix(kernel.grid[1][i],kernel.grid[3][k]);
    end
    for j in 1:kernel.nbpointsgrid[2]
      Phiy[k][j]=phiy(kernel.grid[2][j],kernel.grid[3][k]);
    end
  end

  #
  function d1xphix(x::Float64,z::Float64)
    d1xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1xphipx[i]=d1xphiDx(x,z,kernel.px[i]);
    end
    return d1xphipx;
  end
  function d1yphiy(y::Float64,z::Float64)
    d1yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1yphipy[i]=d1yphiDy(y,z,kernel.py[i]);
    end
    return d1yphipy;
  end
  function d1zphix(x::Float64,z::Float64)
    d1zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1zphipx[i]=d1zphiDx(x,z,kernel.px[i]);
    end
    return d1zphipx;
  end
  function d1zphiy(y::Float64,z::Float64)
    d1zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1zphipy[i]=d1zphiDy(y,z,kernel.py[i]);
    end
    return d1zphipy;
  end
  #

  #
  function d11xzphix(x::Float64,z::Float64)
    d11xzphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d11xzphipx[i]=d11xzphiDx(x,z,kernel.px[i]);
    end
    return d11xzphipx
  end
  function d11yzphiy(y::Float64,z::Float64)
    d11yzphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d11yzphipy[i]=d11yzphiDy(y,z,kernel.py[i]);
    end
    return d11yzphipy
  end
  #

  #
  function d2xphix(x::Float64,z::Float64)
    d2xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2xphipx[i]=d2xphiDx(x,z,kernel.px[i]);
    end
    return d2xphipx;
  end
  function d2yphiy(y::Float64,z::Float64)
    d2yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2yphipy[i]=d2yphiDy(y,z,kernel.py[i]);
    end
    return d2yphipy;
  end
  #

  #
  function d2zphix(x::Float64,z::Float64)
    d2zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2zphipx[i]=d2zphiDx(x,z,kernel.px[i]);
    end
    return d2zphipx;
  end
  function d2zphiy(y::Float64,z::Float64)
    d2zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2zphipy[i]=d2zphiDy(y,z,kernel.py[i]);
    end
    return d2zphipy;
  end
  #
  ##

  ##
  function phiVect(x::Array{Float64,1})
    phipx=phix(x[1],x[3]);
    phipy=phiy(x[2],x[3]);
    v=zeros(kernel.Npx*kernel.Npy);
    l=1;
    for j in 1:kernel.Npy
      for i in 1:kernel.Npx
        v[l]=phipx[i]*phipy[j];
        l+=1;
      end
    end
    return v;
  end
  function d1phiVect(m::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy);
    if m==1
      l=1;
      #
      phipy=phiy(x[2],x[3]);
      #
      d1xphipx=d1xphix(x[1],x[3]);
      #
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d1xphipx[i]*phipy[j];
          l+=1;
        end
      end
      return v;
    elseif m==2
      l=1;
      #
      phipx=phix(x[1],x[3]);
      #
      d1yphipy=d1yphiy(x[2],x[3]);
      #
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=phipx[i]*d1yphipy[j];
          l+=1;
        end
      end
      return v;
    else
      l=1;
      #
      phipx=phix(x[1],x[3]);
      phipy=phiy(x[2],x[3]);
      #
      d1zphipx=d1zphix(x[1],x[3]);
      d1zphipy=d1zphiy(x[2],x[3]);
      #
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d1zphipx[i]*phipy[j]+phipx[i]*d1zphipy[j];
          l+=1;
        end
      end
      return v;
    end
  end
  function d11phiVect(m::Int64,n::Int64,x::Array{Float64,1})
    #
    phipx=phix(x[1],x[3]);
    phipy=phiy(x[2],x[3]);
    #
    d1xphipx=d1xphix(x[1],x[3]);
    d1yphipy=d1yphiy(x[2],x[3]);
    d1zphipx=d1zphix(x[1],x[3]);
    d1zphipy=d1zphiy(x[2],x[3]);
    #
    d11xzphipx=d11xzphix(x[1],x[3]);
    d11yzphipy=d11yzphiy(x[2],x[3]);
    #
    v=zeros(kernel.Npx*kernel.Npy);
    l=1;
    if m==1 && n==2 || m==2 && n==1
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d1xphipx[i]*d1yphipy[j];
          l+=1;
        end
      end
      return v;
    elseif m==1 && n==3 || m==3 && n==1
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d11xzphipx[i]*phipy[j]+d1xphipx[i]*d1zphipy[j];
          l+=1;
        end
      end
      return v;
    elseif m==2 && n==3 || m==3 && n==2
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=phipx[i]*d11yzphipy[j]+d1zphipx[i]*d1yphipy[j];
          l+=1;
        end
      end
      return v;
    end
  end
  function d2phiVect(m::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy);
    l=1;
    if m==1
      #
      phipy=phiy(x[2],x[3]);
      #
      d2xphipx=d2xphix(x[1],x[3]);
      #
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d2xphipx[i]*phipy[j];
          l+=1;
        end
      end
      return v;
    elseif m==2
      #
      phipx=phix(x[1],x[3]);
      #
      d2yphipy=d2yphiy(x[2],x[3]);
      #
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=phipx[i]*d2yphipy[j];
          l+=1;
        end
      end
      return v;
    else
      #
      phipx=phix(x[1],x[3]);
      phipy=phiy(x[2],x[3]);
      #
      d1zphipx=d1zphix(x[1],x[3]);
      d1zphipy=d1zphiy(x[2],x[3]);
      #
      d2zphipx=d2zphix(x[1],x[3]);
      d2zphipy=d2zphiy(x[2],x[3]);
      #
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d2zphipx[i]*phipy[j]+phipx[i]*d2zphipy[j]+2*d1zphipx[i]*d1zphipy[j];
          l+=1;
        end
      end
      return v;
    end
  end

  c(x1::Array{Float64,1},x2::Array{Float64,1})=dot(phiVect(x1),phiVect(x2));
  function d10c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return dot(d1phiVect(i,x1),phiVect(x2));
  end
  function d01c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return dot(phiVect(x1),d1phiVect(i,x2));
  end
  function d11c(i::Int64,j::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    if i==1 && j==2 || i==2 && j==1
      return dot(d1phiVect(1,x1),d1phiVect(1,x2));
    end
    if i==1 && j==3 || i==3 && j==1
      return dot(d11phiVect(1,2,x1),phiVect(x2));
    end
    if i==1 && j==4 || i==4 && j==1
      return dot(d1phiVect(1,x1),d1phiVect(2,x2));
    end
    if i==1 && j==5 || i==5 && j==1
      return dot(d11phiVect(1,3,x1),phiVect(x2));
    end
    if i==1 && j==6 || i==6 && j==1
      return dot(d1phiVect(1,x1),d1phiVect(3,x2));
    end

    if i==2 && j==3 || i==3 && j==2
      return dot(d1phiVect(2,x1),d1phiVect(1,x2));
    end
    if i==2 && j==4 || i==4 && j==2
      return dot(phiVect(x1),d11phiVect(1,2,x2));
    end
    if i==2 && j==5 || i==5 && j==2
      return dot(d1phiVect(3,x1),d1phiVect(1,x2));
    end
    if i==2 && j==6 || i==6 && j==2
      return dot(phiVect(x1),d11phiVect(1,3,x2));
    end

    if i==3 && j==4 || i==4 && j==3
      return dot(d1phiVect(2,x1),d1phiVect(2,x2));
    end
    if i==3 && j==5 || i==5 && j==3
      return dot(d11phiVect(2,3,x1),phiVect(x2));
    end
    if i==3 && j==6 || i==6 && j==3
      return dot(d1phiVect(2,x1),d1phiVect(3,x2));
    end

    if i==4 && j==5 || i==5 && j==4
      return dot(d1phiVect(3,x1),d1phiVect(2,x2));
    end
    if i==4 && j==6 || i==6 && j==4
      return dot(phiVect(x1),d11phiVect(2,3,x2));
    end

    if i==5 && j==6 || i==6 && j==5
      return dot(d1phiVect(3,x1),d1phiVect(3,x2));
    end
  end
  function d20c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return dot(d2phiVect(i,x1),phiVect(x2));
  end
  function d02c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return dot(phiVect(x1),d2phiVect(i,x2));
  end

  y=sum([a0[i]*phiVect(x0[i]) for i in 1:length(x0)])+w;
  normObs=.5*norm(y)^2;

  PhisY=zeros(prod(kernel.nbpointsgrid));
  l=1;
  for k in 1:kernel.nbpointsgrid[3]
    for i in 1:kernel.nbpointsgrid[1]
      for j in 1:kernel.nbpointsgrid[2]
        v=zeros(kernel.Npx*kernel.Npy);
        lp=1;
        for jp in 1:kernel.Npy
          for ip in 1:kernel.Npx
            v[lp]=Phix[k][i][ip]*Phiy[k][j][jp];
            lp+=1;
          end
        end
        PhisY[l]=dot(v,y);
        l+=1;
      end
    end
  end

  function ob(x::Array{Float64,1},y::Array{Float64,1}=y)
    return dot(phiVect(x),y);
  end
  function d1ob(k::Int64,x::Array{Float64,1},y::Array{Float64,1}=y)
    return dot(d1phiVect(k,x),y);
  end
  function d11ob(k::Int64,l::Int64,x::Array{Float64,1},y::Array{Float64,1}=y)
    o=1;
    if k>=3 && k<=4
      o=2;
    end
    if k>=5
      o=3;
    end
    p=1;
    if l>=3 && l<=4
      p=2;
    end
    if l>=5
      p=3;
    end
    return dot(d11phiVect(o,p,x),y);
  end
  function d2ob(k::Int64,x::Array{Float64,1},y::Array{Float64,1}=y)
    return dot(d2phiVect(k,x),y);
  end


  function correl(x::Array{Float64,1},Phiu::Array{Array{Array{Float64,1},1},1})
    a=[dot(phix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    b=[dot(phiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    return dot(a,b)-ob(x);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Array{Array{Float64,1},1},1})
    d1c=zeros(kernel.dim);
    a=[dot(phix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    b=[dot(phiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    da=[dot(d1xphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    db=[dot(d1yphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    dc=[dot(d1zphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    dd=[dot(d1zphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    d1c[1]=dot(da,b)-d1ob(1,x);
    d1c[2]=dot(a,db)-d1ob(2,x);
    d1c[3]=dot(dc,b)+dot(a,dd)-d1ob(3,x);
    return d1c
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Array{Array{Float64,1},1},1})
    d2c=zeros(kernel.dim,kernel.dim);
    a=[dot(phix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    b=[dot(phiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    da=[dot(d1xphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    db=[dot(d1yphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    dc=[dot(d1zphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    dd=[dot(d1zphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    dac=[dot(d11xzphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    dbd=[dot(d11yzphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    dda=[dot(d2xphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    ddb=[dot(d2yphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    ddc=[dot(d2zphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    ddd=[dot(d2zphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];

    d2c[1,2]=dot(da,db)-dot(d11phiVect(1,2,x),y);
    d2c[1,3]=dot(da,dd)+dot(dac,b)-dot(d11phiVect(1,3,x),y);
    d2c[2,3]=dot(dc,db)+dot(a,dbd)-dot(d11phiVect(2,3,x),y);
    d2c=d2c+d2c';
    d2c[1,1]=dot(dda,b)-d2ob(1,x);
    d2c[2,2]=dot(a,ddb)-d2ob(2,x);
    d2c[3,3]=dot(ddc,b)+dot(a,ddd)-d2ob(3,x);
    return(d2c)
  end

  gaussconv2DAstigmatism(typeof(kernel),kernel.dim,kernel.bounds,normObs,Phix,Phiy,PhisY,phix,phiy,phiVect,d1phiVect,d11phiVect,d2phiVect,y,c,d10c,d01c,d11c,d20c,d02c,ob,d1ob,d11ob,d2ob,correl,d1correl,d2correl);
end

function setoperator(kernel::astigmatism3D,a0::Array{Float64,1},x0::Array{Array{Float64,1},1},w::Array{Float64,1},nPhoton::Float64)
  function phiDx(x::Float64,z::Float64,xi::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return .5*(erf(alphap/kernel.sigmax(z))-erf(alpham/kernel.sigmax(z)));
  end
  function phiDy(y::Float64,z::Float64,yi::Float64)
    return .5*(erf( (yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)) )-erf( (yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)) ));
  end
  function d1xphiDx(x::Float64,z::Float64,xi::Float64)
    return -1/(sqrt(2*pi)*kernel.sigmax(z))*(exp(-((xi+kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))^2) - exp(-((xi-kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))^2));
  end
  function d1yphiDy(y::Float64,z::Float64,yi::Float64)
    return -1/(sqrt(2*pi)*kernel.sigmay(z))*(exp(-((yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))^2) - exp(-((yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))^2));
  end
  function d11xzphiDx(x::Float64,z::Float64,xi::Float64)
    return -1/(sqrt(2*pi))*(kernel.d1sigmax(z)/kernel.sigmax(z)^2)*((1+((xi+kernel.Dpx/2-x)/kernel.sigmax(z))^2)*exp(-((xi+kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))^2) - (1+((xi-kernel.Dpx/2-x)/kernel.sigmax(z))^2)*exp(-((xi-kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))^2));
  end
  function d11yzphiDy(y::Float64,z::Float64,yi::Float64)
    return -1/(sqrt(2*pi))*(kernel.d1sigmay(z)/kernel.sigmay(z)^2)*((1+((yi+kernel.Dpy/2-y)/kernel.sigmay(z))^2)*exp(-((yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))^2) - (1+((yi-kernel.Dpy/2-y)/kernel.sigmay(z))^2)*exp(-((yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))^2));
  end
  function d2xphiDx(x::Float64,z::Float64,xi::Float64)
    return -1/(sqrt(pi)*kernel.sigmax(z)^2)*( ((xi+kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))*exp(-((xi+kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))^2) - ((xi-kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))*exp(-((xi-kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))^2) );
  end
  function d2yphiDy(y::Float64,z::Float64,yi::Float64)
    return -1/(sqrt(pi)*kernel.sigmay(z)^2)*( ((yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))*exp(-((yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))^2) - ((yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))*exp(-((yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))^2) );
  end

  function d1zphiDx(x::Float64,z::Float64,xi::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return -kernel.d1sigmax(z)/(sqrt(pi)*kernel.sigmax(z)^2)*( alphap*exp(-(alphap/kernel.sigmax(z))^2) - alpham*exp(-(alpham/kernel.sigmax(z))^2) );
  end
  function d1zphiDy(y::Float64,z::Float64,yi::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return -kernel.d1sigmay(z)/(sqrt(pi)*kernel.sigmay(z)^2)*( alphap*exp(-(alphap/kernel.sigmay(z))^2) - alpham*exp(-(alpham/kernel.sigmay(z))^2) );
  end

  function d2zphiDx(x::Float64,z::Float64,xi::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return -1/(sqrt(pi)*kernel.sigmax(z)^2)*( alphap*exp(-(alphap/kernel.sigmax(z))^2) - alpham*exp(-(alpham/kernel.sigmax(z))^2) )*(kernel.d2sigmax(z)-2*kernel.d1sigmax(z)^2/kernel.sigmax(z)) - 2*kernel.d1sigmax(z)^2/(sqrt(pi)*kernel.sigmax(z)^5)*(alphap^3*exp(-(alphap/kernel.sigmax(z))^2)-alpham^3*exp(-(alpham/kernel.sigmax(z))^2));
  end
  function d2zphiDy(y::Float64,z::Float64,yi::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return -1/(sqrt(pi)*kernel.sigmay(z)^2)*( alphap*exp(-(alphap/kernel.sigmay(z))^2) - alpham*exp(-(alpham/kernel.sigmay(z))^2) )*(kernel.d2sigmay(z)-2*kernel.d1sigmay(z)^2/kernel.sigmay(z)) - 2*kernel.d1sigmay(z)^2/(sqrt(pi)*kernel.sigmay(z)^5)*(alphap^3*exp(-(alphap/kernel.sigmay(z))^2)-alpham^3*exp(-(alpham/kernel.sigmay(z))^2));
  end

  ##
  function phix(x::Float64,z::Float64)
    phipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      phipx[i]=phiDx(x,z,kernel.px[i]);
    end
    return phipx;
  end
  function phiy(y::Float64,z::Float64)
    phipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      phipy[i]=phiDy(y,z,kernel.py[i]);
    end
    return phipy;
  end

  Phix=Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3]);
  Phiy=Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3]);
  for k in 1:kernel.nbpointsgrid[3]
    Phix[k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[1]);
    Phiy[k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[2]);
    for i in 1:kernel.nbpointsgrid[1]
      Phix[k][i]=phix(kernel.grid[1][i],kernel.grid[3][k]);
    end
    for j in 1:kernel.nbpointsgrid[2]
      Phiy[k][j]=phiy(kernel.grid[2][j],kernel.grid[3][k]);
    end
  end

  #
  function d1xphix(x::Float64,z::Float64)
    d1xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1xphipx[i]=d1xphiDx(x,z,kernel.px[i]);
    end
    return d1xphipx;
  end
  function d1yphiy(y::Float64,z::Float64)
    d1yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1yphipy[i]=d1yphiDy(y,z,kernel.py[i]);
    end
    return d1yphipy;
  end
  function d1zphix(x::Float64,z::Float64)
    d1zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1zphipx[i]=d1zphiDx(x,z,kernel.px[i]);
    end
    return d1zphipx;
  end
  function d1zphiy(y::Float64,z::Float64)
    d1zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1zphipy[i]=d1zphiDy(y,z,kernel.py[i]);
    end
    return d1zphipy;
  end
  #

  #
  function d11xzphix(x::Float64,z::Float64)
    d11xzphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d11xzphipx[i]=d11xzphiDx(x,z,kernel.px[i]);
    end
    return d11xzphipx
  end
  function d11yzphiy(y::Float64,z::Float64)
    d11yzphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d11yzphipy[i]=d11yzphiDy(y,z,kernel.py[i]);
    end
    return d11yzphipy
  end
  #

  #
  function d2xphix(x::Float64,z::Float64)
    d2xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2xphipx[i]=d2xphiDx(x,z,kernel.px[i]);
    end
    return d2xphipx;
  end
  function d2yphiy(y::Float64,z::Float64)
    d2yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2yphipy[i]=d2yphiDy(y,z,kernel.py[i]);
    end
    return d2yphipy;
  end
  #

  #
  function d2zphix(x::Float64,z::Float64)
    d2zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2zphipx[i]=d2zphiDx(x,z,kernel.px[i]);
    end
    return d2zphipx;
  end
  function d2zphiy(y::Float64,z::Float64)
    d2zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2zphipy[i]=d2zphiDy(y,z,kernel.py[i]);
    end
    return d2zphipy;
  end
  #
  ##

  ##
  function phiVect(x::Array{Float64,1})
    phipx=phix(x[1],x[3]);
    phipy=phiy(x[2],x[3]);
    v=zeros(kernel.Npx*kernel.Npy);
    l=1;
    for j in 1:kernel.Npy
      for i in 1:kernel.Npx
        v[l]=phipx[i]*phipy[j];
        l+=1;
      end
    end
    return v;
  end
  function d1phiVect(m::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy);
    if m==1
      l=1;
      #
      phipy=phiy(x[2],x[3]);
      #
      d1xphipx=d1xphix(x[1],x[3]);
      #
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d1xphipx[i]*phipy[j];
          l+=1;
        end
      end
      return v;
    elseif m==2
      l=1;
      #
      phipx=phix(x[1],x[3]);
      #
      d1yphipy=d1yphiy(x[2],x[3]);
      #
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=phipx[i]*d1yphipy[j];
          l+=1;
        end
      end
      return v;
    else
      l=1;
      #
      phipx=phix(x[1],x[3]);
      phipy=phiy(x[2],x[3]);
      #
      d1zphipx=d1zphix(x[1],x[3]);
      d1zphipy=d1zphiy(x[2],x[3]);
      #
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d1zphipx[i]*phipy[j]+phipx[i]*d1zphipy[j];
          l+=1;
        end
      end
      return v;
    end
  end
  function d11phiVect(m::Int64,n::Int64,x::Array{Float64,1})
    #
    phipx=phix(x[1],x[3]);
    phipy=phiy(x[2],x[3]);
    #
    d1xphipx=d1xphix(x[1],x[3]);
    d1yphipy=d1yphiy(x[2],x[3]);
    d1zphipx=d1zphix(x[1],x[3]);
    d1zphipy=d1zphiy(x[2],x[3]);
    #
    d11xzphipx=d11xzphix(x[1],x[3]);
    d11yzphipy=d11yzphiy(x[2],x[3]);
    #
    v=zeros(kernel.Npx*kernel.Npy);
    l=1;
    if m==1 && n==2 || m==2 && n==1
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d1xphipx[i]*d1yphipy[j];
          l+=1;
        end
      end
      return v;
    elseif m==1 && n==3 || m==3 && n==1
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d11xzphipx[i]*phipy[j]+d1xphipx[i]*d1zphipy[j];
          l+=1;
        end
      end
      return v;
    elseif m==2 && n==3 || m==3 && n==2
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=phipx[i]*d11yzphipy[j]+d1zphipx[i]*d1yphipy[j];
          l+=1;
        end
      end
      return v;
    end
  end
  function d2phiVect(m::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy);
    l=1;
    if m==1
      #
      phipy=phiy(x[2],x[3]);
      #
      d2xphipx=d2xphix(x[1],x[3]);
      #
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d2xphipx[i]*phipy[j];
          l+=1;
        end
      end
      return v;
    elseif m==2
      #
      phipx=phix(x[1],x[3]);
      #
      d2yphipy=d2yphiy(x[2],x[3]);
      #
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=phipx[i]*d2yphipy[j];
          l+=1;
        end
      end
      return v;
    else
      #
      phipx=phix(x[1],x[3]);
      phipy=phiy(x[2],x[3]);
      #
      d1zphipx=d1zphix(x[1],x[3]);
      d1zphipy=d1zphiy(x[2],x[3]);
      #
      d2zphipx=d2zphix(x[1],x[3]);
      d2zphipy=d2zphiy(x[2],x[3]);
      #
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d2zphipx[i]*phipy[j]+phipx[i]*d2zphipy[j]+2*d1zphipx[i]*d1zphipy[j];
          l+=1;
        end
      end
      return v;
    end
  end

  c(x1::Array{Float64,1},x2::Array{Float64,1})=dot(phiVect(x1),phiVect(x2));
  function d10c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return dot(d1phiVect(i,x1),phiVect(x2));
  end
  function d01c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return dot(phiVect(x1),d1phiVect(i,x2));
  end
  function d11c(i::Int64,j::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    if i==1 && j==2 || i==2 && j==1
      return dot(d1phiVect(1,x1),d1phiVect(1,x2));
    end
    if i==1 && j==3 || i==3 && j==1
      return dot(d11phiVect(1,2,x1),phiVect(x2));
    end
    if i==1 && j==4 || i==4 && j==1
      return dot(d1phiVect(1,x1),d1phiVect(2,x2));
    end
    if i==1 && j==5 || i==5 && j==1
      return dot(d11phiVect(1,3,x1),phiVect(x2));
    end
    if i==1 && j==6 || i==6 && j==1
      return dot(d1phiVect(1,x1),d1phiVect(3,x2));
    end

    if i==2 && j==3 || i==3 && j==2
      return dot(d1phiVect(2,x1),d1phiVect(1,x2));
    end
    if i==2 && j==4 || i==4 && j==2
      return dot(phiVect(x1),d11phiVect(1,2,x2));
    end
    if i==2 && j==5 || i==5 && j==2
      return dot(d1phiVect(3,x1),d1phiVect(1,x2));
    end
    if i==2 && j==6 || i==6 && j==2
      return dot(phiVect(x1),d11phiVect(1,3,x2));
    end

    if i==3 && j==4 || i==4 && j==3
      return dot(d1phiVect(2,x1),d1phiVect(2,x2));
    end
    if i==3 && j==5 || i==5 && j==3
      return dot(d11phiVect(2,3,x1),phiVect(x2));
    end
    if i==3 && j==6 || i==6 && j==3
      return dot(d1phiVect(2,x1),d1phiVect(3,x2));
    end

    if i==4 && j==5 || i==5 && j==4
      return dot(d1phiVect(3,x1),d1phiVect(2,x2));
    end
    if i==4 && j==6 || i==6 && j==4
      return dot(phiVect(x1),d11phiVect(2,3,x2));
    end

    if i==5 && j==6 || i==6 && j==5
      return dot(d1phiVect(3,x1),d1phiVect(3,x2));
    end
  end
  function d20c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return dot(d2phiVect(i,x1),phiVect(x2));
  end
  function d02c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return dot(phiVect(x1),d2phiVect(i,x2));
  end

  y=sum([a0[i]*phiVect(x0[i]) for i in 1:length(x0)]);
  y=copy(blasso.poisson.((y./maximum(y)).*nPhoton)./nPhoton);
  y=y+w;
  normObs=.5*dot(y,y);

  PhisY=zeros(prod(kernel.nbpointsgrid));
  l=1;
  for k in 1:kernel.nbpointsgrid[3]
    for i in 1:kernel.nbpointsgrid[1]
      for j in 1:kernel.nbpointsgrid[2]
        v=zeros(kernel.Npx*kernel.Npy);
        lp=1;
        for jp in 1:kernel.Npy
          for ip in 1:kernel.Npx
            v[lp]=Phix[k][i][ip]*Phiy[k][j][jp];
            lp+=1;
          end
        end
        PhisY[l]=dot(v,y);
        l+=1;
      end
    end
  end

  function ob(x::Array{Float64,1},y::Array{Float64,1}=y)
    return dot(phiVect(x),y);
  end
  function d1ob(k::Int64,x::Array{Float64,1},y::Array{Float64,1}=y)
    return dot(d1phiVect(k,x),y);
  end
  function d11ob(k::Int64,l::Int64,x::Array{Float64,1},y::Array{Float64,1}=y)
    o=1;
    if k>=3 && k<=4
      o=2;
    end
    if k>=5
      o=3;
    end
    p=1;
    if l>=3 && l<=4
      p=2;
    end
    if l>=5
      p=3;
    end
    return dot(d11phiVect(o,p,x),y);
  end
  function d2ob(k::Int64,x::Array{Float64,1},y::Array{Float64,1}=y)
    return dot(d2phiVect(k,x),y);
  end


  function correl(x::Array{Float64,1},Phiu::Array{Array{Array{Float64,1},1},1})
    a=[dot(phix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    b=[dot(phiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    return dot(a,b)-ob(x);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Array{Array{Float64,1},1},1})
    d1c=zeros(kernel.dim);
    a=[dot(phix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    b=[dot(phiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    da=[dot(d1xphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    db=[dot(d1yphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    dc=[dot(d1zphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    dd=[dot(d1zphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    d1c[1]=dot(da,b)-d1ob(1,x);
    d1c[2]=dot(a,db)-d1ob(2,x);
    d1c[3]=dot(dc,b)+dot(a,dd)-d1ob(3,x);
    return d1c
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Array{Array{Float64,1},1},1})
    d2c=zeros(kernel.dim,kernel.dim);
    a=[dot(phix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    b=[dot(phiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    da=[dot(d1xphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    db=[dot(d1yphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    dc=[dot(d1zphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    dd=[dot(d1zphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    dac=[dot(d11xzphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    dbd=[dot(d11yzphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    dda=[dot(d2xphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    ddb=[dot(d2yphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    ddc=[dot(d2zphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    ddd=[dot(d2zphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];

    d2c[1,2]=dot(da,db)-dot(d11phiVect(1,2,x),y);
    d2c[1,3]=dot(da,dd)+dot(dac,b)-dot(d11phiVect(1,3,x),y);
    d2c[2,3]=dot(dc,db)+dot(a,dbd)-dot(d11phiVect(2,3,x),y);
    d2c=d2c+d2c';
    d2c[1,1]=dot(dda,b)-d2ob(1,x);
    d2c[2,2]=dot(a,ddb)-d2ob(2,x);
    d2c[3,3]=dot(ddc,b)+dot(a,ddd)-d2ob(3,x);
    return(d2c)
  end

  gaussconv2DAstigmatism(typeof(kernel),kernel.dim,kernel.bounds,normObs,Phix,Phiy,PhisY,phix,phiy,phiVect,d1phiVect,d11phiVect,d2phiVect,y,c,d10c,d01c,d11c,d20c,d02c,ob,d1ob,d11ob,d2ob,correl,d1correl,d2correl);
end

# gaussian convolution in 2D when only y is given
function setoperator(kernel::astigmatism3D,y::Array{Float64,1})
  function phiDx(x::Float64,z::Float64,xi::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return .5*(erf(alphap/kernel.sigmax(z))-erf(alpham/kernel.sigmax(z)));
  end
  function phiDy(y::Float64,z::Float64,yi::Float64)
    return .5*(erf( (yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)) )-erf( (yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)) ));
  end
  function d1xphiDx(x::Float64,z::Float64,xi::Float64)
    return -1/(sqrt(2*pi)*kernel.sigmax(z))*(exp(-((xi+kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))^2) - exp(-((xi-kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))^2));
  end
  function d1yphiDy(y::Float64,z::Float64,yi::Float64)
    return -1/(sqrt(2*pi)*kernel.sigmay(z))*(exp(-((yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))^2) - exp(-((yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))^2));
  end
  function d11xzphiDx(x::Float64,z::Float64,xi::Float64)
    return -1/(sqrt(2*pi))*(kernel.d1sigmax(z)/kernel.sigmax(z)^2)*((1+((xi+kernel.Dpx/2-x)/kernel.sigmax(z))^2)*exp(-((xi+kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))^2) - (1+((xi-kernel.Dpx/2-x)/kernel.sigmax(z))^2)*exp(-((xi-kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))^2));
  end
  function d11yzphiDy(y::Float64,z::Float64,yi::Float64)
    return -1/(sqrt(2*pi))*(kernel.d1sigmay(z)/kernel.sigmay(z)^2)*((1+((yi+kernel.Dpy/2-y)/kernel.sigmay(z))^2)*exp(-((yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))^2) - (1+((yi-kernel.Dpy/2-y)/kernel.sigmay(z))^2)*exp(-((yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))^2));
  end
  function d2xphiDx(x::Float64,z::Float64,xi::Float64)
    return -1/(sqrt(pi)*kernel.sigmax(z)^2)*( ((xi+kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))*exp(-((xi+kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))^2) - ((xi-kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))*exp(-((xi-kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z)))^2) );
  end
  function d2yphiDy(y::Float64,z::Float64,yi::Float64)
    return -1/(sqrt(pi)*kernel.sigmay(z)^2)*( ((yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))*exp(-((yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))^2) - ((yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))*exp(-((yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z)))^2) );
  end

  function d1zphiDx(x::Float64,z::Float64,xi::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return -kernel.d1sigmax(z)/(sqrt(pi)*kernel.sigmax(z)^2)*( alphap*exp(-(alphap/kernel.sigmax(z))^2) - alpham*exp(-(alpham/kernel.sigmax(z))^2) );
  end
  function d1zphiDy(y::Float64,z::Float64,yi::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return -kernel.d1sigmay(z)/(sqrt(pi)*kernel.sigmay(z)^2)*( alphap*exp(-(alphap/kernel.sigmay(z))^2) - alpham*exp(-(alpham/kernel.sigmay(z))^2) );
  end

  function d2zphiDx(x::Float64,z::Float64,xi::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return -1/(sqrt(pi)*kernel.sigmax(z)^2)*( alphap*exp(-(alphap/kernel.sigmax(z))^2) - alpham*exp(-(alpham/kernel.sigmax(z))^2) )*(kernel.d2sigmax(z)-2*kernel.d1sigmax(z)^2/kernel.sigmax(z)) - 2*kernel.d1sigmax(z)^2/(sqrt(pi)*kernel.sigmax(z)^5)*(alphap^3*exp(-(alphap/kernel.sigmax(z))^2)-alpham^3*exp(-(alpham/kernel.sigmax(z))^2));
  end
  function d2zphiDy(y::Float64,z::Float64,yi::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return -1/(sqrt(pi)*kernel.sigmay(z)^2)*( alphap*exp(-(alphap/kernel.sigmay(z))^2) - alpham*exp(-(alpham/kernel.sigmay(z))^2) )*(kernel.d2sigmay(z)-2*kernel.d1sigmay(z)^2/kernel.sigmay(z)) - 2*kernel.d1sigmay(z)^2/(sqrt(pi)*kernel.sigmay(z)^5)*(alphap^3*exp(-(alphap/kernel.sigmay(z))^2)-alpham^3*exp(-(alpham/kernel.sigmay(z))^2));
  end

  ##
  function phix(x::Float64,z::Float64)
    phipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      phipx[i]=phiDx(x,z,kernel.px[i]);
    end
    return phipx;
  end
  function phiy(y::Float64,z::Float64)
    phipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      phipy[i]=phiDy(y,z,kernel.py[i]);
    end
    return phipy;
  end

  Phix=Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3]);
  Phiy=Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3]);
  for k in 1:kernel.nbpointsgrid[3]
    Phix[k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[1]);
    Phiy[k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[2]);
    for i in 1:kernel.nbpointsgrid[1]
      Phix[k][i]=phix(kernel.grid[1][i],kernel.grid[3][k]);
    end
    for j in 1:kernel.nbpointsgrid[2]
      Phiy[k][j]=phiy(kernel.grid[2][j],kernel.grid[3][k]);
    end
  end

  #
  function d1xphix(x::Float64,z::Float64)
    d1xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1xphipx[i]=d1xphiDx(x,z,kernel.px[i]);
    end
    return d1xphipx;
  end
  function d1yphiy(y::Float64,z::Float64)
    d1yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1yphipy[i]=d1yphiDy(y,z,kernel.py[i]);
    end
    return d1yphipy;
  end
  function d1zphix(x::Float64,z::Float64)
    d1zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1zphipx[i]=d1zphiDx(x,z,kernel.px[i]);
    end
    return d1zphipx;
  end
  function d1zphiy(y::Float64,z::Float64)
    d1zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1zphipy[i]=d1zphiDy(y,z,kernel.py[i]);
    end
    return d1zphipy;
  end
  #

  #
  function d11xzphix(x::Float64,z::Float64)
    d11xzphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d11xzphipx[i]=d11xzphiDx(x,z,kernel.px[i]);
    end
    return d11xzphipx
  end
  function d11yzphiy(y::Float64,z::Float64)
    d11yzphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d11yzphipy[i]=d11yzphiDy(y,z,kernel.py[i]);
    end
    return d11yzphipy
  end
  #

  #
  function d2xphix(x::Float64,z::Float64)
    d2xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2xphipx[i]=d2xphiDx(x,z,kernel.px[i]);
    end
    return d2xphipx;
  end
  function d2yphiy(y::Float64,z::Float64)
    d2yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2yphipy[i]=d2yphiDy(y,z,kernel.py[i]);
    end
    return d2yphipy;
  end
  #

  #
  function d2zphix(x::Float64,z::Float64)
    d2zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2zphipx[i]=d2zphiDx(x,z,kernel.px[i]);
    end
    return d2zphipx;
  end
  function d2zphiy(y::Float64,z::Float64)
    d2zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2zphipy[i]=d2zphiDy(y,z,kernel.py[i]);
    end
    return d2zphipy;
  end
  #
  ##

  ##
  function phiVect(x::Array{Float64,1})
    phipx=phix(x[1],x[3]);
    phipy=phiy(x[2],x[3]);
    v=zeros(kernel.Npx*kernel.Npy);
    l=1;
    for j in 1:kernel.Npy
      for i in 1:kernel.Npx
        v[l]=phipx[i]*phipy[j];
        l+=1;
      end
    end
    return v;
  end
  function d1phiVect(m::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy);
    if m==1
      l=1;
      #
      phipy=phiy(x[2],x[3]);
      #
      d1xphipx=d1xphix(x[1],x[3]);
      #
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d1xphipx[i]*phipy[j];
          l+=1;
        end
      end
      return v;
    elseif m==2
      l=1;
      #
      phipx=phix(x[1],x[3]);
      #
      d1yphipy=d1yphiy(x[2],x[3]);
      #
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=phipx[i]*d1yphipy[j];
          l+=1;
        end
      end
      return v;
    else
      l=1;
      #
      phipx=phix(x[1],x[3]);
      phipy=phiy(x[2],x[3]);
      #
      d1zphipx=d1zphix(x[1],x[3]);
      d1zphipy=d1zphiy(x[2],x[3]);
      #
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d1zphipx[i]*phipy[j]+phipx[i]*d1zphipy[j];
          l+=1;
        end
      end
      return v;
    end
  end
  function d11phiVect(m::Int64,n::Int64,x::Array{Float64,1})
    #
    phipx=phix(x[1],x[3]);
    phipy=phiy(x[2],x[3]);
    #
    d1xphipx=d1xphix(x[1],x[3]);
    d1yphipy=d1yphiy(x[2],x[3]);
    d1zphipx=d1zphix(x[1],x[3]);
    d1zphipy=d1zphiy(x[2],x[3]);
    #
    d11xzphipx=d11xzphix(x[1],x[3]);
    d11yzphipy=d11yzphiy(x[2],x[3]);
    #
    v=zeros(kernel.Npx*kernel.Npy);
    l=1;
    if m==1 && n==2 || m==2 && n==1
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d1xphipx[i]*d1yphipy[j];
          l+=1;
        end
      end
      return v;
    elseif m==1 && n==3 || m==3 && n==1
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d11xzphipx[i]*phipy[j]+d1xphipx[i]*d1zphipy[j];
          l+=1;
        end
      end
      return v;
    elseif m==2 && n==3 || m==3 && n==2
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=phipx[i]*d11yzphipy[j]+d1zphipx[i]*d1yphipy[j];
          l+=1;
        end
      end
      return v;
    end
  end
  function d2phiVect(m::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy);
    l=1;
    if m==1
      #
      phipy=phiy(x[2],x[3]);
      #
      d2xphipx=d2xphix(x[1],x[3]);
      #
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d2xphipx[i]*phipy[j];
          l+=1;
        end
      end
      return v;
    elseif m==2
      #
      phipx=phix(x[1],x[3]);
      #
      d2yphipy=d2yphiy(x[2],x[3]);
      #
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=phipx[i]*d2yphipy[j];
          l+=1;
        end
      end
      return v;
    else
      #
      phipx=phix(x[1],x[3]);
      phipy=phiy(x[2],x[3]);
      #
      d1zphipx=d1zphix(x[1],x[3]);
      d1zphipy=d1zphiy(x[2],x[3]);
      #
      d2zphipx=d2zphix(x[1],x[3]);
      d2zphipy=d2zphiy(x[2],x[3]);
      #
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d2zphipx[i]*phipy[j]+phipx[i]*d2zphipy[j]+2*d1zphipx[i]*d1zphipy[j];
          l+=1;
        end
      end
      return v;
    end
  end

  c(x1::Array{Float64,1},x2::Array{Float64,1})=dot(phiVect(x1),phiVect(x2));
  function d10c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return dot(d1phiVect(i,x1),phiVect(x2));
  end
  function d01c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return dot(phiVect(x1),d1phiVect(i,x2));
  end
  function d11c(i::Int64,j::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    if i==1 && j==2 || i==2 && j==1
      return dot(d1phiVect(1,x1),d1phiVect(1,x2));
    end
    if i==1 && j==3 || i==3 && j==1
      return dot(d11phiVect(1,2,x1),phiVect(x2));
    end
    if i==1 && j==4 || i==4 && j==1
      return dot(d1phiVect(1,x1),d1phiVect(2,x2));
    end
    if i==1 && j==5 || i==5 && j==1
      return dot(d11phiVect(1,3,x1),phiVect(x2));
    end
    if i==1 && j==6 || i==6 && j==1
      return dot(d1phiVect(1,x1),d1phiVect(3,x2));
    end

    if i==2 && j==3 || i==3 && j==2
      return dot(d1phiVect(2,x1),d1phiVect(1,x2));
    end
    if i==2 && j==4 || i==4 && j==2
      return dot(phiVect(x1),d11phiVect(1,2,x2));
    end
    if i==2 && j==5 || i==5 && j==2
      return dot(d1phiVect(3,x1),d1phiVect(1,x2));
    end
    if i==2 && j==6 || i==6 && j==2
      return dot(phiVect(x1),d11phiVect(1,3,x2));
    end

    if i==3 && j==4 || i==4 && j==3
      return dot(d1phiVect(2,x1),d1phiVect(2,x2));
    end
    if i==3 && j==5 || i==5 && j==3
      return dot(d11phiVect(2,3,x1),phiVect(x2));
    end
    if i==3 && j==6 || i==6 && j==3
      return dot(d1phiVect(2,x1),d1phiVect(3,x2));
    end

    if i==4 && j==5 || i==5 && j==4
      return dot(d1phiVect(3,x1),d1phiVect(2,x2));
    end
    if i==4 && j==6 || i==6 && j==4
      return dot(phiVect(x1),d11phiVect(2,3,x2));
    end

    if i==5 && j==6 || i==6 && j==5
      return dot(d1phiVect(3,x1),d1phiVect(3,x2));
    end
  end
  function d20c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return dot(d2phiVect(i,x1),phiVect(x2));
  end
  function d02c(i::Int64,x1::Array{Float64,1},x2::Array{Float64,1})
    return dot(phiVect(x1),d2phiVect(i,x2));
  end

  normObs=.5*norm(y)^2;

  PhisY=zeros(prod(kernel.nbpointsgrid));
  l=1;
  for k in 1:kernel.nbpointsgrid[3]
    for i in 1:kernel.nbpointsgrid[1]
      for j in 1:kernel.nbpointsgrid[2]
        v=zeros(kernel.Npx*kernel.Npy);
        lp=1;
        for jp in 1:kernel.Npy
          for ip in 1:kernel.Npx
            v[lp]=Phix[k][i][ip]*Phiy[k][j][jp];
            lp+=1;
          end
        end
        PhisY[l]=dot(v,y);
        l+=1;
      end
    end
  end

  function ob(x::Array{Float64,1},y::Array{Float64,1}=y)
    return dot(phiVect(x),y);
  end
  function d1ob(k::Int64,x::Array{Float64,1},y::Array{Float64,1}=y)
    return dot(d1phiVect(k,x),y);
  end
  function d11ob(k::Int64,l::Int64,x::Array{Float64,1},y::Array{Float64,1}=y)
    o=1;
    if k>=3 && k<=4
      o=2;
    end
    if k>=5
      o=3;
    end
    p=1;
    if l>=3 && l<=4
      p=2;
    end
    if l>=5
      p=3;
    end
    return dot(d11phiVect(o,p,x),y);
  end
  function d2ob(k::Int64,x::Array{Float64,1},y::Array{Float64,1}=y)
    return dot(d2phiVect(k,x),y);
  end


  function correl(x::Array{Float64,1},Phiu::Array{Array{Array{Float64,1},1},1})
    a=[dot(phix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    b=[dot(phiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    return dot(a,b)-ob(x);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Array{Array{Float64,1},1},1})
    d1c=zeros(kernel.dim);
    a=[dot(phix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    b=[dot(phiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    da=[dot(d1xphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    db=[dot(d1yphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    dc=[dot(d1zphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    dd=[dot(d1zphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    d1c[1]=dot(da,b)-d1ob(1,x);
    d1c[2]=dot(a,db)-d1ob(2,x);
    d1c[3]=dot(dc,b)+dot(a,dd)-d1ob(3,x);
    return d1c
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Array{Array{Float64,1},1},1})
    d2c=zeros(kernel.dim,kernel.dim);
    a=[dot(phix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    b=[dot(phiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    da=[dot(d1xphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    db=[dot(d1yphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    dc=[dot(d1zphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    dd=[dot(d1zphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    dac=[dot(d11xzphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    dbd=[dot(d11yzphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    dda=[dot(d2xphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    ddb=[dot(d2yphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];
    ddc=[dot(d2zphix(x[1],x[3]),Phiu[1][i]) for i in 1:length(Phiu[1])];
    ddd=[dot(d2zphiy(x[2],x[3]),Phiu[2][i]) for i in 1:length(Phiu[2])];

    d2c[1,2]=dot(da,db)-dot(d11phiVect(1,2,x),y);
    d2c[1,3]=dot(da,dd)+dot(dac,b)-dot(d11phiVect(1,3,x),y);
    d2c[2,3]=dot(dc,db)+dot(a,dbd)-dot(d11phiVect(2,3,x),y);
    d2c=d2c+d2c';
    d2c[1,1]=dot(dda,b)-d2ob(1,x);
    d2c[2,2]=dot(a,ddb)-d2ob(2,x);
    d2c[3,3]=dot(ddc,b)+dot(a,ddd)-d2ob(3,x);
    return(d2c)
  end

  gaussconv2DAstigmatism(typeof(kernel),kernel.dim,kernel.bounds,normObs,Phix,Phiy,PhisY,phix,phiy,phiVect,d1phiVect,d11phiVect,d2phiVect,y,c,d10c,d01c,d11c,d20c,d02c,ob,d1ob,d11ob,d2ob,correl,d1correl,d2correl);
end

function computePhiu(u::Array{Float64,1},op::blasso.gaussconv2DAstigmatism)
  a,X=blasso.decompAmpPos(u,d=op.dim);
  Phiux=[a[i]*op.phix(X[i][1],X[i][3]) for i in 1:length(a)];
  Phiuy=[op.phiy(X[i][2],X[i][3]) for i in 1:length(a)];
  return [Phiux,Phiuy];
end

# Compute the argmin and min of the correl on the grid.
function minCorrelOnGrid(Phiu::Array{Array{Array{Float64,1},1},1},kernel::blasso.astigmatism3D,op::blasso.operator,positivity::Bool=true)
  correl_min,argmin=Inf,zeros(op.dim);
  l=1;
  for k in 1:kernel.nbpointsgrid[3]
    for i in 1:kernel.nbpointsgrid[1]
      a=[dot(op.Phix[k][i],Phiu[1][m]) for m in 1:length(Phiu[1])];
      for j in 1:kernel.nbpointsgrid[2]
        b=[dot(op.Phiy[k][j],Phiu[2][m]) for m in 1:length(Phiu[2])];
        buffer=dot(a,b)-op.PhisY[l];
        if !positivity
          buffer=-abs(buffer);
        end
        if buffer<correl_min
          correl_min=buffer;
          argmin=[kernel.grid[1][i],kernel.grid[2][j],kernel.grid[3][k]];
        end
        l+=1;
      end
    end
  end

  return argmin,correl_min
end

function setbounds(op::blasso.gaussconv2DAstigmatism,positivity::Bool=true,ampbounds::Bool=true)
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
