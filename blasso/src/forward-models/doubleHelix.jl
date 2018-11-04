mutable struct doubleHelix <: discrete
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

  sigmax::Float64
  sigmay::Float64
  rotx::Function
  roty::Function
  d1rotx::Function
  d1roty::Function
  d2rotx::Function
  d2roty::Function

  bounds::Array{Array{Float64,1},1}
end

function setKernel(Npx::Int64,Npy::Int64,Dpx::Float64,Dpy::Float64,fp::Float64,wdth::Float64,sigma::Array{Float64,1},bounds::Array{Array{Float64,1},1})
  sigmax,sigmay=sigma[1],sigma[2];

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

  theta(z::Float64)=pi/3*(z-fp)/(bounds[2][end]-fp);
  d1theta(z::Float64)=(pi/3)/(bounds[2][end]-fp);
  d2theta(z::Float64)=0.0;

  rotx(z::Float64)=wdth/2*cos(theta(z));
  roty(z::Float64)=-wdth/2*sin(theta(z));
  d1rotx(z::Float64)=-wdth/2*d1theta(z)*sin(theta(z));
  d1roty(z::Float64)=-wdth/2*d1theta(z)*cos(theta(z));
  d2rotx(z::Float64)=-wdth/2*d1theta(z)^2*cos(theta(z));
  d2roty(z::Float64)=wdth/2*d1theta(z)^2*sin(theta(z));

  coeff=.25;
  lSample=[coeff/sigmax,coeff/sigmay,12];#8
  nbpointsgrid=[convert(Int64,round(5*lSample[i]*abs(b[i]-a[i]);digits=0)) for i in 1:3];
  g=Array{Array{Float64,1}}(undef,dim);
  for i in 1:3
    g[i]=collect(range(a[i],stop=b[i],length=nbpointsgrid[i]));
  end

  return doubleHelix(dim,px,py,p,Npx,Npy,Dpx,Dpy,nbpointsgrid,g,sigmax,sigmay,rotx,roty,d1rotx,d1roty,d2rotx,d2roty,bounds)
end

mutable struct gaussconv2DDoubleHelix <: operator
    ker::DataType
    dim::Int64
    bounds::Array{Array{Float64,1},1}

    normObs::Float64

    Phix::Array{Array{Array{Array{Float64,1},1},1},1}
    Phiy::Array{Array{Array{Array{Float64,1},1},1},1}
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
function setoperator(kernel::doubleHelix,a0::Array{Float64,1},x0::Array{Array{Float64,1},1})
  function phiDx(x::Float64,z::Float64,xi::Float64,s::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2);
    return .5*(erf(alphap/kernel.sigmax)-erf(alpham/kernel.sigmax));
  end
  function phiDy(y::Float64,z::Float64,yi::Float64,s::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2);
    return .5*(erf(alphap/kernel.sigmay)-erf(alpham/kernel.sigmay));
  end

  function d1xphiDx(x::Float64,z::Float64,xi::Float64,s::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmax)*(exp(-(alphap/kernel.sigmax)^2) - exp(-(alpham/kernel.sigmax)^2));
  end
  function d1yphiDy(y::Float64,z::Float64,yi::Float64,s::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmay)*(exp(-(alphap/kernel.sigmay)^2) - exp(-(alpham/kernel.sigmay)^2));
  end

  function d11xzphiDx(x::Float64,z::Float64,xi::Float64,s::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2);
    return -s*kernel.d1rotx(z)/(sqrt(pi)*kernel.sigmax^3)*(alphap*exp(-(alphap/kernel.sigmax)^2) - alpham*exp(-(alpham/kernel.sigmax)^2));
  end
  function d11yzphiDy(y::Float64,z::Float64,yi::Float64,s::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2);
    return -s*kernel.d1roty(z)/(sqrt(pi)*kernel.sigmay^3)*(alphap*exp(-(alphap/kernel.sigmay)^2) - alpham*exp(-(alpham/kernel.sigmay)^2));
  end

  function d2xphiDx(x::Float64,z::Float64,xi::Float64,s::Float64)
    return -1/(sqrt(pi)*kernel.sigmax^2)*( ((xi+kernel.Dpx/2-x-s*kernel.rotx(z))/(sqrt(2)*kernel.sigmax))*exp(-((xi+kernel.Dpx/2-x-s*kernel.rotx(z))/(sqrt(2)*kernel.sigmax))^2) - ((xi-kernel.Dpx/2-x-s*kernel.rotx(z))/(sqrt(2)*kernel.sigmax))*exp(-((xi-kernel.Dpx/2-x-s*kernel.rotx(z))/(sqrt(2)*kernel.sigmax))^2) );
  end
  function d2yphiDy(y::Float64,z::Float64,yi::Float64,s::Float64)
    return -1/(sqrt(pi)*kernel.sigmay^2)*( ((yi+kernel.Dpy/2-y-s*kernel.roty(z))/(sqrt(2)*kernel.sigmay))*exp(-((yi+kernel.Dpy/2-y-s*kernel.roty(z))/(sqrt(2)*kernel.sigmay))^2) - ((yi-kernel.Dpy/2-y-s*kernel.roty(z))/(sqrt(2)*kernel.sigmay))*exp(-((yi-kernel.Dpy/2-y-s*kernel.roty(z))/(sqrt(2)*kernel.sigmay))^2) );
  end

  function d1zphiDx(x::Float64,z::Float64,xi::Float64,s::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2);
    return -s*kernel.d1rotx(z)/(sqrt(2*pi)*kernel.sigmax)*(exp(-(alphap/kernel.sigmax)^2) - exp(-(alpham/kernel.sigmax)^2));
  end
  function d1zphiDy(y::Float64,z::Float64,yi::Float64,s::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2);
    return -s*kernel.d1roty(z)/(sqrt(2*pi)*kernel.sigmay)*(exp(-(alphap/kernel.sigmay)^2) - exp(-(alpham/kernel.sigmay)^2));
  end

  function d2zphiDx(x::Float64,z::Float64,xi::Float64,s::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2);
    return -s*kernel.d2rotx(z)/(sqrt(2*pi)*kernel.sigmax)*(exp(-(alphap/kernel.sigmax)^2) - exp(-(alpham/kernel.sigmax)^2)) - s^2*kernel.d1rotx(z)^2/(sqrt(pi)*kernel.sigmax^3)*(alphap*exp(-(alphap/kernel.sigmax)^2) - alpham*exp(-(alpham/kernel.sigmax)^2));
  end
  function d2zphiDy(y::Float64,z::Float64,yi::Float64,s::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2);
    return -s*kernel.d2roty(z)/(sqrt(2*pi)*kernel.sigmay)*(exp(-(alphap/kernel.sigmay)^2) - exp(-(alpham/kernel.sigmay)^2)) - s^2*kernel.d1roty(z)^2/(sqrt(pi)*kernel.sigmay^3)*(alphap*exp(-(alphap/kernel.sigmay)^2) - alpham*exp(-(alpham/kernel.sigmay)^2));
  end
  ##

  ##
  function phix(x::Float64,z::Float64,s::Float64)
    phipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      phipx[i]=phiDx(x,z,kernel.px[i],s);
    end
    return phipx;
  end
  #
  function phiy(y::Float64,z::Float64,s::Float64)
    phipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      phipy[i]=phiDy(y,z,kernel.py[i],s);
    end
    return phipy;
  end
  #

  Phix=[Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3]),Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3])];
  Phiy=[Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3]),Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3])];
  for k in 1:kernel.nbpointsgrid[3]
    Phix[1][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[1]);
    Phiy[1][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[2]);
    Phix[2][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[1]);
    Phiy[2][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[2]);
    for i in 1:kernel.nbpointsgrid[1]
      Phix[1][k][i]=phix(kernel.grid[1][i],kernel.grid[3][k],1.0);
      Phix[2][k][i]=phix(kernel.grid[1][i],kernel.grid[3][k],-1.0);
    end
    for j in 1:kernel.nbpointsgrid[2]
      Phiy[1][k][j]=phiy(kernel.grid[2][j],kernel.grid[3][k],1.0);
      Phiy[2][k][j]=phiy(kernel.grid[2][j],kernel.grid[3][k],-1.0);
    end
  end

  #
  function d1xphix(x::Float64,z::Float64,s::Float64)
    d1xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1xphipx[i]=d1xphiDx(x,z,kernel.px[i],s);
    end
    return d1xphipx;
  end
  #
  function d1yphiy(y::Float64,z::Float64,s::Float64)
    d1yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1yphipy[i]=d1yphiDy(y,z,kernel.py[i],s);
    end
    return d1yphipy;
  end
  #
  function d1zphix(x::Float64,z::Float64,s::Float64)
    d1zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1zphipx[i]=d1zphiDx(x,z,kernel.px[i],s);
    end
    return d1zphipx;
  end
  #
  function d1zphiy(y::Float64,z::Float64,s::Float64)
    d1zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1zphipy[i]=d1zphiDy(y,z,kernel.py[i],s);
    end
    return d1zphipy;
  end
  #

  #
  function d11xzphix(x::Float64,z::Float64,s::Float64)
    d11xzphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d11xzphipx[i]=d11xzphiDx(x,z,kernel.px[i],s);
    end
    return d11xzphipx
  end
  #
  function d11yzphiy(y::Float64,z::Float64,s::Float64)
    d11yzphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d11yzphipy[i]=d11yzphiDy(y,z,kernel.py[i],s);
    end
    return d11yzphipy
  end
  #

  #
  function d2xphix(x::Float64,z::Float64,s::Float64)
    d2xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2xphipx[i]=d2xphiDx(x,z,kernel.px[i],s);
    end
    return d2xphipx;
  end
  function d2yphiy(y::Float64,z::Float64,s::Float64)
    d2yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2yphipy[i]=d2yphiDy(y,z,kernel.py[i],s);
    end
    return d2yphipy;
  end
  #
  function d2zphix(x::Float64,z::Float64,s::Float64)
    d2zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2zphipx[i]=d2zphiDx(x,z,kernel.px[i],s);
    end
    return d2zphipx;
  end
  function d2zphiy(y::Float64,z::Float64,s::Float64)
    d2zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2zphipy[i]=d2zphiDy(y,z,kernel.py[i],s);
    end
    return d2zphipy;
  end
  #
  ##

  ##
  function phiVect(x::Array{Float64,1})
    phipx1=phix(x[1],x[3],1.0);
    phipy1=phiy(x[2],x[3],1.0);
    phipx2=phix(x[1],x[3],-1.0);
    phipy2=phiy(x[2],x[3],-1.0);
    v=zeros(kernel.Npx*kernel.Npy);
    l=1;
    for j in 1:kernel.Npy
      for i in 1:kernel.Npx
        v[l]=phipx1[i]*phipy1[j]+phipx2[i]*phipy2[j];
        l+=1;
      end
    end
    return v;
  end
  function d1phiVect(m::Int64,x::Array{Float64,1})
    #
    phipx1=phix(x[1],x[3],1.0);
    phipy1=phiy(x[2],x[3],1.0);
    phipx2=phix(x[1],x[3],-1.0);
    phipy2=phiy(x[2],x[3],-1.0);
    #
    d1xphipx1=d1xphix(x[1],x[3],1.0);
    d1yphipy1=d1yphiy(x[2],x[3],1.0);
    d1zphipx1=d1zphix(x[1],x[3],1.0);
    d1zphipy1=d1zphiy(x[2],x[3],1.0);
    d1xphipx2=d1xphix(x[1],x[3],-1.0);
    d1yphipy2=d1yphiy(x[2],x[3],-1.0);
    d1zphipx2=d1zphix(x[1],x[3],-1.0);
    d1zphipy2=d1zphiy(x[2],x[3],-1.0);
    #
    v=zeros(kernel.Npx*kernel.Npy);
    if m==1
      l=1;
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d1xphipx1[i]*phipy1[j]+d1xphipx2[i]*phipy2[j];
          l+=1;
        end
      end
      return v;
    elseif m==2
      l=1;
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=phipx1[i]*d1yphipy1[j]+phipx2[i]*d1yphipy2[j];
          l+=1;
        end
      end
      return v;
    else
      l=1;
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=(d1zphipx1[i]*phipy1[j]+phipx1[i]*d1zphipy1[j])+(d1zphipx2[i]*phipy2[j]+phipx2[i]*d1zphipy2[j]);
          l+=1;
        end
      end
      return v;
    end
  end
  function d11phiVect(m::Int64,n::Int64,x::Array{Float64,1})
    #
    phipx1=phix(x[1],x[3],1.0);
    phipy1=phiy(x[2],x[3],1.0);
    phipx2=phix(x[1],x[3],-1.0);
    phipy2=phiy(x[2],x[3],-1.0);
    #
    d1xphipx1=d1xphix(x[1],x[3],1.0);
    d1yphipy1=d1yphiy(x[2],x[3],1.0);
    d1zphipx1=d1zphix(x[1],x[3],1.0);
    d1zphipy1=d1zphiy(x[2],x[3],1.0);
    d1xphipx2=d1xphix(x[1],x[3],-1.0);
    d1yphipy2=d1yphiy(x[2],x[3],-1.0);
    d1zphipx2=d1zphix(x[1],x[3],-1.0);
    d1zphipy2=d1zphiy(x[2],x[3],-1.0);
    #
    d11xzphipx1=d11xzphix(x[1],x[3],1.0);
    d11yzphipy1=d11yzphiy(x[2],x[3],1.0);
    d11xzphipx2=d11xzphix(x[1],x[3],-1.0);
    d11yzphipy2=d11yzphiy(x[2],x[3],-1.0);
    #
    v=zeros(kernel.Npx*kernel.Npy);
    l=1;
    if m==1 && n==2 || m==2 && n==1
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d1xphipx1[i]*d1yphipy1[j]+d1xphipx2[i]*d1yphipy2[j];
          l+=1;
        end
      end
      return v;
    elseif m==1 && n==3 || m==3 && n==1
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=(d11xzphipx1[i]*phipy1[j]+d1xphipx1[i]*d1zphipy1[j])+(d11xzphipx2[i]*phipy2[j]+d1xphipx2[i]*d1zphipy2[j]);
          l+=1;
        end
      end
      return v;
    elseif m==2 && n==3 || m==3 && n==2
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=(phipx1[i]*d11yzphipy1[j]+d1zphipx1[i]*d1yphipy1[j])+(phipx2[i]*d11yzphipy2[j]+d1zphipx2[i]*d1yphipy2[j]);
          l+=1;
        end
      end
      return v;
    end
  end
  function d2phiVect(m::Int64,x::Array{Float64,1})
    #
    phipx1=phix(x[1],x[3],1.0);
    phipy1=phiy(x[2],x[3],1.0);
    phipx2=phix(x[1],x[3],-1.0);
    phipy2=phiy(x[2],x[3],-1.0);
    #
    d1zphipx1=d1zphix(x[1],x[3],1.0);
    d1zphipy1=d1zphiy(x[2],x[3],1.0);
    d1zphipx2=d1zphix(x[1],x[3],-1.0);
    d1zphipy2=d1zphiy(x[2],x[3],-1.0);
    #
    d2xphipx1=d2xphix(x[1],x[3],1.0);
    d2yphipy1=d2yphiy(x[2],x[3],1.0);
    d2zphipx1=d2zphix(x[1],x[3],1.0);
    d2zphipy1=d2zphiy(x[2],x[3],1.0);
    d2xphipx2=d2xphix(x[1],x[3],-1.0);
    d2yphipy2=d2yphiy(x[2],x[3],-1.0);
    d2zphipx2=d2zphix(x[1],x[3],-1.0);
    d2zphipy2=d2zphiy(x[2],x[3],-1.0);
    #
    v=zeros(kernel.Npx*kernel.Npy);
    l=1;
    if m==1
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=(d2xphipx1[i]*phipy1[j])+(d2xphipx2[i]*phipy2[j]);
          l+=1;
        end
      end
      return v;
    elseif m==2
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=(phipx1[i]*d2yphipy1[j])+(phipx2[i]*d2yphipy2[j]);
          l+=1;
        end
      end
      return v;
    else
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=(d2zphipx1[i]*phipy1[j]+phipx1[i]*d2zphipy1[j]+2*d1zphipx1[i]*d1zphipy1[j])+(d2zphipx2[i]*phipy2[j]+phipx2[i]*d2zphipy2[j]+2*d1zphipx2[i]*d1zphipy2[j]);
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
            v[lp]=Phix[1][k][i][ip]*Phiy[1][k][j][jp]+Phix[2][k][i][ip]*Phiy[2][k][j][jp];
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


  function correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Float64,1},1},1},1})
    a11=[dot(phix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    b11=[dot(phiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    a12=[dot(phix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    b12=[dot(phiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    a21=[dot(phix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    b21=[dot(phiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    a22=[dot(phix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    b22=[dot(phiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    return dot(a11,b11)+dot(a12,b12)+dot(a21,b21)+dot(a22,b22)-ob(x);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Float64,1},1},1},1})
    d1c=zeros(kernel.dim);
    a11=[dot(phix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    b11=[dot(phiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    a12=[dot(phix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    b12=[dot(phiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    a21=[dot(phix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    b21=[dot(phiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    a22=[dot(phix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    b22=[dot(phiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];

    da11=[dot(d1xphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    da12=[dot(d1xphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    db11=[dot(d1yphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    db12=[dot(d1yphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dc11=[dot(d1zphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dc12=[dot(d1zphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    dd11=[dot(d1zphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    dd12=[dot(d1zphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    da22=[dot(d1xphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    da21=[dot(d1xphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    db22=[dot(d1yphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    db21=[dot(d1yphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    dc22=[dot(d1zphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    dc21=[dot(d1zphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dd22=[dot(d1zphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dd21=[dot(d1zphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];

    d1c[1]=dot(da11,b11)+dot(da12,b12)+dot(da22,b22)+dot(da21,b21)-d1ob(1,x);
    d1c[2]=dot(a11,db11)+dot(a12,db12)+dot(a22,db22)+dot(a21,db21)-d1ob(2,x);
    d1c[3]=(dot(dc11,b11)+dot(a11,dd11))+(dot(dc22,b22)+dot(a22,dd22))+(dot(dc12,b12)+dot(a12,dd12))+(dot(dc21,b21)+dot(a21,dd21))-d1ob(3,x);
    return d1c
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Float64,1},1},1},1})
    d2c=zeros(kernel.dim,kernel.dim);
    a11=[dot(phix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    b11=[dot(phiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    a12=[dot(phix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    b12=[dot(phiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    a21=[dot(phix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    b21=[dot(phiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    a22=[dot(phix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    b22=[dot(phiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];

    da11=[dot(d1xphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    da12=[dot(d1xphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    db11=[dot(d1yphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    db12=[dot(d1yphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dc11=[dot(d1zphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dc12=[dot(d1zphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    dd11=[dot(d1zphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    dd12=[dot(d1zphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    da22=[dot(d1xphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    da21=[dot(d1xphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    db22=[dot(d1yphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    db21=[dot(d1yphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    dc22=[dot(d1zphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    dc21=[dot(d1zphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dd22=[dot(d1zphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dd21=[dot(d1zphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];


    dac11=[dot(d11xzphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dac12=[dot(d11xzphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    dbd11=[dot(d11yzphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    dbd12=[dot(d11yzphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dda11=[dot(d2xphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dda12=[dot(d2xphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    ddb11=[dot(d2yphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    ddb12=[dot(d2yphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    ddc11=[dot(d2zphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    ddc12=[dot(d2zphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    ddd11=[dot(d2zphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    ddd12=[dot(d2zphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];

    dac22=[dot(d11xzphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    dbd22=[dot(d11yzphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dda22=[dot(d2xphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    ddb22=[dot(d2yphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    ddc22=[dot(d2zphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    ddd22=[dot(d2zphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dac21=[dot(d11xzphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dbd21=[dot(d11yzphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    dda21=[dot(d2xphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    ddb21=[dot(d2yphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    ddc21=[dot(d2zphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    ddd21=[dot(d2zphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];

    d2c[1,2]=dot(da11,db11)+dot(da22,db22)+dot(da12,db12)+dot(da21,db21)-dot(d11phiVect(1,2,x),y);
    d2c[1,3]=(dot(da11,dd11)+dot(dac11,b11))+(dot(da22,dd22)+dot(dac22,b22))+(dot(da12,dd12)+dot(dac12,b12))+(dot(da21,dd21)+dot(dac21,b21))-dot(d11phiVect(1,3,x),y);
    d2c[2,3]=(dot(dc11,db11)+dot(a11,dbd11))+(dot(dc22,db22)+dot(a22,dbd22))+(dot(dc12,db12)+dot(a12,dbd12))+(dot(dc21,db21)+dot(a21,dbd21))-dot(d11phiVect(2,3,x),y);
    d2c=d2c+d2c';
    d2c[1,1]=dot(dda11,b11)+dot(dda22,b22)+dot(dda12,b12)+dot(dda21,b21)-d2ob(1,x);
    d2c[2,2]=dot(a11,ddb11)+dot(a22,ddb22)+dot(a12,ddb12)+dot(a21,ddb21)-d2ob(2,x);
    d2c[3,3]=(dot(ddc11,b11)+dot(a11,ddd11)+2*dot(dc11,dd11))+(dot(ddc22,b22)+dot(a22,ddd22)+2*dot(dc22,dd22))+(dot(ddc12,b12)+dot(a12,ddd12)+2*dot(dc12,dd12))+(dot(ddc21,b21)+dot(a21,ddd21)+2*dot(dc21,dd21))-d2ob(3,x);
    return(d2c)
  end

  gaussconv2DDoubleHelix(typeof(kernel),kernel.dim,kernel.bounds,normObs,Phix,Phiy,PhisY,phix,phiy,phiVect,d1phiVect,d11phiVect,d2phiVect,y,c,d10c,d01c,d11c,d20c,d02c,ob,d1ob,d11ob,d2ob,correl,d1correl,d2correl);
end

function setoperator(kernel::doubleHelix,a0::Array{Float64,1},x0::Array{Array{Float64,1},1},w::Array{Float64,1})
  function phiDx(x::Float64,z::Float64,xi::Float64,s::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2);
    return .5*(erf(alphap/kernel.sigmax)-erf(alpham/kernel.sigmax));
  end
  function phiDy(y::Float64,z::Float64,yi::Float64,s::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2);
    return .5*(erf(alphap/kernel.sigmay)-erf(alpham/kernel.sigmay));
  end

  function d1xphiDx(x::Float64,z::Float64,xi::Float64,s::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmax)*(exp(-(alphap/kernel.sigmax)^2) - exp(-(alpham/kernel.sigmax)^2));
  end
  function d1yphiDy(y::Float64,z::Float64,yi::Float64,s::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmay)*(exp(-(alphap/kernel.sigmay)^2) - exp(-(alpham/kernel.sigmay)^2));
  end

  function d11xzphiDx(x::Float64,z::Float64,xi::Float64,s::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2);
    return -s*kernel.d1rotx(z)/(sqrt(pi)*kernel.sigmax^3)*(alphap*exp(-(alphap/kernel.sigmax)^2) - alpham*exp(-(alpham/kernel.sigmax)^2));
  end
  function d11yzphiDy(y::Float64,z::Float64,yi::Float64,s::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2);
    return -s*kernel.d1roty(z)/(sqrt(pi)*kernel.sigmay^3)*(alphap*exp(-(alphap/kernel.sigmay)^2) - alpham*exp(-(alpham/kernel.sigmay)^2));
  end

  function d2xphiDx(x::Float64,z::Float64,xi::Float64,s::Float64)
    return -1/(sqrt(pi)*kernel.sigmax^2)*( ((xi+kernel.Dpx/2-x-s*kernel.rotx(z))/(sqrt(2)*kernel.sigmax))*exp(-((xi+kernel.Dpx/2-x-s*kernel.rotx(z))/(sqrt(2)*kernel.sigmax))^2) - ((xi-kernel.Dpx/2-x-s*kernel.rotx(z))/(sqrt(2)*kernel.sigmax))*exp(-((xi-kernel.Dpx/2-x-s*kernel.rotx(z))/(sqrt(2)*kernel.sigmax))^2) );
  end
  function d2yphiDy(y::Float64,z::Float64,yi::Float64,s::Float64)
    return -1/(sqrt(pi)*kernel.sigmay^2)*( ((yi+kernel.Dpy/2-y-s*kernel.roty(z))/(sqrt(2)*kernel.sigmay))*exp(-((yi+kernel.Dpy/2-y-s*kernel.roty(z))/(sqrt(2)*kernel.sigmay))^2) - ((yi-kernel.Dpy/2-y-s*kernel.roty(z))/(sqrt(2)*kernel.sigmay))*exp(-((yi-kernel.Dpy/2-y-s*kernel.roty(z))/(sqrt(2)*kernel.sigmay))^2) );
  end

  function d1zphiDx(x::Float64,z::Float64,xi::Float64,s::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2);
    return -s*kernel.d1rotx(z)/(sqrt(2*pi)*kernel.sigmax)*(exp(-(alphap/kernel.sigmax)^2) - exp(-(alpham/kernel.sigmax)^2));
  end
  function d1zphiDy(y::Float64,z::Float64,yi::Float64,s::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2);
    return -s*kernel.d1roty(z)/(sqrt(2*pi)*kernel.sigmay)*(exp(-(alphap/kernel.sigmay)^2) - exp(-(alpham/kernel.sigmay)^2));
  end

  function d2zphiDx(x::Float64,z::Float64,xi::Float64,s::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2);
    return -s*kernel.d2rotx(z)/(sqrt(2*pi)*kernel.sigmax)*(exp(-(alphap/kernel.sigmax)^2) - exp(-(alpham/kernel.sigmax)^2)) - s^2*kernel.d1rotx(z)^2/(sqrt(pi)*kernel.sigmax^3)*(alphap*exp(-(alphap/kernel.sigmax)^2) - alpham*exp(-(alpham/kernel.sigmax)^2));
  end
  function d2zphiDy(y::Float64,z::Float64,yi::Float64,s::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2);
    return -s*kernel.d2roty(z)/(sqrt(2*pi)*kernel.sigmay)*(exp(-(alphap/kernel.sigmay)^2) - exp(-(alpham/kernel.sigmay)^2)) - s^2*kernel.d1roty(z)^2/(sqrt(pi)*kernel.sigmay^3)*(alphap*exp(-(alphap/kernel.sigmay)^2) - alpham*exp(-(alpham/kernel.sigmay)^2));
  end
  ##

  ##
  function phix(x::Float64,z::Float64,s::Float64)
    phipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      phipx[i]=phiDx(x,z,kernel.px[i],s);
    end
    return phipx;
  end
  #
  function phiy(y::Float64,z::Float64,s::Float64)
    phipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      phipy[i]=phiDy(y,z,kernel.py[i],s);
    end
    return phipy;
  end
  #

  Phix=[Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3]),Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3])];
  Phiy=[Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3]),Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3])];
  for k in 1:kernel.nbpointsgrid[3]
    Phix[1][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[1]);
    Phiy[1][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[2]);
    Phix[2][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[1]);
    Phiy[2][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[2]);
    for i in 1:kernel.nbpointsgrid[1]
      Phix[1][k][i]=phix(kernel.grid[1][i],kernel.grid[3][k],1.0);
      Phix[2][k][i]=phix(kernel.grid[1][i],kernel.grid[3][k],-1.0);
    end
    for j in 1:kernel.nbpointsgrid[2]
      Phiy[1][k][j]=phiy(kernel.grid[2][j],kernel.grid[3][k],1.0);
      Phiy[2][k][j]=phiy(kernel.grid[2][j],kernel.grid[3][k],-1.0);
    end
  end

  #
  function d1xphix(x::Float64,z::Float64,s::Float64)
    d1xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1xphipx[i]=d1xphiDx(x,z,kernel.px[i],s);
    end
    return d1xphipx;
  end
  #
  function d1yphiy(y::Float64,z::Float64,s::Float64)
    d1yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1yphipy[i]=d1yphiDy(y,z,kernel.py[i],s);
    end
    return d1yphipy;
  end
  #
  function d1zphix(x::Float64,z::Float64,s::Float64)
    d1zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1zphipx[i]=d1zphiDx(x,z,kernel.px[i],s);
    end
    return d1zphipx;
  end
  #
  function d1zphiy(y::Float64,z::Float64,s::Float64)
    d1zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1zphipy[i]=d1zphiDy(y,z,kernel.py[i],s);
    end
    return d1zphipy;
  end
  #

  #
  function d11xzphix(x::Float64,z::Float64,s::Float64)
    d11xzphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d11xzphipx[i]=d11xzphiDx(x,z,kernel.px[i],s);
    end
    return d11xzphipx
  end
  #
  function d11yzphiy(y::Float64,z::Float64,s::Float64)
    d11yzphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d11yzphipy[i]=d11yzphiDy(y,z,kernel.py[i],s);
    end
    return d11yzphipy
  end
  #

  #
  function d2xphix(x::Float64,z::Float64,s::Float64)
    d2xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2xphipx[i]=d2xphiDx(x,z,kernel.px[i],s);
    end
    return d2xphipx;
  end
  function d2yphiy(y::Float64,z::Float64,s::Float64)
    d2yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2yphipy[i]=d2yphiDy(y,z,kernel.py[i],s);
    end
    return d2yphipy;
  end
  #
  function d2zphix(x::Float64,z::Float64,s::Float64)
    d2zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2zphipx[i]=d2zphiDx(x,z,kernel.px[i],s);
    end
    return d2zphipx;
  end
  function d2zphiy(y::Float64,z::Float64,s::Float64)
    d2zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2zphipy[i]=d2zphiDy(y,z,kernel.py[i],s);
    end
    return d2zphipy;
  end
  #
  ##

  ##
  function phiVect(x::Array{Float64,1})
    phipx1=phix(x[1],x[3],1.0);
    phipy1=phiy(x[2],x[3],1.0);
    phipx2=phix(x[1],x[3],-1.0);
    phipy2=phiy(x[2],x[3],-1.0);
    v=zeros(kernel.Npx*kernel.Npy);
    l=1;
    for j in 1:kernel.Npy
      for i in 1:kernel.Npx
        v[l]=phipx1[i]*phipy1[j]+phipx2[i]*phipy2[j];
        l+=1;
      end
    end
    return v;
  end
  function d1phiVect(m::Int64,x::Array{Float64,1})
    #
    phipx1=phix(x[1],x[3],1.0);
    phipy1=phiy(x[2],x[3],1.0);
    phipx2=phix(x[1],x[3],-1.0);
    phipy2=phiy(x[2],x[3],-1.0);
    #
    d1xphipx1=d1xphix(x[1],x[3],1.0);
    d1yphipy1=d1yphiy(x[2],x[3],1.0);
    d1zphipx1=d1zphix(x[1],x[3],1.0);
    d1zphipy1=d1zphiy(x[2],x[3],1.0);
    d1xphipx2=d1xphix(x[1],x[3],-1.0);
    d1yphipy2=d1yphiy(x[2],x[3],-1.0);
    d1zphipx2=d1zphix(x[1],x[3],-1.0);
    d1zphipy2=d1zphiy(x[2],x[3],-1.0);
    #
    v=zeros(kernel.Npx*kernel.Npy);
    if m==1
      l=1;
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d1xphipx1[i]*phipy1[j]+d1xphipx2[i]*phipy2[j];
          l+=1;
        end
      end
      return v;
    elseif m==2
      l=1;
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=phipx1[i]*d1yphipy1[j]+phipx2[i]*d1yphipy2[j];
          l+=1;
        end
      end
      return v;
    else
      l=1;
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=(d1zphipx1[i]*phipy1[j]+phipx1[i]*d1zphipy1[j])+(d1zphipx2[i]*phipy2[j]+phipx2[i]*d1zphipy2[j]);
          l+=1;
        end
      end
      return v;
    end
  end
  function d11phiVect(m::Int64,n::Int64,x::Array{Float64,1})
    #
    phipx1=phix(x[1],x[3],1.0);
    phipy1=phiy(x[2],x[3],1.0);
    phipx2=phix(x[1],x[3],-1.0);
    phipy2=phiy(x[2],x[3],-1.0);
    #
    d1xphipx1=d1xphix(x[1],x[3],1.0);
    d1yphipy1=d1yphiy(x[2],x[3],1.0);
    d1zphipx1=d1zphix(x[1],x[3],1.0);
    d1zphipy1=d1zphiy(x[2],x[3],1.0);
    d1xphipx2=d1xphix(x[1],x[3],-1.0);
    d1yphipy2=d1yphiy(x[2],x[3],-1.0);
    d1zphipx2=d1zphix(x[1],x[3],-1.0);
    d1zphipy2=d1zphiy(x[2],x[3],-1.0);
    #
    d11xzphipx1=d11xzphix(x[1],x[3],1.0);
    d11yzphipy1=d11yzphiy(x[2],x[3],1.0);
    d11xzphipx2=d11xzphix(x[1],x[3],-1.0);
    d11yzphipy2=d11yzphiy(x[2],x[3],-1.0);
    #
    v=zeros(kernel.Npx*kernel.Npy);
    l=1;
    if m==1 && n==2 || m==2 && n==1
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d1xphipx1[i]*d1yphipy1[j]+d1xphipx2[i]*d1yphipy2[j];
          l+=1;
        end
      end
      return v;
    elseif m==1 && n==3 || m==3 && n==1
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=(d11xzphipx1[i]*phipy1[j]+d1xphipx1[i]*d1zphipy1[j])+(d11xzphipx2[i]*phipy2[j]+d1xphipx2[i]*d1zphipy2[j]);
          l+=1;
        end
      end
      return v;
    elseif m==2 && n==3 || m==3 && n==2
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=(phipx1[i]*d11yzphipy1[j]+d1zphipx1[i]*d1yphipy1[j])+(phipx2[i]*d11yzphipy2[j]+d1zphipx2[i]*d1yphipy2[j]);
          l+=1;
        end
      end
      return v;
    end
  end
  function d2phiVect(m::Int64,x::Array{Float64,1})
    #
    phipx1=phix(x[1],x[3],1.0);
    phipy1=phiy(x[2],x[3],1.0);
    phipx2=phix(x[1],x[3],-1.0);
    phipy2=phiy(x[2],x[3],-1.0);
    #
    d1zphipx1=d1zphix(x[1],x[3],1.0);
    d1zphipy1=d1zphiy(x[2],x[3],1.0);
    d1zphipx2=d1zphix(x[1],x[3],-1.0);
    d1zphipy2=d1zphiy(x[2],x[3],-1.0);
    #
    d2xphipx1=d2xphix(x[1],x[3],1.0);
    d2yphipy1=d2yphiy(x[2],x[3],1.0);
    d2zphipx1=d2zphix(x[1],x[3],1.0);
    d2zphipy1=d2zphiy(x[2],x[3],1.0);
    d2xphipx2=d2xphix(x[1],x[3],-1.0);
    d2yphipy2=d2yphiy(x[2],x[3],-1.0);
    d2zphipx2=d2zphix(x[1],x[3],-1.0);
    d2zphipy2=d2zphiy(x[2],x[3],-1.0);
    #
    v=zeros(kernel.Npx*kernel.Npy);
    l=1;
    if m==1
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=(d2xphipx1[i]*phipy1[j])+(d2xphipx2[i]*phipy2[j]);
          l+=1;
        end
      end
      return v;
    elseif m==2
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=(phipx1[i]*d2yphipy1[j])+(phipx2[i]*d2yphipy2[j]);
          l+=1;
        end
      end
      return v;
    else
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=(d2zphipx1[i]*phipy1[j]+phipx1[i]*d2zphipy1[j]+2*d1zphipx1[i]*d1zphipy1[j])+(d2zphipx2[i]*phipy2[j]+phipx2[i]*d2zphipy2[j]+2*d1zphipx2[i]*d1zphipy2[j]);
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
            v[lp]=Phix[1][k][i][ip]*Phiy[1][k][j][jp]+Phix[2][k][i][ip]*Phiy[2][k][j][jp];
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


  function correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Float64,1},1},1},1})
    a11=[dot(phix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    b11=[dot(phiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    a12=[dot(phix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    b12=[dot(phiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    a21=[dot(phix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    b21=[dot(phiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    a22=[dot(phix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    b22=[dot(phiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    return dot(a11,b11)+dot(a12,b12)+dot(a21,b21)+dot(a22,b22)-ob(x);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Float64,1},1},1},1})
    d1c=zeros(kernel.dim);
    a11=[dot(phix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    b11=[dot(phiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    a12=[dot(phix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    b12=[dot(phiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    a21=[dot(phix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    b21=[dot(phiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    a22=[dot(phix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    b22=[dot(phiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];

    da11=[dot(d1xphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    da12=[dot(d1xphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    db11=[dot(d1yphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    db12=[dot(d1yphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dc11=[dot(d1zphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dc12=[dot(d1zphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    dd11=[dot(d1zphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    dd12=[dot(d1zphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    da22=[dot(d1xphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    da21=[dot(d1xphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    db22=[dot(d1yphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    db21=[dot(d1yphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    dc22=[dot(d1zphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    dc21=[dot(d1zphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dd22=[dot(d1zphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dd21=[dot(d1zphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];

    d1c[1]=dot(da11,b11)+dot(da12,b12)+dot(da22,b22)+dot(da21,b21)-d1ob(1,x);
    d1c[2]=dot(a11,db11)+dot(a12,db12)+dot(a22,db22)+dot(a21,db21)-d1ob(2,x);
    d1c[3]=(dot(dc11,b11)+dot(a11,dd11))+(dot(dc22,b22)+dot(a22,dd22))+(dot(dc12,b12)+dot(a12,dd12))+(dot(dc21,b21)+dot(a21,dd21))-d1ob(3,x);
    return d1c
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Float64,1},1},1},1})
    d2c=zeros(kernel.dim,kernel.dim);
    a11=[dot(phix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    b11=[dot(phiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    a12=[dot(phix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    b12=[dot(phiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    a21=[dot(phix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    b21=[dot(phiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    a22=[dot(phix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    b22=[dot(phiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];

    da11=[dot(d1xphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    da12=[dot(d1xphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    db11=[dot(d1yphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    db12=[dot(d1yphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dc11=[dot(d1zphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dc12=[dot(d1zphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    dd11=[dot(d1zphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    dd12=[dot(d1zphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    da22=[dot(d1xphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    da21=[dot(d1xphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    db22=[dot(d1yphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    db21=[dot(d1yphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    dc22=[dot(d1zphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    dc21=[dot(d1zphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dd22=[dot(d1zphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dd21=[dot(d1zphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];


    dac11=[dot(d11xzphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dac12=[dot(d11xzphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    dbd11=[dot(d11yzphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    dbd12=[dot(d11yzphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dda11=[dot(d2xphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dda12=[dot(d2xphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    ddb11=[dot(d2yphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    ddb12=[dot(d2yphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    ddc11=[dot(d2zphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    ddc12=[dot(d2zphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    ddd11=[dot(d2zphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    ddd12=[dot(d2zphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];

    dac22=[dot(d11xzphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    dbd22=[dot(d11yzphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dda22=[dot(d2xphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    ddb22=[dot(d2yphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    ddc22=[dot(d2zphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    ddd22=[dot(d2zphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dac21=[dot(d11xzphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dbd21=[dot(d11yzphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    dda21=[dot(d2xphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    ddb21=[dot(d2yphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    ddc21=[dot(d2zphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    ddd21=[dot(d2zphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];

    d2c[1,2]=dot(da11,db11)+dot(da22,db22)+dot(da12,db12)+dot(da21,db21)-dot(d11phiVect(1,2,x),y);
    d2c[1,3]=(dot(da11,dd11)+dot(dac11,b11))+(dot(da22,dd22)+dot(dac22,b22))+(dot(da12,dd12)+dot(dac12,b12))+(dot(da21,dd21)+dot(dac21,b21))-dot(d11phiVect(1,3,x),y);
    d2c[2,3]=(dot(dc11,db11)+dot(a11,dbd11))+(dot(dc22,db22)+dot(a22,dbd22))+(dot(dc12,db12)+dot(a12,dbd12))+(dot(dc21,db21)+dot(a21,dbd21))-dot(d11phiVect(2,3,x),y);
    d2c=d2c+d2c';
    d2c[1,1]=dot(dda11,b11)+dot(dda22,b22)+dot(dda12,b12)+dot(dda21,b21)-d2ob(1,x);
    d2c[2,2]=dot(a11,ddb11)+dot(a22,ddb22)+dot(a12,ddb12)+dot(a21,ddb21)-d2ob(2,x);
    d2c[3,3]=(dot(ddc11,b11)+dot(a11,ddd11)+2*dot(dc11,dd11))+(dot(ddc22,b22)+dot(a22,ddd22)+2*dot(dc22,dd22))+(dot(ddc12,b12)+dot(a12,ddd12)+2*dot(dc12,dd12))+(dot(ddc21,b21)+dot(a21,ddd21)+2*dot(dc21,dd21))-d2ob(3,x);
    return(d2c)
  end

  gaussconv2DDoubleHelix(typeof(kernel),kernel.dim,kernel.bounds,normObs,Phix,Phiy,PhisY,phix,phiy,phiVect,d1phiVect,d11phiVect,d2phiVect,y,c,d10c,d01c,d11c,d20c,d02c,ob,d1ob,d11ob,d2ob,correl,d1correl,d2correl);
end

function setoperator(kernel::doubleHelix,a0::Array{Float64,1},x0::Array{Array{Float64,1},1},w::Array{Float64,1},nPhoton::Float64)
  function phiDx(x::Float64,z::Float64,xi::Float64,s::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2);
    return .5*(erf(alphap/kernel.sigmax)-erf(alpham/kernel.sigmax));
  end
  function phiDy(y::Float64,z::Float64,yi::Float64,s::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2);
    return .5*(erf(alphap/kernel.sigmay)-erf(alpham/kernel.sigmay));
  end

  function d1xphiDx(x::Float64,z::Float64,xi::Float64,s::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmax)*(exp(-(alphap/kernel.sigmax)^2) - exp(-(alpham/kernel.sigmax)^2));
  end
  function d1yphiDy(y::Float64,z::Float64,yi::Float64,s::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmay)*(exp(-(alphap/kernel.sigmay)^2) - exp(-(alpham/kernel.sigmay)^2));
  end

  function d11xzphiDx(x::Float64,z::Float64,xi::Float64,s::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2);
    return -s*kernel.d1rotx(z)/(sqrt(pi)*kernel.sigmax^3)*(alphap*exp(-(alphap/kernel.sigmax)^2) - alpham*exp(-(alpham/kernel.sigmax)^2));
  end
  function d11yzphiDy(y::Float64,z::Float64,yi::Float64,s::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2);
    return -s*kernel.d1roty(z)/(sqrt(pi)*kernel.sigmay^3)*(alphap*exp(-(alphap/kernel.sigmay)^2) - alpham*exp(-(alpham/kernel.sigmay)^2));
  end

  function d2xphiDx(x::Float64,z::Float64,xi::Float64,s::Float64)
    return -1/(sqrt(pi)*kernel.sigmax^2)*( ((xi+kernel.Dpx/2-x-s*kernel.rotx(z))/(sqrt(2)*kernel.sigmax))*exp(-((xi+kernel.Dpx/2-x-s*kernel.rotx(z))/(sqrt(2)*kernel.sigmax))^2) - ((xi-kernel.Dpx/2-x-s*kernel.rotx(z))/(sqrt(2)*kernel.sigmax))*exp(-((xi-kernel.Dpx/2-x-s*kernel.rotx(z))/(sqrt(2)*kernel.sigmax))^2) );
  end
  function d2yphiDy(y::Float64,z::Float64,yi::Float64,s::Float64)
    return -1/(sqrt(pi)*kernel.sigmay^2)*( ((yi+kernel.Dpy/2-y-s*kernel.roty(z))/(sqrt(2)*kernel.sigmay))*exp(-((yi+kernel.Dpy/2-y-s*kernel.roty(z))/(sqrt(2)*kernel.sigmay))^2) - ((yi-kernel.Dpy/2-y-s*kernel.roty(z))/(sqrt(2)*kernel.sigmay))*exp(-((yi-kernel.Dpy/2-y-s*kernel.roty(z))/(sqrt(2)*kernel.sigmay))^2) );
  end

  function d1zphiDx(x::Float64,z::Float64,xi::Float64,s::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2);
    return -s*kernel.d1rotx(z)/(sqrt(2*pi)*kernel.sigmax)*(exp(-(alphap/kernel.sigmax)^2) - exp(-(alpham/kernel.sigmax)^2));
  end
  function d1zphiDy(y::Float64,z::Float64,yi::Float64,s::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2);
    return -s*kernel.d1roty(z)/(sqrt(2*pi)*kernel.sigmay)*(exp(-(alphap/kernel.sigmay)^2) - exp(-(alpham/kernel.sigmay)^2));
  end

  function d2zphiDx(x::Float64,z::Float64,xi::Float64,s::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2);
    return -s*kernel.d2rotx(z)/(sqrt(2*pi)*kernel.sigmax)*(exp(-(alphap/kernel.sigmax)^2) - exp(-(alpham/kernel.sigmax)^2)) - s^2*kernel.d1rotx(z)^2/(sqrt(pi)*kernel.sigmax^3)*(alphap*exp(-(alphap/kernel.sigmax)^2) - alpham*exp(-(alpham/kernel.sigmax)^2));
  end
  function d2zphiDy(y::Float64,z::Float64,yi::Float64,s::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2);
    return -s*kernel.d2roty(z)/(sqrt(2*pi)*kernel.sigmay)*(exp(-(alphap/kernel.sigmay)^2) - exp(-(alpham/kernel.sigmay)^2)) - s^2*kernel.d1roty(z)^2/(sqrt(pi)*kernel.sigmay^3)*(alphap*exp(-(alphap/kernel.sigmay)^2) - alpham*exp(-(alpham/kernel.sigmay)^2));
  end
  ##

  ##
  function phix(x::Float64,z::Float64,s::Float64)
    phipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      phipx[i]=phiDx(x,z,kernel.px[i],s);
    end
    return phipx;
  end
  #
  function phiy(y::Float64,z::Float64,s::Float64)
    phipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      phipy[i]=phiDy(y,z,kernel.py[i],s);
    end
    return phipy;
  end
  #

  Phix=[Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3]),Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3])];
  Phiy=[Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3]),Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3])];
  for k in 1:kernel.nbpointsgrid[3]
    Phix[1][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[1]);
    Phiy[1][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[2]);
    Phix[2][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[1]);
    Phiy[2][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[2]);
    for i in 1:kernel.nbpointsgrid[1]
      Phix[1][k][i]=phix(kernel.grid[1][i],kernel.grid[3][k],1.0);
      Phix[2][k][i]=phix(kernel.grid[1][i],kernel.grid[3][k],-1.0);
    end
    for j in 1:kernel.nbpointsgrid[2]
      Phiy[1][k][j]=phiy(kernel.grid[2][j],kernel.grid[3][k],1.0);
      Phiy[2][k][j]=phiy(kernel.grid[2][j],kernel.grid[3][k],-1.0);
    end
  end

  #
  function d1xphix(x::Float64,z::Float64,s::Float64)
    d1xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1xphipx[i]=d1xphiDx(x,z,kernel.px[i],s);
    end
    return d1xphipx;
  end
  #
  function d1yphiy(y::Float64,z::Float64,s::Float64)
    d1yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1yphipy[i]=d1yphiDy(y,z,kernel.py[i],s);
    end
    return d1yphipy;
  end
  #
  function d1zphix(x::Float64,z::Float64,s::Float64)
    d1zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1zphipx[i]=d1zphiDx(x,z,kernel.px[i],s);
    end
    return d1zphipx;
  end
  #
  function d1zphiy(y::Float64,z::Float64,s::Float64)
    d1zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1zphipy[i]=d1zphiDy(y,z,kernel.py[i],s);
    end
    return d1zphipy;
  end
  #

  #
  function d11xzphix(x::Float64,z::Float64,s::Float64)
    d11xzphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d11xzphipx[i]=d11xzphiDx(x,z,kernel.px[i],s);
    end
    return d11xzphipx
  end
  #
  function d11yzphiy(y::Float64,z::Float64,s::Float64)
    d11yzphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d11yzphipy[i]=d11yzphiDy(y,z,kernel.py[i],s);
    end
    return d11yzphipy
  end
  #

  #
  function d2xphix(x::Float64,z::Float64,s::Float64)
    d2xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2xphipx[i]=d2xphiDx(x,z,kernel.px[i],s);
    end
    return d2xphipx;
  end
  function d2yphiy(y::Float64,z::Float64,s::Float64)
    d2yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2yphipy[i]=d2yphiDy(y,z,kernel.py[i],s);
    end
    return d2yphipy;
  end
  #
  function d2zphix(x::Float64,z::Float64,s::Float64)
    d2zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2zphipx[i]=d2zphiDx(x,z,kernel.px[i],s);
    end
    return d2zphipx;
  end
  function d2zphiy(y::Float64,z::Float64,s::Float64)
    d2zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2zphipy[i]=d2zphiDy(y,z,kernel.py[i],s);
    end
    return d2zphipy;
  end
  #
  ##

  ##
  function phiVect(x::Array{Float64,1})
    phipx1=phix(x[1],x[3],1.0);
    phipy1=phiy(x[2],x[3],1.0);
    phipx2=phix(x[1],x[3],-1.0);
    phipy2=phiy(x[2],x[3],-1.0);
    v=zeros(kernel.Npx*kernel.Npy);
    l=1;
    for j in 1:kernel.Npy
      for i in 1:kernel.Npx
        v[l]=phipx1[i]*phipy1[j]+phipx2[i]*phipy2[j];
        l+=1;
      end
    end
    return v;
  end
  function d1phiVect(m::Int64,x::Array{Float64,1})
    #
    phipx1=phix(x[1],x[3],1.0);
    phipy1=phiy(x[2],x[3],1.0);
    phipx2=phix(x[1],x[3],-1.0);
    phipy2=phiy(x[2],x[3],-1.0);
    #
    d1xphipx1=d1xphix(x[1],x[3],1.0);
    d1yphipy1=d1yphiy(x[2],x[3],1.0);
    d1zphipx1=d1zphix(x[1],x[3],1.0);
    d1zphipy1=d1zphiy(x[2],x[3],1.0);
    d1xphipx2=d1xphix(x[1],x[3],-1.0);
    d1yphipy2=d1yphiy(x[2],x[3],-1.0);
    d1zphipx2=d1zphix(x[1],x[3],-1.0);
    d1zphipy2=d1zphiy(x[2],x[3],-1.0);
    #
    v=zeros(kernel.Npx*kernel.Npy);
    if m==1
      l=1;
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d1xphipx1[i]*phipy1[j]+d1xphipx2[i]*phipy2[j];
          l+=1;
        end
      end
      return v;
    elseif m==2
      l=1;
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=phipx1[i]*d1yphipy1[j]+phipx2[i]*d1yphipy2[j];
          l+=1;
        end
      end
      return v;
    else
      l=1;
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=(d1zphipx1[i]*phipy1[j]+phipx1[i]*d1zphipy1[j])+(d1zphipx2[i]*phipy2[j]+phipx2[i]*d1zphipy2[j]);
          l+=1;
        end
      end
      return v;
    end
  end
  function d11phiVect(m::Int64,n::Int64,x::Array{Float64,1})
    #
    phipx1=phix(x[1],x[3],1.0);
    phipy1=phiy(x[2],x[3],1.0);
    phipx2=phix(x[1],x[3],-1.0);
    phipy2=phiy(x[2],x[3],-1.0);
    #
    d1xphipx1=d1xphix(x[1],x[3],1.0);
    d1yphipy1=d1yphiy(x[2],x[3],1.0);
    d1zphipx1=d1zphix(x[1],x[3],1.0);
    d1zphipy1=d1zphiy(x[2],x[3],1.0);
    d1xphipx2=d1xphix(x[1],x[3],-1.0);
    d1yphipy2=d1yphiy(x[2],x[3],-1.0);
    d1zphipx2=d1zphix(x[1],x[3],-1.0);
    d1zphipy2=d1zphiy(x[2],x[3],-1.0);
    #
    d11xzphipx1=d11xzphix(x[1],x[3],1.0);
    d11yzphipy1=d11yzphiy(x[2],x[3],1.0);
    d11xzphipx2=d11xzphix(x[1],x[3],-1.0);
    d11yzphipy2=d11yzphiy(x[2],x[3],-1.0);
    #
    v=zeros(kernel.Npx*kernel.Npy);
    l=1;
    if m==1 && n==2 || m==2 && n==1
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d1xphipx1[i]*d1yphipy1[j]+d1xphipx2[i]*d1yphipy2[j];
          l+=1;
        end
      end
      return v;
    elseif m==1 && n==3 || m==3 && n==1
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=(d11xzphipx1[i]*phipy1[j]+d1xphipx1[i]*d1zphipy1[j])+(d11xzphipx2[i]*phipy2[j]+d1xphipx2[i]*d1zphipy2[j]);
          l+=1;
        end
      end
      return v;
    elseif m==2 && n==3 || m==3 && n==2
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=(phipx1[i]*d11yzphipy1[j]+d1zphipx1[i]*d1yphipy1[j])+(phipx2[i]*d11yzphipy2[j]+d1zphipx2[i]*d1yphipy2[j]);
          l+=1;
        end
      end
      return v;
    end
  end
  function d2phiVect(m::Int64,x::Array{Float64,1})
    #
    phipx1=phix(x[1],x[3],1.0);
    phipy1=phiy(x[2],x[3],1.0);
    phipx2=phix(x[1],x[3],-1.0);
    phipy2=phiy(x[2],x[3],-1.0);
    #
    d1zphipx1=d1zphix(x[1],x[3],1.0);
    d1zphipy1=d1zphiy(x[2],x[3],1.0);
    d1zphipx2=d1zphix(x[1],x[3],-1.0);
    d1zphipy2=d1zphiy(x[2],x[3],-1.0);
    #
    d2xphipx1=d2xphix(x[1],x[3],1.0);
    d2yphipy1=d2yphiy(x[2],x[3],1.0);
    d2zphipx1=d2zphix(x[1],x[3],1.0);
    d2zphipy1=d2zphiy(x[2],x[3],1.0);
    d2xphipx2=d2xphix(x[1],x[3],-1.0);
    d2yphipy2=d2yphiy(x[2],x[3],-1.0);
    d2zphipx2=d2zphix(x[1],x[3],-1.0);
    d2zphipy2=d2zphiy(x[2],x[3],-1.0);
    #
    v=zeros(kernel.Npx*kernel.Npy);
    l=1;
    if m==1
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=(d2xphipx1[i]*phipy1[j])+(d2xphipx2[i]*phipy2[j]);
          l+=1;
        end
      end
      return v;
    elseif m==2
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=(phipx1[i]*d2yphipy1[j])+(phipx2[i]*d2yphipy2[j]);
          l+=1;
        end
      end
      return v;
    else
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=(d2zphipx1[i]*phipy1[j]+phipx1[i]*d2zphipy1[j]+2*d1zphipx1[i]*d1zphipy1[j])+(d2zphipx2[i]*phipy2[j]+phipx2[i]*d2zphipy2[j]+2*d1zphipx2[i]*d1zphipy2[j]);
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
            v[lp]=Phix[1][k][i][ip]*Phiy[1][k][j][jp]+Phix[2][k][i][ip]*Phiy[2][k][j][jp];
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


  function correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Float64,1},1},1},1})
    a11=[dot(phix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    b11=[dot(phiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    a12=[dot(phix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    b12=[dot(phiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    a21=[dot(phix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    b21=[dot(phiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    a22=[dot(phix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    b22=[dot(phiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    return dot(a11,b11)+dot(a12,b12)+dot(a21,b21)+dot(a22,b22)-ob(x);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Float64,1},1},1},1})
    d1c=zeros(kernel.dim);
    a11=[dot(phix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    b11=[dot(phiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    a12=[dot(phix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    b12=[dot(phiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    a21=[dot(phix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    b21=[dot(phiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    a22=[dot(phix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    b22=[dot(phiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];

    da11=[dot(d1xphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    da12=[dot(d1xphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    db11=[dot(d1yphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    db12=[dot(d1yphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dc11=[dot(d1zphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dc12=[dot(d1zphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    dd11=[dot(d1zphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    dd12=[dot(d1zphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    da22=[dot(d1xphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    da21=[dot(d1xphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    db22=[dot(d1yphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    db21=[dot(d1yphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    dc22=[dot(d1zphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    dc21=[dot(d1zphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dd22=[dot(d1zphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dd21=[dot(d1zphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];

    d1c[1]=dot(da11,b11)+dot(da12,b12)+dot(da22,b22)+dot(da21,b21)-d1ob(1,x);
    d1c[2]=dot(a11,db11)+dot(a12,db12)+dot(a22,db22)+dot(a21,db21)-d1ob(2,x);
    d1c[3]=(dot(dc11,b11)+dot(a11,dd11))+(dot(dc22,b22)+dot(a22,dd22))+(dot(dc12,b12)+dot(a12,dd12))+(dot(dc21,b21)+dot(a21,dd21))-d1ob(3,x);
    return d1c
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Float64,1},1},1},1})
    d2c=zeros(kernel.dim,kernel.dim);
    a11=[dot(phix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    b11=[dot(phiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    a12=[dot(phix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    b12=[dot(phiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    a21=[dot(phix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    b21=[dot(phiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    a22=[dot(phix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    b22=[dot(phiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];

    da11=[dot(d1xphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    da12=[dot(d1xphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    db11=[dot(d1yphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    db12=[dot(d1yphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dc11=[dot(d1zphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dc12=[dot(d1zphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    dd11=[dot(d1zphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    dd12=[dot(d1zphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    da22=[dot(d1xphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    da21=[dot(d1xphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    db22=[dot(d1yphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    db21=[dot(d1yphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    dc22=[dot(d1zphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    dc21=[dot(d1zphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dd22=[dot(d1zphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dd21=[dot(d1zphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];


    dac11=[dot(d11xzphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dac12=[dot(d11xzphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    dbd11=[dot(d11yzphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    dbd12=[dot(d11yzphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dda11=[dot(d2xphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dda12=[dot(d2xphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    ddb11=[dot(d2yphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    ddb12=[dot(d2yphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    ddc11=[dot(d2zphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    ddc12=[dot(d2zphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    ddd11=[dot(d2zphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    ddd12=[dot(d2zphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];

    dac22=[dot(d11xzphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    dbd22=[dot(d11yzphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dda22=[dot(d2xphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    ddb22=[dot(d2yphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    ddc22=[dot(d2zphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    ddd22=[dot(d2zphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dac21=[dot(d11xzphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dbd21=[dot(d11yzphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    dda21=[dot(d2xphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    ddb21=[dot(d2yphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    ddc21=[dot(d2zphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    ddd21=[dot(d2zphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];

    d2c[1,2]=dot(da11,db11)+dot(da22,db22)+dot(da12,db12)+dot(da21,db21)-dot(d11phiVect(1,2,x),y);
    d2c[1,3]=(dot(da11,dd11)+dot(dac11,b11))+(dot(da22,dd22)+dot(dac22,b22))+(dot(da12,dd12)+dot(dac12,b12))+(dot(da21,dd21)+dot(dac21,b21))-dot(d11phiVect(1,3,x),y);
    d2c[2,3]=(dot(dc11,db11)+dot(a11,dbd11))+(dot(dc22,db22)+dot(a22,dbd22))+(dot(dc12,db12)+dot(a12,dbd12))+(dot(dc21,db21)+dot(a21,dbd21))-dot(d11phiVect(2,3,x),y);
    d2c=d2c+d2c';
    d2c[1,1]=dot(dda11,b11)+dot(dda22,b22)+dot(dda12,b12)+dot(dda21,b21)-d2ob(1,x);
    d2c[2,2]=dot(a11,ddb11)+dot(a22,ddb22)+dot(a12,ddb12)+dot(a21,ddb21)-d2ob(2,x);
    d2c[3,3]=(dot(ddc11,b11)+dot(a11,ddd11)+2*dot(dc11,dd11))+(dot(ddc22,b22)+dot(a22,ddd22)+2*dot(dc22,dd22))+(dot(ddc12,b12)+dot(a12,ddd12)+2*dot(dc12,dd12))+(dot(ddc21,b21)+dot(a21,ddd21)+2*dot(dc21,dd21))-d2ob(3,x);
    return(d2c)
  end

  gaussconv2DDoubleHelix(typeof(kernel),kernel.dim,kernel.bounds,normObs,Phix,Phiy,PhisY,phix,phiy,phiVect,d1phiVect,d11phiVect,d2phiVect,y,c,d10c,d01c,d11c,d20c,d02c,ob,d1ob,d11ob,d2ob,correl,d1correl,d2correl);
end

function setoperator(kernel::doubleHelix,y::Array{Float64,1})
  function phiDx(x::Float64,z::Float64,xi::Float64,s::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2);
    return .5*(erf(alphap/kernel.sigmax)-erf(alpham/kernel.sigmax));
  end
  function phiDy(y::Float64,z::Float64,yi::Float64,s::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2);
    return .5*(erf(alphap/kernel.sigmay)-erf(alpham/kernel.sigmay));
  end

  function d1xphiDx(x::Float64,z::Float64,xi::Float64,s::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmax)*(exp(-(alphap/kernel.sigmax)^2) - exp(-(alpham/kernel.sigmax)^2));
  end
  function d1yphiDy(y::Float64,z::Float64,yi::Float64,s::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmay)*(exp(-(alphap/kernel.sigmay)^2) - exp(-(alpham/kernel.sigmay)^2));
  end

  function d11xzphiDx(x::Float64,z::Float64,xi::Float64,s::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2);
    return -s*kernel.d1rotx(z)/(sqrt(pi)*kernel.sigmax^3)*(alphap*exp(-(alphap/kernel.sigmax)^2) - alpham*exp(-(alpham/kernel.sigmax)^2));
  end
  function d11yzphiDy(y::Float64,z::Float64,yi::Float64,s::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2);
    return -s*kernel.d1roty(z)/(sqrt(pi)*kernel.sigmay^3)*(alphap*exp(-(alphap/kernel.sigmay)^2) - alpham*exp(-(alpham/kernel.sigmay)^2));
  end

  function d2xphiDx(x::Float64,z::Float64,xi::Float64,s::Float64)
    return -1/(sqrt(pi)*kernel.sigmax^2)*( ((xi+kernel.Dpx/2-x-s*kernel.rotx(z))/(sqrt(2)*kernel.sigmax))*exp(-((xi+kernel.Dpx/2-x-s*kernel.rotx(z))/(sqrt(2)*kernel.sigmax))^2) - ((xi-kernel.Dpx/2-x-s*kernel.rotx(z))/(sqrt(2)*kernel.sigmax))*exp(-((xi-kernel.Dpx/2-x-s*kernel.rotx(z))/(sqrt(2)*kernel.sigmax))^2) );
  end
  function d2yphiDy(y::Float64,z::Float64,yi::Float64,s::Float64)
    return -1/(sqrt(pi)*kernel.sigmay^2)*( ((yi+kernel.Dpy/2-y-s*kernel.roty(z))/(sqrt(2)*kernel.sigmay))*exp(-((yi+kernel.Dpy/2-y-s*kernel.roty(z))/(sqrt(2)*kernel.sigmay))^2) - ((yi-kernel.Dpy/2-y-s*kernel.roty(z))/(sqrt(2)*kernel.sigmay))*exp(-((yi-kernel.Dpy/2-y-s*kernel.roty(z))/(sqrt(2)*kernel.sigmay))^2) );
  end

  function d1zphiDx(x::Float64,z::Float64,xi::Float64,s::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2);
    return -s*kernel.d1rotx(z)/(sqrt(2*pi)*kernel.sigmax)*(exp(-(alphap/kernel.sigmax)^2) - exp(-(alpham/kernel.sigmax)^2));
  end
  function d1zphiDy(y::Float64,z::Float64,yi::Float64,s::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2);
    return -s*kernel.d1roty(z)/(sqrt(2*pi)*kernel.sigmay)*(exp(-(alphap/kernel.sigmay)^2) - exp(-(alpham/kernel.sigmay)^2));
  end

  function d2zphiDx(x::Float64,z::Float64,xi::Float64,s::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z))/sqrt(2);
    return -s*kernel.d2rotx(z)/(sqrt(2*pi)*kernel.sigmax)*(exp(-(alphap/kernel.sigmax)^2) - exp(-(alpham/kernel.sigmax)^2)) - s^2*kernel.d1rotx(z)^2/(sqrt(pi)*kernel.sigmax^3)*(alphap*exp(-(alphap/kernel.sigmax)^2) - alpham*exp(-(alpham/kernel.sigmax)^2));
  end
  function d2zphiDy(y::Float64,z::Float64,yi::Float64,s::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z))/sqrt(2);
    return -s*kernel.d2roty(z)/(sqrt(2*pi)*kernel.sigmay)*(exp(-(alphap/kernel.sigmay)^2) - exp(-(alpham/kernel.sigmay)^2)) - s^2*kernel.d1roty(z)^2/(sqrt(pi)*kernel.sigmay^3)*(alphap*exp(-(alphap/kernel.sigmay)^2) - alpham*exp(-(alpham/kernel.sigmay)^2));
  end
  ##

  ##
  function phix(x::Float64,z::Float64,s::Float64)
    phipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      phipx[i]=phiDx(x,z,kernel.px[i],s);
    end
    return phipx;
  end
  #
  function phiy(y::Float64,z::Float64,s::Float64)
    phipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      phipy[i]=phiDy(y,z,kernel.py[i],s);
    end
    return phipy;
  end
  #

  Phix=[Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3]),Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3])];
  Phiy=[Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3]),Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3])];
  for k in 1:kernel.nbpointsgrid[3]
    Phix[1][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[1]);
    Phiy[1][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[2]);
    Phix[2][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[1]);
    Phiy[2][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[2]);
    for i in 1:kernel.nbpointsgrid[1]
      Phix[1][k][i]=phix(kernel.grid[1][i],kernel.grid[3][k],1.0);
      Phix[2][k][i]=phix(kernel.grid[1][i],kernel.grid[3][k],-1.0);
    end
    for j in 1:kernel.nbpointsgrid[2]
      Phiy[1][k][j]=phiy(kernel.grid[2][j],kernel.grid[3][k],1.0);
      Phiy[2][k][j]=phiy(kernel.grid[2][j],kernel.grid[3][k],-1.0);
    end
  end

  #
  function d1xphix(x::Float64,z::Float64,s::Float64)
    d1xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1xphipx[i]=d1xphiDx(x,z,kernel.px[i],s);
    end
    return d1xphipx;
  end
  #
  function d1yphiy(y::Float64,z::Float64,s::Float64)
    d1yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1yphipy[i]=d1yphiDy(y,z,kernel.py[i],s);
    end
    return d1yphipy;
  end
  #
  function d1zphix(x::Float64,z::Float64,s::Float64)
    d1zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1zphipx[i]=d1zphiDx(x,z,kernel.px[i],s);
    end
    return d1zphipx;
  end
  #
  function d1zphiy(y::Float64,z::Float64,s::Float64)
    d1zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1zphipy[i]=d1zphiDy(y,z,kernel.py[i],s);
    end
    return d1zphipy;
  end
  #

  #
  function d11xzphix(x::Float64,z::Float64,s::Float64)
    d11xzphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d11xzphipx[i]=d11xzphiDx(x,z,kernel.px[i],s);
    end
    return d11xzphipx
  end
  #
  function d11yzphiy(y::Float64,z::Float64,s::Float64)
    d11yzphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d11yzphipy[i]=d11yzphiDy(y,z,kernel.py[i],s);
    end
    return d11yzphipy
  end
  #

  #
  function d2xphix(x::Float64,z::Float64,s::Float64)
    d2xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2xphipx[i]=d2xphiDx(x,z,kernel.px[i],s);
    end
    return d2xphipx;
  end
  function d2yphiy(y::Float64,z::Float64,s::Float64)
    d2yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2yphipy[i]=d2yphiDy(y,z,kernel.py[i],s);
    end
    return d2yphipy;
  end
  #
  function d2zphix(x::Float64,z::Float64,s::Float64)
    d2zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2zphipx[i]=d2zphiDx(x,z,kernel.px[i],s);
    end
    return d2zphipx;
  end
  function d2zphiy(y::Float64,z::Float64,s::Float64)
    d2zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2zphipy[i]=d2zphiDy(y,z,kernel.py[i],s);
    end
    return d2zphipy;
  end
  #
  ##

  ##
  function phiVect(x::Array{Float64,1})
    phipx1=phix(x[1],x[3],1.0);
    phipy1=phiy(x[2],x[3],1.0);
    phipx2=phix(x[1],x[3],-1.0);
    phipy2=phiy(x[2],x[3],-1.0);
    v=zeros(kernel.Npx*kernel.Npy);
    l=1;
    for j in 1:kernel.Npy
      for i in 1:kernel.Npx
        v[l]=phipx1[i]*phipy1[j]+phipx2[i]*phipy2[j];
        l+=1;
      end
    end
    return v;
  end
  function d1phiVect(m::Int64,x::Array{Float64,1})
    #
    phipx1=phix(x[1],x[3],1.0);
    phipy1=phiy(x[2],x[3],1.0);
    phipx2=phix(x[1],x[3],-1.0);
    phipy2=phiy(x[2],x[3],-1.0);
    #
    d1xphipx1=d1xphix(x[1],x[3],1.0);
    d1yphipy1=d1yphiy(x[2],x[3],1.0);
    d1zphipx1=d1zphix(x[1],x[3],1.0);
    d1zphipy1=d1zphiy(x[2],x[3],1.0);
    d1xphipx2=d1xphix(x[1],x[3],-1.0);
    d1yphipy2=d1yphiy(x[2],x[3],-1.0);
    d1zphipx2=d1zphix(x[1],x[3],-1.0);
    d1zphipy2=d1zphiy(x[2],x[3],-1.0);
    #
    v=zeros(kernel.Npx*kernel.Npy);
    if m==1
      l=1;
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d1xphipx1[i]*phipy1[j]+d1xphipx2[i]*phipy2[j];
          l+=1;
        end
      end
      return v;
    elseif m==2
      l=1;
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=phipx1[i]*d1yphipy1[j]+phipx2[i]*d1yphipy2[j];
          l+=1;
        end
      end
      return v;
    else
      l=1;
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=(d1zphipx1[i]*phipy1[j]+phipx1[i]*d1zphipy1[j])+(d1zphipx2[i]*phipy2[j]+phipx2[i]*d1zphipy2[j]);
          l+=1;
        end
      end
      return v;
    end
  end
  function d11phiVect(m::Int64,n::Int64,x::Array{Float64,1})
    #
    phipx1=phix(x[1],x[3],1.0);
    phipy1=phiy(x[2],x[3],1.0);
    phipx2=phix(x[1],x[3],-1.0);
    phipy2=phiy(x[2],x[3],-1.0);
    #
    d1xphipx1=d1xphix(x[1],x[3],1.0);
    d1yphipy1=d1yphiy(x[2],x[3],1.0);
    d1zphipx1=d1zphix(x[1],x[3],1.0);
    d1zphipy1=d1zphiy(x[2],x[3],1.0);
    d1xphipx2=d1xphix(x[1],x[3],-1.0);
    d1yphipy2=d1yphiy(x[2],x[3],-1.0);
    d1zphipx2=d1zphix(x[1],x[3],-1.0);
    d1zphipy2=d1zphiy(x[2],x[3],-1.0);
    #
    d11xzphipx1=d11xzphix(x[1],x[3],1.0);
    d11yzphipy1=d11yzphiy(x[2],x[3],1.0);
    d11xzphipx2=d11xzphix(x[1],x[3],-1.0);
    d11yzphipy2=d11yzphiy(x[2],x[3],-1.0);
    #
    v=zeros(kernel.Npx*kernel.Npy);
    l=1;
    if m==1 && n==2 || m==2 && n==1
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=d1xphipx1[i]*d1yphipy1[j]+d1xphipx2[i]*d1yphipy2[j];
          l+=1;
        end
      end
      return v;
    elseif m==1 && n==3 || m==3 && n==1
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=(d11xzphipx1[i]*phipy1[j]+d1xphipx1[i]*d1zphipy1[j])+(d11xzphipx2[i]*phipy2[j]+d1xphipx2[i]*d1zphipy2[j]);
          l+=1;
        end
      end
      return v;
    elseif m==2 && n==3 || m==3 && n==2
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=(phipx1[i]*d11yzphipy1[j]+d1zphipx1[i]*d1yphipy1[j])+(phipx2[i]*d11yzphipy2[j]+d1zphipx2[i]*d1yphipy2[j]);
          l+=1;
        end
      end
      return v;
    end
  end
  function d2phiVect(m::Int64,x::Array{Float64,1})
    #
    phipx1=phix(x[1],x[3],1.0);
    phipy1=phiy(x[2],x[3],1.0);
    phipx2=phix(x[1],x[3],-1.0);
    phipy2=phiy(x[2],x[3],-1.0);
    #
    d1zphipx1=d1zphix(x[1],x[3],1.0);
    d1zphipy1=d1zphiy(x[2],x[3],1.0);
    d1zphipx2=d1zphix(x[1],x[3],-1.0);
    d1zphipy2=d1zphiy(x[2],x[3],-1.0);
    #
    d2xphipx1=d2xphix(x[1],x[3],1.0);
    d2yphipy1=d2yphiy(x[2],x[3],1.0);
    d2zphipx1=d2zphix(x[1],x[3],1.0);
    d2zphipy1=d2zphiy(x[2],x[3],1.0);
    d2xphipx2=d2xphix(x[1],x[3],-1.0);
    d2yphipy2=d2yphiy(x[2],x[3],-1.0);
    d2zphipx2=d2zphix(x[1],x[3],-1.0);
    d2zphipy2=d2zphiy(x[2],x[3],-1.0);
    #
    v=zeros(kernel.Npx*kernel.Npy);
    l=1;
    if m==1
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=(d2xphipx1[i]*phipy1[j])+(d2xphipx2[i]*phipy2[j]);
          l+=1;
        end
      end
      return v;
    elseif m==2
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=(phipx1[i]*d2yphipy1[j])+(phipx2[i]*d2yphipy2[j]);
          l+=1;
        end
      end
      return v;
    else
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=(d2zphipx1[i]*phipy1[j]+phipx1[i]*d2zphipy1[j]+2*d1zphipx1[i]*d1zphipy1[j])+(d2zphipx2[i]*phipy2[j]+phipx2[i]*d2zphipy2[j]+2*d1zphipx2[i]*d1zphipy2[j]);
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
            v[lp]=Phix[1][k][i][ip]*Phiy[1][k][j][jp]+Phix[2][k][i][ip]*Phiy[2][k][j][jp];
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


  function correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Float64,1},1},1},1})
    a11=[dot(phix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    b11=[dot(phiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    a12=[dot(phix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    b12=[dot(phiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    a21=[dot(phix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    b21=[dot(phiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    a22=[dot(phix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    b22=[dot(phiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    return dot(a11,b11)+dot(a12,b12)+dot(a21,b21)+dot(a22,b22)-ob(x);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Float64,1},1},1},1})
    d1c=zeros(kernel.dim);
    a11=[dot(phix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    b11=[dot(phiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    a12=[dot(phix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    b12=[dot(phiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    a21=[dot(phix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    b21=[dot(phiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    a22=[dot(phix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    b22=[dot(phiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];

    da11=[dot(d1xphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    da12=[dot(d1xphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    db11=[dot(d1yphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    db12=[dot(d1yphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dc11=[dot(d1zphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dc12=[dot(d1zphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    dd11=[dot(d1zphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    dd12=[dot(d1zphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    da22=[dot(d1xphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    da21=[dot(d1xphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    db22=[dot(d1yphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    db21=[dot(d1yphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    dc22=[dot(d1zphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    dc21=[dot(d1zphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dd22=[dot(d1zphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dd21=[dot(d1zphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];

    d1c[1]=dot(da11,b11)+dot(da12,b12)+dot(da22,b22)+dot(da21,b21)-d1ob(1,x);
    d1c[2]=dot(a11,db11)+dot(a12,db12)+dot(a22,db22)+dot(a21,db21)-d1ob(2,x);
    d1c[3]=(dot(dc11,b11)+dot(a11,dd11))+(dot(dc22,b22)+dot(a22,dd22))+(dot(dc12,b12)+dot(a12,dd12))+(dot(dc21,b21)+dot(a21,dd21))-d1ob(3,x);
    return d1c
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Float64,1},1},1},1})
    d2c=zeros(kernel.dim,kernel.dim);
    a11=[dot(phix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    b11=[dot(phiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    a12=[dot(phix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    b12=[dot(phiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    a21=[dot(phix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    b21=[dot(phiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    a22=[dot(phix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    b22=[dot(phiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];

    da11=[dot(d1xphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    da12=[dot(d1xphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    db11=[dot(d1yphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    db12=[dot(d1yphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dc11=[dot(d1zphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dc12=[dot(d1zphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    dd11=[dot(d1zphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    dd12=[dot(d1zphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    da22=[dot(d1xphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    da21=[dot(d1xphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    db22=[dot(d1yphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    db21=[dot(d1yphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    dc22=[dot(d1zphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    dc21=[dot(d1zphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dd22=[dot(d1zphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dd21=[dot(d1zphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];


    dac11=[dot(d11xzphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dac12=[dot(d11xzphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    dbd11=[dot(d11yzphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    dbd12=[dot(d11yzphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dda11=[dot(d2xphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dda12=[dot(d2xphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    ddb11=[dot(d2yphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    ddb12=[dot(d2yphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    ddc11=[dot(d2zphix(x[1],x[3],1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    ddc12=[dot(d2zphix(x[1],x[3],1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    ddd11=[dot(d2zphiy(x[2],x[3],1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    ddd12=[dot(d2zphiy(x[2],x[3],1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];

    dac22=[dot(d11xzphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    dbd22=[dot(d11yzphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dda22=[dot(d2xphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    ddb22=[dot(d2yphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    ddc22=[dot(d2zphix(x[1],x[3],-1.0),Phiu[2][1][i]) for i in 1:length(Phiu[2][1])];
    ddd22=[dot(d2zphiy(x[2],x[3],-1.0),Phiu[2][2][i]) for i in 1:length(Phiu[2][2])];
    dac21=[dot(d11xzphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    dbd21=[dot(d11yzphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    dda21=[dot(d2xphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    ddb21=[dot(d2yphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];
    ddc21=[dot(d2zphix(x[1],x[3],-1.0),Phiu[1][1][i]) for i in 1:length(Phiu[1][1])];
    ddd21=[dot(d2zphiy(x[2],x[3],-1.0),Phiu[1][2][i]) for i in 1:length(Phiu[1][2])];

    d2c[1,2]=dot(da11,db11)+dot(da22,db22)+dot(da12,db12)+dot(da21,db21)-dot(d11phiVect(1,2,x),y);
    d2c[1,3]=(dot(da11,dd11)+dot(dac11,b11))+(dot(da22,dd22)+dot(dac22,b22))+(dot(da12,dd12)+dot(dac12,b12))+(dot(da21,dd21)+dot(dac21,b21))-dot(d11phiVect(1,3,x),y);
    d2c[2,3]=(dot(dc11,db11)+dot(a11,dbd11))+(dot(dc22,db22)+dot(a22,dbd22))+(dot(dc12,db12)+dot(a12,dbd12))+(dot(dc21,db21)+dot(a21,dbd21))-dot(d11phiVect(2,3,x),y);
    d2c=d2c+d2c';
    d2c[1,1]=dot(dda11,b11)+dot(dda22,b22)+dot(dda12,b12)+dot(dda21,b21)-d2ob(1,x);
    d2c[2,2]=dot(a11,ddb11)+dot(a22,ddb22)+dot(a12,ddb12)+dot(a21,ddb21)-d2ob(2,x);
    d2c[3,3]=(dot(ddc11,b11)+dot(a11,ddd11)+2*dot(dc11,dd11))+(dot(ddc22,b22)+dot(a22,ddd22)+2*dot(dc22,dd22))+(dot(ddc12,b12)+dot(a12,ddd12)+2*dot(dc12,dd12))+(dot(ddc21,b21)+dot(a21,ddd21)+2*dot(dc21,dd21))-d2ob(3,x);
    return(d2c)
  end

  gaussconv2DDoubleHelix(typeof(kernel),kernel.dim,kernel.bounds,normObs,Phix,Phiy,PhisY,phix,phiy,phiVect,d1phiVect,d11phiVect,d2phiVect,y,c,d10c,d01c,d11c,d20c,d02c,ob,d1ob,d11ob,d2ob,correl,d1correl,d2correl);
end

function computePhiu(u::Array{Float64,1},op::blasso.gaussconv2DDoubleHelix)
  a,X=blasso.decompAmpPos(u,d=op.dim);
  Phiux1=[a[i]*op.phix(X[i][1],X[i][3],1.0) for i in 1:length(a)];
  Phiuy1=[op.phiy(X[i][2],X[i][3],1.0) for i in 1:length(a)];
  Phiux2=[a[i]*op.phix(X[i][1],X[i][3],-1.0) for i in 1:length(a)];
  Phiuy2=[op.phiy(X[i][2],X[i][3],-1.0) for i in 1:length(a)];
  return [[Phiux1,Phiuy1],[Phiux2,Phiuy2]];
end

# Compute the argmin and min of the correl on the grid.
function minCorrelOnGrid(Phiu::Array{Array{Array{Array{Float64,1},1},1},1},kernel::blasso.doubleHelix,op::blasso.operator,positivity::Bool=true)
  correl_min,argmin=Inf,zeros(op.dim);
  l=1;
  for k in 1:kernel.nbpointsgrid[3]
    for i in 1:kernel.nbpointsgrid[1]
      a11=[dot(op.Phix[1][k][i],Phiu[1][1][m]) for m in 1:length(Phiu[1][1])];
      a12=[dot(op.Phix[1][k][i],Phiu[2][1][m]) for m in 1:length(Phiu[2][1])];
      a22=[dot(op.Phix[2][k][i],Phiu[2][1][m]) for m in 1:length(Phiu[2][1])];
      a21=[dot(op.Phix[2][k][i],Phiu[1][1][m]) for m in 1:length(Phiu[1][1])];
      for j in 1:kernel.nbpointsgrid[2]
        b11=[dot(op.Phiy[1][k][j],Phiu[1][2][m]) for m in 1:length(Phiu[1][2])];
        b12=[dot(op.Phiy[1][k][j],Phiu[2][2][m]) for m in 1:length(Phiu[2][2])];
        b22=[dot(op.Phiy[2][k][j],Phiu[2][2][m]) for m in 1:length(Phiu[2][2])];
        b21=[dot(op.Phiy[2][k][j],Phiu[1][2][m]) for m in 1:length(Phiu[1][2])];
        buffer=dot(a11,b11)+dot(a22,b22)+dot(a12,b12)+dot(a21,b21)-op.PhisY[l];
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

function setbounds(op::blasso.gaussconv2DDoubleHelix,positivity::Bool=true,ampbounds::Bool=true)
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
