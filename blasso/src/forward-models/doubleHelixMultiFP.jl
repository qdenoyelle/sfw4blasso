mutable struct doubleHelixMultiFP <: discrete
  dim::Int64
  px::Array{Float64,1}
  py::Array{Float64,1}
  p::Array{Array{Float64,1},1}
  Npx::Int64
  Npy::Int64
  Dpx::Float64
  Dpy::Float64
  K::Int64
  fp::Array{Float64,1}

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

function setKernel(Npx::Int64,Npy::Int64,Dpx::Float64,Dpy::Float64,fp::Array{Float64,1},wdth::Float64,sigma::Array{Float64,1},bounds::Array{Array{Float64,1},1})
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

  fpmax=maximum(fp);

  theta(z::Float64,f::Float64)=pi/3*(z-f)/fpmax;
  d1theta(z::Float64,f::Float64)=(pi/3)/fpmax;
  d2theta(z::Float64,f::Float64)=0.0;

  #theta(z::Float64,fp::Float64)=pi/3*(z-fp)/(bounds[2][end]-fp);
  #d1theta(z::Float64,fp::Float64)=(pi/3)/(bounds[2][end]-fp);
  #d2theta(z::Float64,fp::Float64)=0.0;

  rotx(z::Float64,fp::Float64)=wdth/2*cos(theta(z,fp));
  roty(z::Float64,fp::Float64)=-wdth/2*sin(theta(z,fp));
  d1rotx(z::Float64,fp::Float64)=-wdth/2*d1theta(z,fp)*sin(theta(z,fp));
  d1roty(z::Float64,fp::Float64)=-wdth/2*d1theta(z,fp)*cos(theta(z,fp));
  d2rotx(z::Float64,fp::Float64)=-wdth/2*d1theta(z,fp)^2*cos(theta(z,fp));
  d2roty(z::Float64,fp::Float64)=wdth/2*d1theta(z,fp)^2*sin(theta(z,fp));

  coeff=.2;
  lSample=[coeff/sigmax,coeff/sigmay,9];
  nbpointsgrid=[convert(Int64,round(5*lSample[i]*abs(b[i]-a[i]);digits=0)) for i in 1:3];
  g=Array{Array{Float64,1}}(undef,dim);
  for i in 1:3
    g[i]=collect(range(a[i],stop=b[i],length=nbpointsgrid[i]));
  end

  return doubleHelixMultiFP(dim,px,py,p,Npx,Npy,Dpx,Dpy,length(fp),fp,nbpointsgrid,g,sigmax,sigmay,rotx,roty,d1rotx,d1roty,d2rotx,d2roty,bounds)
end

mutable struct gaussconv2DDoubleHelixMultiFP <: operator
    ker::DataType
    dim::Int64
    bounds::Array{Array{Float64,1},1}
    K::Int64
    fp::Array{Float64,1}

    normObs::Float64

    Phix::Array{Array{Array{Array{Array{Float64,1},1},1},1},1}
    Phiy::Array{Array{Array{Array{Array{Float64,1},1},1},1},1}
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
function setoperator(kernel::doubleHelixMultiFP,a0::Array{Float64,1},x0::Array{Array{Float64,1},1})
  function phiDx(x::Float64,z::Float64,xi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2);
    return .5*(erf(alphap/kernel.sigmax)-erf(alpham/kernel.sigmax));
  end
  function phiDy(y::Float64,z::Float64,yi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2);
    return .5*(erf(alphap/kernel.sigmay)-erf(alpham/kernel.sigmay));
  end

  function d1xphiDx(x::Float64,z::Float64,xi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmax)*(exp(-(alphap/kernel.sigmax)^2) - exp(-(alpham/kernel.sigmax)^2));
  end
  function d1yphiDy(y::Float64,z::Float64,yi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmay)*(exp(-(alphap/kernel.sigmay)^2) - exp(-(alpham/kernel.sigmay)^2));
  end

  function d11xzphiDx(x::Float64,z::Float64,xi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2);
    return -s*kernel.d1rotx(z,fp)/(sqrt(pi)*kernel.sigmax^3)*(alphap*exp(-(alphap/kernel.sigmax)^2) - alpham*exp(-(alpham/kernel.sigmax)^2));
  end
  function d11yzphiDy(y::Float64,z::Float64,yi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2);
    return -s*kernel.d1roty(z,fp)/(sqrt(pi)*kernel.sigmay^3)*(alphap*exp(-(alphap/kernel.sigmay)^2) - alpham*exp(-(alpham/kernel.sigmay)^2));
  end

  function d2xphiDx(x::Float64,z::Float64,xi::Float64,s::Float64,fp::Float64)
    return -1/(sqrt(pi)*kernel.sigmax^2)*( ((xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/(sqrt(2)*kernel.sigmax))*exp(-((xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/(sqrt(2)*kernel.sigmax))^2) - ((xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/(sqrt(2)*kernel.sigmax))*exp(-((xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/(sqrt(2)*kernel.sigmax))^2) );
  end
  function d2yphiDy(y::Float64,z::Float64,yi::Float64,s::Float64,fp::Float64)
    return -1/(sqrt(pi)*kernel.sigmay^2)*( ((yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/(sqrt(2)*kernel.sigmay))*exp(-((yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/(sqrt(2)*kernel.sigmay))^2) - ((yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/(sqrt(2)*kernel.sigmay))*exp(-((yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/(sqrt(2)*kernel.sigmay))^2) );
  end

  function d1zphiDx(x::Float64,z::Float64,xi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2);
    return -s*kernel.d1rotx(z,fp)/(sqrt(2*pi)*kernel.sigmax)*(exp(-(alphap/kernel.sigmax)^2) - exp(-(alpham/kernel.sigmax)^2));
  end
  function d1zphiDy(y::Float64,z::Float64,yi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2);
    return -s*kernel.d1roty(z,fp)/(sqrt(2*pi)*kernel.sigmay)*(exp(-(alphap/kernel.sigmay)^2) - exp(-(alpham/kernel.sigmay)^2));
  end

  function d2zphiDx(x::Float64,z::Float64,xi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2);
    return -s*kernel.d2rotx(z,fp)/(sqrt(2*pi)*kernel.sigmax)*(exp(-(alphap/kernel.sigmax)^2) - exp(-(alpham/kernel.sigmax)^2)) - s^2*kernel.d1rotx(z,fp)^2/(sqrt(pi)*kernel.sigmax^3)*(alphap*exp(-(alphap/kernel.sigmax)^2) - alpham*exp(-(alpham/kernel.sigmax)^2));
  end
  function d2zphiDy(y::Float64,z::Float64,yi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2);
    return -s*kernel.d2roty(z,fp)/(sqrt(2*pi)*kernel.sigmay)*(exp(-(alphap/kernel.sigmay)^2) - exp(-(alpham/kernel.sigmay)^2)) - s^2*kernel.d1roty(z,fp)^2/(sqrt(pi)*kernel.sigmay^3)*(alphap*exp(-(alphap/kernel.sigmay)^2) - alpham*exp(-(alpham/kernel.sigmay)^2));
  end
  ##

  ##
  function phix(x::Float64,z::Float64,s::Float64,fp::Float64)
    phipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      phipx[i]=phiDx(x,z,kernel.px[i],s,fp);
    end
    return phipx;
  end
  #
  function phiy(y::Float64,z::Float64,s::Float64,fp::Float64)
    phipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      phipy[i]=phiDy(y,z,kernel.py[i],s,fp);
    end
    return phipy;
  end
  #

  Phix=[Array{Array{Array{Array{Float64,1},1},1}}(kernel.K),Array{Array{Array{Array{Float64,1},1},1}}(kernel.K)];
  Phiy=[Array{Array{Array{Array{Float64,1},1},1}}(kernel.K),Array{Array{Array{Array{Float64,1},1},1}}(kernel.K)];
  for l in 1:kernel.K
    Phix[1][l]=Array{Array{Array{Float64,1},1},1}(kernel.nbpointsgrid[3]);
    Phiy[1][l]=Array{Array{Array{Float64,1},1},1}(kernel.nbpointsgrid[3]);
    Phix[2][l]=Array{Array{Array{Float64,1},1},1}(kernel.nbpointsgrid[3]);
    Phiy[2][l]=Array{Array{Array{Float64,1},1},1}(kernel.nbpointsgrid[3]);
    for k in 1:kernel.nbpointsgrid[3]
      Phix[1][l][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[1]);
      Phiy[1][l][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[2]);
      Phix[2][l][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[1]);
      Phiy[2][l][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[2]);
      for i in 1:kernel.nbpointsgrid[1]
        Phix[1][l][k][i]=phix(kernel.grid[1][i],kernel.grid[3][k],1.0,kernel.fp[l]);
        Phix[2][l][k][i]=phix(kernel.grid[1][i],kernel.grid[3][k],-1.0,kernel.fp[l]);
      end
      for j in 1:kernel.nbpointsgrid[2]
        Phiy[1][l][k][j]=phiy(kernel.grid[2][j],kernel.grid[3][k],1.0,kernel.fp[l]);
        Phiy[2][l][k][j]=phiy(kernel.grid[2][j],kernel.grid[3][k],-1.0,kernel.fp[l]);
      end
    end
  end

  #
  function d1xphix(x::Float64,z::Float64,s::Float64,fp::Float64)
    d1xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1xphipx[i]=d1xphiDx(x,z,kernel.px[i],s,fp);
    end
    return d1xphipx;
  end
  #
  function d1yphiy(y::Float64,z::Float64,s::Float64,fp::Float64)
    d1yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1yphipy[i]=d1yphiDy(y,z,kernel.py[i],s,fp);
    end
    return d1yphipy;
  end
  #
  function d1zphix(x::Float64,z::Float64,s::Float64,fp::Float64)
    d1zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1zphipx[i]=d1zphiDx(x,z,kernel.px[i],s,fp);
    end
    return d1zphipx;
  end
  #
  function d1zphiy(y::Float64,z::Float64,s::Float64,fp::Float64)
    d1zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1zphipy[i]=d1zphiDy(y,z,kernel.py[i],s,fp);
    end
    return d1zphipy;
  end
  #

  #
  function d11xzphix(x::Float64,z::Float64,s::Float64,fp::Float64)
    d11xzphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d11xzphipx[i]=d11xzphiDx(x,z,kernel.px[i],s,fp);
    end
    return d11xzphipx
  end
  #
  function d11yzphiy(y::Float64,z::Float64,s::Float64,fp::Float64)
    d11yzphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d11yzphipy[i]=d11yzphiDy(y,z,kernel.py[i],s,fp);
    end
    return d11yzphipy
  end
  #

  #
  function d2xphix(x::Float64,z::Float64,s::Float64,fp::Float64)
    d2xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2xphipx[i]=d2xphiDx(x,z,kernel.px[i],s,fp);
    end
    return d2xphipx;
  end
  function d2yphiy(y::Float64,z::Float64,s::Float64,fp::Float64)
    d2yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2yphipy[i]=d2yphiDy(y,z,kernel.py[i],s,fp);
    end
    return d2yphipy;
  end
  #
  function d2zphix(x::Float64,z::Float64,s::Float64,fp::Float64)
    d2zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2zphipx[i]=d2zphiDx(x,z,kernel.px[i],s,fp);
    end
    return d2zphipx;
  end
  function d2zphiy(y::Float64,z::Float64,s::Float64,fp::Float64)
    d2zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2zphipy[i]=d2zphiDy(y,z,kernel.py[i],s,fp);
    end
    return d2zphipy;
  end
  #
  ##

  ##
  function phiVect(x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy*kernel.K);
    l=1;
    for k in 1:kernel.K
      phipx1=phix(x[1],x[3],1.0,kernel.fp[k]);
      phipy1=phiy(x[2],x[3],1.0,kernel.fp[k]);
      phipx2=phix(x[1],x[3],-1.0,kernel.fp[k]);
      phipy2=phiy(x[2],x[3],-1.0,kernel.fp[k]);
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=phipx1[i]*phipy1[j]+phipx2[i]*phipy2[j];
          l+=1;
        end
      end
    end
    return v;
  end
  function d1phiVect(m::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy*kernel.K);
    l=1;
    if m==1
      for k in 1:kernel.K
        #
        phipy1=phiy(x[2],x[3],1.0,kernel.fp[k]);
        phipy2=phiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d1xphipx1=d1xphix(x[1],x[3],1.0,kernel.fp[k]);
        d1xphipx2=d1xphix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d1xphipx1[i]*phipy1[j]+d1xphipx2[i]*phipy2[j];
            l+=1;
          end
        end
      end
      return v;
    elseif m==2
      for k in 1:kernel.K
        #
        phipx1=phix(x[1],x[3],1.0,kernel.fp[k]);
        phipx2=phix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        d1yphipy1=d1yphiy(x[2],x[3],1.0,kernel.fp[k]);
        d1yphipy2=d1yphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=phipx1[i]*d1yphipy1[j]+phipx2[i]*d1yphipy2[j];
            l+=1;
          end
        end
      end
      return v;
    else
      for k in 1:kernel.K
        #
        phipx1=phix(x[1],x[3],1.0,kernel.fp[k]);
        phipy1=phiy(x[2],x[3],1.0,kernel.fp[k]);
        phipx2=phix(x[1],x[3],-1.0,kernel.fp[k]);
        phipy2=phiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d1zphipx1=d1zphix(x[1],x[3],1.0,kernel.fp[k]);
        d1zphipy1=d1zphiy(x[2],x[3],1.0,kernel.fp[k]);
        d1zphipx2=d1zphix(x[1],x[3],-1.0,kernel.fp[k]);
        d1zphipy2=d1zphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=(d1zphipx1[i]*phipy1[j]+phipx1[i]*d1zphipy1[j])+(d1zphipx2[i]*phipy2[j]+phipx2[i]*d1zphipy2[j]);
            l+=1;
          end
        end
      end
      return v;
    end
  end
  function d11phiVect(m::Int64,n::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy*kernel.K);
    l=1;
    if m==1 && n==2 || m==2 && n==1
      for k in 1:kernel.K
        #
        d1xphipx1=d1xphix(x[1],x[3],1.0,kernel.fp[k]);
        d1yphipy1=d1yphiy(x[2],x[3],1.0,kernel.fp[k]);
        d1xphipx2=d1xphix(x[1],x[3],-1.0,kernel.fp[k]);
        d1yphipy2=d1yphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d1xphipx1[i]*d1yphipy1[j]+d1xphipx2[i]*d1yphipy2[j];
            l+=1;
          end
        end
      end
      return v;
    elseif m==1 && n==3 || m==3 && n==1
      for k in 1:kernel.K
        #
        phipy1=phiy(x[2],x[3],1.0,kernel.fp[k]);
        phipy2=phiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d1xphipx1=d1xphix(x[1],x[3],1.0,kernel.fp[k]);
        d1zphipy1=d1zphiy(x[2],x[3],1.0,kernel.fp[k]);
        d1xphipx2=d1xphix(x[1],x[3],-1.0,kernel.fp[k]);
        d1zphipy2=d1zphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d11xzphipx1=d11xzphix(x[1],x[3],1.0,kernel.fp[k]);
        d11xzphipx2=d11xzphix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=(d11xzphipx1[i]*phipy1[j]+d1xphipx1[i]*d1zphipy1[j])+(d11xzphipx2[i]*phipy2[j]+d1xphipx2[i]*d1zphipy2[j]);
            l+=1;
          end
        end
      end
      return v;
    elseif m==2 && n==3 || m==3 && n==2
      for k in 1:kernel.K
        #
        phipx1=phix(x[1],x[3],1.0,kernel.fp[k]);
        phipx2=phix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        d1yphipy1=d1yphiy(x[2],x[3],1.0,kernel.fp[k]);
        d1zphipx1=d1zphix(x[1],x[3],1.0,kernel.fp[k]);
        d1yphipy2=d1yphiy(x[2],x[3],-1.0,kernel.fp[k]);
        d1zphipx2=d1zphix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        d11yzphipy1=d11yzphiy(x[2],x[3],1.0,kernel.fp[k]);
        d11yzphipy2=d11yzphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=(phipx1[i]*d11yzphipy1[j]+d1zphipx1[i]*d1yphipy1[j])+(phipx2[i]*d11yzphipy2[j]+d1zphipx2[i]*d1yphipy2[j]);
            l+=1;
          end
        end
      end
      return v;
    end
  end
  function d2phiVect(m::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy*kernel.K);
    l=1;
    if m==1
      for k in 1:kernel.K
        #
        phipy1=phiy(x[2],x[3],1.0,kernel.fp[k]);
        phipy2=phiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d2xphipx1=d2xphix(x[1],x[3],1.0,kernel.fp[k]);
        d2xphipx2=d2xphix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=(d2xphipx1[i]*phipy1[j])+(d2xphipx2[i]*phipy2[j]);
            l+=1;
          end
        end
      end
      return v;
    elseif m==2
      for k in 1:kernel.K
        #
        phipx1=phix(x[1],x[3],1.0,kernel.fp[k]);
        phipx2=phix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        d2yphipy1=d2yphiy(x[2],x[3],1.0,kernel.fp[k]);
        d2yphipy2=d2yphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=(phipx1[i]*d2yphipy1[j])+(phipx2[i]*d2yphipy2[j]);
            l+=1;
          end
        end
      end
      return v;
    else
      for k in 1:kernel.K
        #
        phipx1=phix(x[1],x[3],1.0,kernel.fp[k]);
        phipy1=phiy(x[2],x[3],1.0,kernel.fp[k]);
        phipx2=phix(x[1],x[3],-1.0,kernel.fp[k]);
        phipy2=phiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d1zphipx1=d1zphix(x[1],x[3],1.0,kernel.fp[k]);
        d1zphipy1=d1zphiy(x[2],x[3],1.0,kernel.fp[k]);
        d1zphipx2=d1zphix(x[1],x[3],-1.0,kernel.fp[k]);
        d1zphipy2=d1zphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d2zphipx1=d2zphix(x[1],x[3],1.0,kernel.fp[k]);
        d2zphipy1=d2zphiy(x[2],x[3],1.0,kernel.fp[k]);
        d2zphipx2=d2zphix(x[1],x[3],-1.0,kernel.fp[k]);
        d2zphipy2=d2zphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=(d2zphipx1[i]*phipy1[j]+phipx1[i]*d2zphipy1[j]+2*d1zphipx1[i]*d1zphipy1[j])+(d2zphipx2[i]*phipy2[j]+phipx2[i]*d2zphipy2[j]+2*d1zphipx2[i]*d1zphipy2[j]);
            l+=1;
          end
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
        v=zeros(kernel.Npx*kernel.Npy*kernel.K);
        lp=1;
        for kp in 1:kernel.K
          for jp in 1:kernel.Npy
            for ip in 1:kernel.Npx
              v[lp]=Phix[1][kp][k][i][ip]*Phiy[1][kp][k][j][jp]+Phix[2][kp][k][i][ip]*Phiy[2][kp][k][j][jp];
              lp+=1;
            end
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


  function correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Array{Float64,1},1},1},1},1})
    a11=[[dot(phix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    b11=[[dot(phiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    a12=[[dot(phix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    b12=[[dot(phiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    a21=[[dot(phix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    b21=[[dot(phiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    a22=[[dot(phix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    b22=[[dot(phiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    return sum([dot(a11[i],b11[i])+dot(a12[i],b12[i])+dot(a21[i],b21[i])+dot(a22[i],b22[i]) for i in 1:kernel.K])-ob(x);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Array{Float64,1},1},1},1},1})
    d1c=zeros(kernel.dim);
    a11=[[dot(phix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    b11=[[dot(phiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    a12=[[dot(phix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    b12=[[dot(phiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    a21=[[dot(phix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    b21=[[dot(phiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    a22=[[dot(phix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    b22=[[dot(phiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];

    da11=[[dot(d1xphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    da12=[[dot(d1xphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    db11=[[dot(d1yphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    db12=[[dot(d1yphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dc11=[[dot(d1zphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dc12=[[dot(d1zphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    dd11=[[dot(d1zphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    dd12=[[dot(d1zphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    da22=[[dot(d1xphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    da21=[[dot(d1xphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    db22=[[dot(d1yphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    db21=[[dot(d1yphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    dc22=[[dot(d1zphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    dc21=[[dot(d1zphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dd22=[[dot(d1zphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dd21=[[dot(d1zphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];

    d1c[1]=sum([dot(da11[i],b11[i])+dot(da12[i],b12[i])+dot(da22[i],b22[i])+dot(da21[i],b21[i]) for i in 1:kernel.K])-d1ob(1,x);
    d1c[2]=sum([dot(a11[i],db11[i])+dot(a12[i],db12[i])+dot(a22[i],db22[i])+dot(a21[i],db21[i]) for i in 1:kernel.K])-d1ob(2,x);
    d1c[3]=sum([(dot(dc11[i],b11[i])+dot(a11[i],dd11[i]))+(dot(dc22[i],b22[i])+dot(a22[i],dd22[i]))+(dot(dc12[i],b12[i])+dot(a12[i],dd12[i]))+(dot(dc21[i],b21[i])+dot(a21[i],dd21[i])) for i in 1:kernel.K])-d1ob(3,x);
    return d1c
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Array{Float64,1},1},1},1},1})
    d2c=zeros(kernel.dim,kernel.dim);
    a11=[[dot(phix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    b11=[[dot(phiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    a12=[[dot(phix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    b12=[[dot(phiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    a21=[[dot(phix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    b21=[[dot(phiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    a22=[[dot(phix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    b22=[[dot(phiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];

    da11=[[dot(d1xphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    da12=[[dot(d1xphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    db11=[[dot(d1yphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    db12=[[dot(d1yphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dc11=[[dot(d1zphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dc12=[[dot(d1zphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    dd11=[[dot(d1zphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    dd12=[[dot(d1zphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    da22=[[dot(d1xphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    da21=[[dot(d1xphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    db22=[[dot(d1yphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    db21=[[dot(d1yphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    dc22=[[dot(d1zphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    dc21=[[dot(d1zphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dd22=[[dot(d1zphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dd21=[[dot(d1zphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];


    dac11=[[dot(d11xzphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dac12=[[dot(d11xzphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    dbd11=[[dot(d11yzphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    dbd12=[[dot(d11yzphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dda11=[[dot(d2xphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dda12=[[dot(d2xphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    ddb11=[[dot(d2yphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    ddb12=[[dot(d2yphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    ddc11=[[dot(d2zphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    ddc12=[[dot(d2zphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    ddd11=[[dot(d2zphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    ddd12=[[dot(d2zphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];

    dac22=[[dot(d11xzphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    dbd22=[[dot(d11yzphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dda22=[[dot(d2xphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    ddb22=[[dot(d2yphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    ddc22=[[dot(d2zphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    ddd22=[[dot(d2zphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dac21=[[dot(d11xzphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dbd21=[[dot(d11yzphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    dda21=[[dot(d2xphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    ddb21=[[dot(d2yphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    ddc21=[[dot(d2zphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    ddd21=[[dot(d2zphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];

    d2c[1,2]=sum([dot(da11[i],db11[i])+dot(da22[i],db22[i])+dot(da12[i],db12[i])+dot(da21[i],db21[i]) for i in 1:kernel.K])-dot(d11phiVect(1,2,x),y);
    d2c[1,3]=sum([(dot(da11[i],dd11[i])+dot(dac11[i],b11[i]))+(dot(da22[i],dd22[i])+dot(dac22[i],b22[i]))+(dot(da12[i],dd12[i])+dot(dac12[i],b12[i]))+(dot(da21[i],dd21[i])+dot(dac21[i],b21[i])) for i in 1:kernel.K])-dot(d11phiVect(1,3,x),y);
    d2c[2,3]=sum([(dot(dc11[i],db11[i])+dot(a11[i],dbd11[i]))+(dot(dc22[i],db22[i])+dot(a22[i],dbd22[i]))+(dot(dc12[i],db12[i])+dot(a12[i],dbd12[i]))+(dot(dc21[i],db21[i])+dot(a21[i],dbd21[i])) for i in 1:kernel.K])-dot(d11phiVect(2,3,x),y);
    d2c=d2c+d2c';
    d2c[1,1]=sum([dot(dda11[i],b11[i])+dot(dda22[i],b22[i])+dot(dda12[i],b12[i])+dot(dda21[i],b21[i]) for i in 1:kernel.K])-d2ob(1,x);
    d2c[2,2]=sum([dot(a11[i],ddb11[i])+dot(a22[i],ddb22[i])+dot(a12[i],ddb12[i])+dot(a21[i],ddb21[i]) for i in 1:kernel.K])-d2ob(2,x);
    d2c[3,3]=sum([(dot(ddc11[i],b11[i])+dot(a11[i],ddd11[i])+2*dot(dc11[i],dd11[i]))+(dot(ddc22[i],b22[i])+dot(a22[i],ddd22[i])+2*dot(dc22[i],dd22[i]))+(dot(ddc12[i],b12[i])+dot(a12[i],ddd12[i])+2*dot(dc12[i],dd12[i]))+(dot(ddc21[i],b21[i])+dot(a21[i],ddd21[i])+2*dot(dc21[i],dd21[i])) for i in 1:kernel.K])-d2ob(3,x);
    return(d2c)
  end

  gaussconv2DDoubleHelixMultiFP(typeof(kernel),kernel.dim,kernel.bounds,kernel.K,kernel.fp,normObs,Phix,Phiy,PhisY,phix,phiy,phiVect,d1phiVect,d11phiVect,d2phiVect,y,c,d10c,d01c,d11c,d20c,d02c,ob,d1ob,d11ob,d2ob,correl,d1correl,d2correl);
end

function setoperator(kernel::doubleHelixMultiFP,a0::Array{Float64,1},x0::Array{Array{Float64,1},1},w::Array{Float64,1})
  function phiDx(x::Float64,z::Float64,xi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2);
    return .5*(erf(alphap/kernel.sigmax)-erf(alpham/kernel.sigmax));
  end
  function phiDy(y::Float64,z::Float64,yi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2);
    return .5*(erf(alphap/kernel.sigmay)-erf(alpham/kernel.sigmay));
  end

  function d1xphiDx(x::Float64,z::Float64,xi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmax)*(exp(-(alphap/kernel.sigmax)^2) - exp(-(alpham/kernel.sigmax)^2));
  end
  function d1yphiDy(y::Float64,z::Float64,yi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmay)*(exp(-(alphap/kernel.sigmay)^2) - exp(-(alpham/kernel.sigmay)^2));
  end

  function d11xzphiDx(x::Float64,z::Float64,xi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2);
    return -s*kernel.d1rotx(z,fp)/(sqrt(pi)*kernel.sigmax^3)*(alphap*exp(-(alphap/kernel.sigmax)^2) - alpham*exp(-(alpham/kernel.sigmax)^2));
  end
  function d11yzphiDy(y::Float64,z::Float64,yi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2);
    return -s*kernel.d1roty(z,fp)/(sqrt(pi)*kernel.sigmay^3)*(alphap*exp(-(alphap/kernel.sigmay)^2) - alpham*exp(-(alpham/kernel.sigmay)^2));
  end

  function d2xphiDx(x::Float64,z::Float64,xi::Float64,s::Float64,fp::Float64)
    return -1/(sqrt(pi)*kernel.sigmax^2)*( ((xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/(sqrt(2)*kernel.sigmax))*exp(-((xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/(sqrt(2)*kernel.sigmax))^2) - ((xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/(sqrt(2)*kernel.sigmax))*exp(-((xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/(sqrt(2)*kernel.sigmax))^2) );
  end
  function d2yphiDy(y::Float64,z::Float64,yi::Float64,s::Float64,fp::Float64)
    return -1/(sqrt(pi)*kernel.sigmay^2)*( ((yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/(sqrt(2)*kernel.sigmay))*exp(-((yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/(sqrt(2)*kernel.sigmay))^2) - ((yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/(sqrt(2)*kernel.sigmay))*exp(-((yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/(sqrt(2)*kernel.sigmay))^2) );
  end

  function d1zphiDx(x::Float64,z::Float64,xi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2);
    return -s*kernel.d1rotx(z,fp)/(sqrt(2*pi)*kernel.sigmax)*(exp(-(alphap/kernel.sigmax)^2) - exp(-(alpham/kernel.sigmax)^2));
  end
  function d1zphiDy(y::Float64,z::Float64,yi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2);
    return -s*kernel.d1roty(z,fp)/(sqrt(2*pi)*kernel.sigmay)*(exp(-(alphap/kernel.sigmay)^2) - exp(-(alpham/kernel.sigmay)^2));
  end

  function d2zphiDx(x::Float64,z::Float64,xi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2);
    return -s*kernel.d2rotx(z,fp)/(sqrt(2*pi)*kernel.sigmax)*(exp(-(alphap/kernel.sigmax)^2) - exp(-(alpham/kernel.sigmax)^2)) - s^2*kernel.d1rotx(z,fp)^2/(sqrt(pi)*kernel.sigmax^3)*(alphap*exp(-(alphap/kernel.sigmax)^2) - alpham*exp(-(alpham/kernel.sigmax)^2));
  end
  function d2zphiDy(y::Float64,z::Float64,yi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2);
    return -s*kernel.d2roty(z,fp)/(sqrt(2*pi)*kernel.sigmay)*(exp(-(alphap/kernel.sigmay)^2) - exp(-(alpham/kernel.sigmay)^2)) - s^2*kernel.d1roty(z,fp)^2/(sqrt(pi)*kernel.sigmay^3)*(alphap*exp(-(alphap/kernel.sigmay)^2) - alpham*exp(-(alpham/kernel.sigmay)^2));
  end
  ##

  ##
  function phix(x::Float64,z::Float64,s::Float64,fp::Float64)
    phipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      phipx[i]=phiDx(x,z,kernel.px[i],s,fp);
    end
    return phipx;
  end
  #
  function phiy(y::Float64,z::Float64,s::Float64,fp::Float64)
    phipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      phipy[i]=phiDy(y,z,kernel.py[i],s,fp);
    end
    return phipy;
  end
  #

  Phix=[Array{Array{Array{Array{Float64,1},1},1}}(kernel.K),Array{Array{Array{Array{Float64,1},1},1}}(kernel.K)];
  Phiy=[Array{Array{Array{Array{Float64,1},1},1}}(kernel.K),Array{Array{Array{Array{Float64,1},1},1}}(kernel.K)];
  for l in 1:kernel.K
    Phix[1][l]=Array{Array{Array{Float64,1},1},1}(kernel.nbpointsgrid[3]);
    Phiy[1][l]=Array{Array{Array{Float64,1},1},1}(kernel.nbpointsgrid[3]);
    Phix[2][l]=Array{Array{Array{Float64,1},1},1}(kernel.nbpointsgrid[3]);
    Phiy[2][l]=Array{Array{Array{Float64,1},1},1}(kernel.nbpointsgrid[3]);
    for k in 1:kernel.nbpointsgrid[3]
      Phix[1][l][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[1]);
      Phiy[1][l][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[2]);
      Phix[2][l][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[1]);
      Phiy[2][l][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[2]);
      for i in 1:kernel.nbpointsgrid[1]
        Phix[1][l][k][i]=phix(kernel.grid[1][i],kernel.grid[3][k],1.0,kernel.fp[l]);
        Phix[2][l][k][i]=phix(kernel.grid[1][i],kernel.grid[3][k],-1.0,kernel.fp[l]);
      end
      for j in 1:kernel.nbpointsgrid[2]
        Phiy[1][l][k][j]=phiy(kernel.grid[2][j],kernel.grid[3][k],1.0,kernel.fp[l]);
        Phiy[2][l][k][j]=phiy(kernel.grid[2][j],kernel.grid[3][k],-1.0,kernel.fp[l]);
      end
    end
  end

  #
  function d1xphix(x::Float64,z::Float64,s::Float64,fp::Float64)
    d1xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1xphipx[i]=d1xphiDx(x,z,kernel.px[i],s,fp);
    end
    return d1xphipx;
  end
  #
  function d1yphiy(y::Float64,z::Float64,s::Float64,fp::Float64)
    d1yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1yphipy[i]=d1yphiDy(y,z,kernel.py[i],s,fp);
    end
    return d1yphipy;
  end
  #
  function d1zphix(x::Float64,z::Float64,s::Float64,fp::Float64)
    d1zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1zphipx[i]=d1zphiDx(x,z,kernel.px[i],s,fp);
    end
    return d1zphipx;
  end
  #
  function d1zphiy(y::Float64,z::Float64,s::Float64,fp::Float64)
    d1zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1zphipy[i]=d1zphiDy(y,z,kernel.py[i],s,fp);
    end
    return d1zphipy;
  end
  #

  #
  function d11xzphix(x::Float64,z::Float64,s::Float64,fp::Float64)
    d11xzphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d11xzphipx[i]=d11xzphiDx(x,z,kernel.px[i],s,fp);
    end
    return d11xzphipx
  end
  #
  function d11yzphiy(y::Float64,z::Float64,s::Float64,fp::Float64)
    d11yzphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d11yzphipy[i]=d11yzphiDy(y,z,kernel.py[i],s,fp);
    end
    return d11yzphipy
  end
  #

  #
  function d2xphix(x::Float64,z::Float64,s::Float64,fp::Float64)
    d2xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2xphipx[i]=d2xphiDx(x,z,kernel.px[i],s,fp);
    end
    return d2xphipx;
  end
  function d2yphiy(y::Float64,z::Float64,s::Float64,fp::Float64)
    d2yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2yphipy[i]=d2yphiDy(y,z,kernel.py[i],s,fp);
    end
    return d2yphipy;
  end
  #
  function d2zphix(x::Float64,z::Float64,s::Float64,fp::Float64)
    d2zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2zphipx[i]=d2zphiDx(x,z,kernel.px[i],s,fp);
    end
    return d2zphipx;
  end
  function d2zphiy(y::Float64,z::Float64,s::Float64,fp::Float64)
    d2zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2zphipy[i]=d2zphiDy(y,z,kernel.py[i],s,fp);
    end
    return d2zphipy;
  end
  #
  ##

  ##
  function phiVect(x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy*kernel.K);
    l=1;
    for k in 1:kernel.K
      phipx1=phix(x[1],x[3],1.0,kernel.fp[k]);
      phipy1=phiy(x[2],x[3],1.0,kernel.fp[k]);
      phipx2=phix(x[1],x[3],-1.0,kernel.fp[k]);
      phipy2=phiy(x[2],x[3],-1.0,kernel.fp[k]);
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=phipx1[i]*phipy1[j]+phipx2[i]*phipy2[j];
          l+=1;
        end
      end
    end
    return v;
  end
  function d1phiVect(m::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy*kernel.K);
    l=1;
    if m==1
      for k in 1:kernel.K
        #
        phipy1=phiy(x[2],x[3],1.0,kernel.fp[k]);
        phipy2=phiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d1xphipx1=d1xphix(x[1],x[3],1.0,kernel.fp[k]);
        d1xphipx2=d1xphix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d1xphipx1[i]*phipy1[j]+d1xphipx2[i]*phipy2[j];
            l+=1;
          end
        end
      end
      return v;
    elseif m==2
      for k in 1:kernel.K
        #
        phipx1=phix(x[1],x[3],1.0,kernel.fp[k]);
        phipx2=phix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        d1yphipy1=d1yphiy(x[2],x[3],1.0,kernel.fp[k]);
        d1yphipy2=d1yphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=phipx1[i]*d1yphipy1[j]+phipx2[i]*d1yphipy2[j];
            l+=1;
          end
        end
      end
      return v;
    else
      for k in 1:kernel.K
        #
        phipx1=phix(x[1],x[3],1.0,kernel.fp[k]);
        phipy1=phiy(x[2],x[3],1.0,kernel.fp[k]);
        phipx2=phix(x[1],x[3],-1.0,kernel.fp[k]);
        phipy2=phiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d1zphipx1=d1zphix(x[1],x[3],1.0,kernel.fp[k]);
        d1zphipy1=d1zphiy(x[2],x[3],1.0,kernel.fp[k]);
        d1zphipx2=d1zphix(x[1],x[3],-1.0,kernel.fp[k]);
        d1zphipy2=d1zphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=(d1zphipx1[i]*phipy1[j]+phipx1[i]*d1zphipy1[j])+(d1zphipx2[i]*phipy2[j]+phipx2[i]*d1zphipy2[j]);
            l+=1;
          end
        end
      end
      return v;
    end
  end
  function d11phiVect(m::Int64,n::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy*kernel.K);
    l=1;
    if m==1 && n==2 || m==2 && n==1
      for k in 1:kernel.K
        #
        d1xphipx1=d1xphix(x[1],x[3],1.0,kernel.fp[k]);
        d1yphipy1=d1yphiy(x[2],x[3],1.0,kernel.fp[k]);
        d1xphipx2=d1xphix(x[1],x[3],-1.0,kernel.fp[k]);
        d1yphipy2=d1yphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d1xphipx1[i]*d1yphipy1[j]+d1xphipx2[i]*d1yphipy2[j];
            l+=1;
          end
        end
      end
      return v;
    elseif m==1 && n==3 || m==3 && n==1
      for k in 1:kernel.K
        #
        phipy1=phiy(x[2],x[3],1.0,kernel.fp[k]);
        phipy2=phiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d1xphipx1=d1xphix(x[1],x[3],1.0,kernel.fp[k]);
        d1zphipy1=d1zphiy(x[2],x[3],1.0,kernel.fp[k]);
        d1xphipx2=d1xphix(x[1],x[3],-1.0,kernel.fp[k]);
        d1zphipy2=d1zphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d11xzphipx1=d11xzphix(x[1],x[3],1.0,kernel.fp[k]);
        d11xzphipx2=d11xzphix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=(d11xzphipx1[i]*phipy1[j]+d1xphipx1[i]*d1zphipy1[j])+(d11xzphipx2[i]*phipy2[j]+d1xphipx2[i]*d1zphipy2[j]);
            l+=1;
          end
        end
      end
      return v;
    elseif m==2 && n==3 || m==3 && n==2
      for k in 1:kernel.K
        #
        phipx1=phix(x[1],x[3],1.0,kernel.fp[k]);
        phipx2=phix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        d1yphipy1=d1yphiy(x[2],x[3],1.0,kernel.fp[k]);
        d1zphipx1=d1zphix(x[1],x[3],1.0,kernel.fp[k]);
        d1yphipy2=d1yphiy(x[2],x[3],-1.0,kernel.fp[k]);
        d1zphipx2=d1zphix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        d11yzphipy1=d11yzphiy(x[2],x[3],1.0,kernel.fp[k]);
        d11yzphipy2=d11yzphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=(phipx1[i]*d11yzphipy1[j]+d1zphipx1[i]*d1yphipy1[j])+(phipx2[i]*d11yzphipy2[j]+d1zphipx2[i]*d1yphipy2[j]);
            l+=1;
          end
        end
      end
      return v;
    end
  end
  function d2phiVect(m::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy*kernel.K);
    l=1;
    if m==1
      for k in 1:kernel.K
        #
        phipy1=phiy(x[2],x[3],1.0,kernel.fp[k]);
        phipy2=phiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d2xphipx1=d2xphix(x[1],x[3],1.0,kernel.fp[k]);
        d2xphipx2=d2xphix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=(d2xphipx1[i]*phipy1[j])+(d2xphipx2[i]*phipy2[j]);
            l+=1;
          end
        end
      end
      return v;
    elseif m==2
      for k in 1:kernel.K
        #
        phipx1=phix(x[1],x[3],1.0,kernel.fp[k]);
        phipx2=phix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        d2yphipy1=d2yphiy(x[2],x[3],1.0,kernel.fp[k]);
        d2yphipy2=d2yphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=(phipx1[i]*d2yphipy1[j])+(phipx2[i]*d2yphipy2[j]);
            l+=1;
          end
        end
      end
      return v;
    else
      for k in 1:kernel.K
        #
        phipx1=phix(x[1],x[3],1.0,kernel.fp[k]);
        phipy1=phiy(x[2],x[3],1.0,kernel.fp[k]);
        phipx2=phix(x[1],x[3],-1.0,kernel.fp[k]);
        phipy2=phiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d1zphipx1=d1zphix(x[1],x[3],1.0,kernel.fp[k]);
        d1zphipy1=d1zphiy(x[2],x[3],1.0,kernel.fp[k]);
        d1zphipx2=d1zphix(x[1],x[3],-1.0,kernel.fp[k]);
        d1zphipy2=d1zphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d2zphipx1=d2zphix(x[1],x[3],1.0,kernel.fp[k]);
        d2zphipy1=d2zphiy(x[2],x[3],1.0,kernel.fp[k]);
        d2zphipx2=d2zphix(x[1],x[3],-1.0,kernel.fp[k]);
        d2zphipy2=d2zphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=(d2zphipx1[i]*phipy1[j]+phipx1[i]*d2zphipy1[j]+2*d1zphipx1[i]*d1zphipy1[j])+(d2zphipx2[i]*phipy2[j]+phipx2[i]*d2zphipy2[j]+2*d1zphipx2[i]*d1zphipy2[j]);
            l+=1;
          end
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
        v=zeros(kernel.Npx*kernel.Npy*kernel.K);
        lp=1;
        for kp in 1:kernel.K
          for jp in 1:kernel.Npy
            for ip in 1:kernel.Npx
              v[lp]=Phix[1][kp][k][i][ip]*Phiy[1][kp][k][j][jp]+Phix[2][kp][k][i][ip]*Phiy[2][kp][k][j][jp];
              lp+=1;
            end
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


  function correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Array{Float64,1},1},1},1},1})
    a11=[[dot(phix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    b11=[[dot(phiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    a12=[[dot(phix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    b12=[[dot(phiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    a21=[[dot(phix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    b21=[[dot(phiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    a22=[[dot(phix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    b22=[[dot(phiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    return sum([dot(a11[i],b11[i])+dot(a12[i],b12[i])+dot(a21[i],b21[i])+dot(a22[i],b22[i]) for i in 1:kernel.K])-ob(x);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Array{Float64,1},1},1},1},1})
    d1c=zeros(kernel.dim);
    a11=[[dot(phix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    b11=[[dot(phiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    a12=[[dot(phix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    b12=[[dot(phiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    a21=[[dot(phix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    b21=[[dot(phiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    a22=[[dot(phix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    b22=[[dot(phiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];

    da11=[[dot(d1xphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    da12=[[dot(d1xphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    db11=[[dot(d1yphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    db12=[[dot(d1yphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dc11=[[dot(d1zphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dc12=[[dot(d1zphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    dd11=[[dot(d1zphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    dd12=[[dot(d1zphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    da22=[[dot(d1xphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    da21=[[dot(d1xphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    db22=[[dot(d1yphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    db21=[[dot(d1yphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    dc22=[[dot(d1zphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    dc21=[[dot(d1zphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dd22=[[dot(d1zphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dd21=[[dot(d1zphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];

    d1c[1]=sum([dot(da11[i],b11[i])+dot(da12[i],b12[i])+dot(da22[i],b22[i])+dot(da21[i],b21[i]) for i in 1:kernel.K])-d1ob(1,x);
    d1c[2]=sum([dot(a11[i],db11[i])+dot(a12[i],db12[i])+dot(a22[i],db22[i])+dot(a21[i],db21[i]) for i in 1:kernel.K])-d1ob(2,x);
    d1c[3]=sum([(dot(dc11[i],b11[i])+dot(a11[i],dd11[i]))+(dot(dc22[i],b22[i])+dot(a22[i],dd22[i]))+(dot(dc12[i],b12[i])+dot(a12[i],dd12[i]))+(dot(dc21[i],b21[i])+dot(a21[i],dd21[i])) for i in 1:kernel.K])-d1ob(3,x);
    return d1c
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Array{Float64,1},1},1},1},1})
    d2c=zeros(kernel.dim,kernel.dim);
    a11=[[dot(phix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    b11=[[dot(phiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    a12=[[dot(phix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    b12=[[dot(phiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    a21=[[dot(phix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    b21=[[dot(phiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    a22=[[dot(phix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    b22=[[dot(phiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];

    da11=[[dot(d1xphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    da12=[[dot(d1xphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    db11=[[dot(d1yphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    db12=[[dot(d1yphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dc11=[[dot(d1zphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dc12=[[dot(d1zphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    dd11=[[dot(d1zphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    dd12=[[dot(d1zphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    da22=[[dot(d1xphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    da21=[[dot(d1xphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    db22=[[dot(d1yphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    db21=[[dot(d1yphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    dc22=[[dot(d1zphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    dc21=[[dot(d1zphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dd22=[[dot(d1zphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dd21=[[dot(d1zphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];


    dac11=[[dot(d11xzphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dac12=[[dot(d11xzphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    dbd11=[[dot(d11yzphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    dbd12=[[dot(d11yzphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dda11=[[dot(d2xphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dda12=[[dot(d2xphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    ddb11=[[dot(d2yphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    ddb12=[[dot(d2yphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    ddc11=[[dot(d2zphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    ddc12=[[dot(d2zphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    ddd11=[[dot(d2zphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    ddd12=[[dot(d2zphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];

    dac22=[[dot(d11xzphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    dbd22=[[dot(d11yzphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dda22=[[dot(d2xphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    ddb22=[[dot(d2yphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    ddc22=[[dot(d2zphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    ddd22=[[dot(d2zphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dac21=[[dot(d11xzphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dbd21=[[dot(d11yzphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    dda21=[[dot(d2xphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    ddb21=[[dot(d2yphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    ddc21=[[dot(d2zphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    ddd21=[[dot(d2zphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];

    d2c[1,2]=sum([dot(da11[i],db11[i])+dot(da22[i],db22[i])+dot(da12[i],db12[i])+dot(da21[i],db21[i]) for i in 1:kernel.K])-dot(d11phiVect(1,2,x),y);
    d2c[1,3]=sum([(dot(da11[i],dd11[i])+dot(dac11[i],b11[i]))+(dot(da22[i],dd22[i])+dot(dac22[i],b22[i]))+(dot(da12[i],dd12[i])+dot(dac12[i],b12[i]))+(dot(da21[i],dd21[i])+dot(dac21[i],b21[i])) for i in 1:kernel.K])-dot(d11phiVect(1,3,x),y);
    d2c[2,3]=sum([(dot(dc11[i],db11[i])+dot(a11[i],dbd11[i]))+(dot(dc22[i],db22[i])+dot(a22[i],dbd22[i]))+(dot(dc12[i],db12[i])+dot(a12[i],dbd12[i]))+(dot(dc21[i],db21[i])+dot(a21[i],dbd21[i])) for i in 1:kernel.K])-dot(d11phiVect(2,3,x),y);
    d2c=d2c+d2c';
    d2c[1,1]=sum([dot(dda11[i],b11[i])+dot(dda22[i],b22[i])+dot(dda12[i],b12[i])+dot(dda21[i],b21[i]) for i in 1:kernel.K])-d2ob(1,x);
    d2c[2,2]=sum([dot(a11[i],ddb11[i])+dot(a22[i],ddb22[i])+dot(a12[i],ddb12[i])+dot(a21[i],ddb21[i]) for i in 1:kernel.K])-d2ob(2,x);
    d2c[3,3]=sum([(dot(ddc11[i],b11[i])+dot(a11[i],ddd11[i])+2*dot(dc11[i],dd11[i]))+(dot(ddc22[i],b22[i])+dot(a22[i],ddd22[i])+2*dot(dc22[i],dd22[i]))+(dot(ddc12[i],b12[i])+dot(a12[i],ddd12[i])+2*dot(dc12[i],dd12[i]))+(dot(ddc21[i],b21[i])+dot(a21[i],ddd21[i])+2*dot(dc21[i],dd21[i])) for i in 1:kernel.K])-d2ob(3,x);
    return(d2c)
  end

  gaussconv2DDoubleHelixMultiFP(typeof(kernel),kernel.dim,kernel.bounds,kernel.K,kernel.fp,normObs,Phix,Phiy,PhisY,phix,phiy,phiVect,d1phiVect,d11phiVect,d2phiVect,y,c,d10c,d01c,d11c,d20c,d02c,ob,d1ob,d11ob,d2ob,correl,d1correl,d2correl);
end

function setoperator(kernel::doubleHelixMultiFP,a0::Array{Float64,1},x0::Array{Array{Float64,1},1},w::Array{Float64,1},nPhoton::Float64)
  function phiDx(x::Float64,z::Float64,xi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2);
    return .5*(erf(alphap/kernel.sigmax)-erf(alpham/kernel.sigmax));
  end
  function phiDy(y::Float64,z::Float64,yi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2);
    return .5*(erf(alphap/kernel.sigmay)-erf(alpham/kernel.sigmay));
  end

  function d1xphiDx(x::Float64,z::Float64,xi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmax)*(exp(-(alphap/kernel.sigmax)^2) - exp(-(alpham/kernel.sigmax)^2));
  end
  function d1yphiDy(y::Float64,z::Float64,yi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmay)*(exp(-(alphap/kernel.sigmay)^2) - exp(-(alpham/kernel.sigmay)^2));
  end

  function d11xzphiDx(x::Float64,z::Float64,xi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2);
    return -s*kernel.d1rotx(z,fp)/(sqrt(pi)*kernel.sigmax^3)*(alphap*exp(-(alphap/kernel.sigmax)^2) - alpham*exp(-(alpham/kernel.sigmax)^2));
  end
  function d11yzphiDy(y::Float64,z::Float64,yi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2);
    return -s*kernel.d1roty(z,fp)/(sqrt(pi)*kernel.sigmay^3)*(alphap*exp(-(alphap/kernel.sigmay)^2) - alpham*exp(-(alpham/kernel.sigmay)^2));
  end

  function d2xphiDx(x::Float64,z::Float64,xi::Float64,s::Float64,fp::Float64)
    return -1/(sqrt(pi)*kernel.sigmax^2)*( ((xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/(sqrt(2)*kernel.sigmax))*exp(-((xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/(sqrt(2)*kernel.sigmax))^2) - ((xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/(sqrt(2)*kernel.sigmax))*exp(-((xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/(sqrt(2)*kernel.sigmax))^2) );
  end
  function d2yphiDy(y::Float64,z::Float64,yi::Float64,s::Float64,fp::Float64)
    return -1/(sqrt(pi)*kernel.sigmay^2)*( ((yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/(sqrt(2)*kernel.sigmay))*exp(-((yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/(sqrt(2)*kernel.sigmay))^2) - ((yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/(sqrt(2)*kernel.sigmay))*exp(-((yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/(sqrt(2)*kernel.sigmay))^2) );
  end

  function d1zphiDx(x::Float64,z::Float64,xi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2);
    return -s*kernel.d1rotx(z,fp)/(sqrt(2*pi)*kernel.sigmax)*(exp(-(alphap/kernel.sigmax)^2) - exp(-(alpham/kernel.sigmax)^2));
  end
  function d1zphiDy(y::Float64,z::Float64,yi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2);
    return -s*kernel.d1roty(z,fp)/(sqrt(2*pi)*kernel.sigmay)*(exp(-(alphap/kernel.sigmay)^2) - exp(-(alpham/kernel.sigmay)^2));
  end

  function d2zphiDx(x::Float64,z::Float64,xi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2);
    return -s*kernel.d2rotx(z,fp)/(sqrt(2*pi)*kernel.sigmax)*(exp(-(alphap/kernel.sigmax)^2) - exp(-(alpham/kernel.sigmax)^2)) - s^2*kernel.d1rotx(z,fp)^2/(sqrt(pi)*kernel.sigmax^3)*(alphap*exp(-(alphap/kernel.sigmax)^2) - alpham*exp(-(alpham/kernel.sigmax)^2));
  end
  function d2zphiDy(y::Float64,z::Float64,yi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2);
    return -s*kernel.d2roty(z,fp)/(sqrt(2*pi)*kernel.sigmay)*(exp(-(alphap/kernel.sigmay)^2) - exp(-(alpham/kernel.sigmay)^2)) - s^2*kernel.d1roty(z,fp)^2/(sqrt(pi)*kernel.sigmay^3)*(alphap*exp(-(alphap/kernel.sigmay)^2) - alpham*exp(-(alpham/kernel.sigmay)^2));
  end
  ##

  ##
  function phix(x::Float64,z::Float64,s::Float64,fp::Float64)
    phipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      phipx[i]=phiDx(x,z,kernel.px[i],s,fp);
    end
    return phipx;
  end
  #
  function phiy(y::Float64,z::Float64,s::Float64,fp::Float64)
    phipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      phipy[i]=phiDy(y,z,kernel.py[i],s,fp);
    end
    return phipy;
  end
  #

  Phix=[Array{Array{Array{Array{Float64,1},1},1}}(kernel.K),Array{Array{Array{Array{Float64,1},1},1}}(kernel.K)];
  Phiy=[Array{Array{Array{Array{Float64,1},1},1}}(kernel.K),Array{Array{Array{Array{Float64,1},1},1}}(kernel.K)];
  for l in 1:kernel.K
    Phix[1][l]=Array{Array{Array{Float64,1},1},1}(kernel.nbpointsgrid[3]);
    Phiy[1][l]=Array{Array{Array{Float64,1},1},1}(kernel.nbpointsgrid[3]);
    Phix[2][l]=Array{Array{Array{Float64,1},1},1}(kernel.nbpointsgrid[3]);
    Phiy[2][l]=Array{Array{Array{Float64,1},1},1}(kernel.nbpointsgrid[3]);
    for k in 1:kernel.nbpointsgrid[3]
      Phix[1][l][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[1]);
      Phiy[1][l][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[2]);
      Phix[2][l][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[1]);
      Phiy[2][l][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[2]);
      for i in 1:kernel.nbpointsgrid[1]
        Phix[1][l][k][i]=phix(kernel.grid[1][i],kernel.grid[3][k],1.0,kernel.fp[l]);
        Phix[2][l][k][i]=phix(kernel.grid[1][i],kernel.grid[3][k],-1.0,kernel.fp[l]);
      end
      for j in 1:kernel.nbpointsgrid[2]
        Phiy[1][l][k][j]=phiy(kernel.grid[2][j],kernel.grid[3][k],1.0,kernel.fp[l]);
        Phiy[2][l][k][j]=phiy(kernel.grid[2][j],kernel.grid[3][k],-1.0,kernel.fp[l]);
      end
    end
  end

  #
  function d1xphix(x::Float64,z::Float64,s::Float64,fp::Float64)
    d1xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1xphipx[i]=d1xphiDx(x,z,kernel.px[i],s,fp);
    end
    return d1xphipx;
  end
  #
  function d1yphiy(y::Float64,z::Float64,s::Float64,fp::Float64)
    d1yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1yphipy[i]=d1yphiDy(y,z,kernel.py[i],s,fp);
    end
    return d1yphipy;
  end
  #
  function d1zphix(x::Float64,z::Float64,s::Float64,fp::Float64)
    d1zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1zphipx[i]=d1zphiDx(x,z,kernel.px[i],s,fp);
    end
    return d1zphipx;
  end
  #
  function d1zphiy(y::Float64,z::Float64,s::Float64,fp::Float64)
    d1zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1zphipy[i]=d1zphiDy(y,z,kernel.py[i],s,fp);
    end
    return d1zphipy;
  end
  #

  #
  function d11xzphix(x::Float64,z::Float64,s::Float64,fp::Float64)
    d11xzphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d11xzphipx[i]=d11xzphiDx(x,z,kernel.px[i],s,fp);
    end
    return d11xzphipx
  end
  #
  function d11yzphiy(y::Float64,z::Float64,s::Float64,fp::Float64)
    d11yzphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d11yzphipy[i]=d11yzphiDy(y,z,kernel.py[i],s,fp);
    end
    return d11yzphipy
  end
  #

  #
  function d2xphix(x::Float64,z::Float64,s::Float64,fp::Float64)
    d2xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2xphipx[i]=d2xphiDx(x,z,kernel.px[i],s,fp);
    end
    return d2xphipx;
  end
  function d2yphiy(y::Float64,z::Float64,s::Float64,fp::Float64)
    d2yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2yphipy[i]=d2yphiDy(y,z,kernel.py[i],s,fp);
    end
    return d2yphipy;
  end
  #
  function d2zphix(x::Float64,z::Float64,s::Float64,fp::Float64)
    d2zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2zphipx[i]=d2zphiDx(x,z,kernel.px[i],s,fp);
    end
    return d2zphipx;
  end
  function d2zphiy(y::Float64,z::Float64,s::Float64,fp::Float64)
    d2zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2zphipy[i]=d2zphiDy(y,z,kernel.py[i],s,fp);
    end
    return d2zphipy;
  end
  #
  ##

  ##
  function phiVect(x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy*kernel.K);
    l=1;
    for k in 1:kernel.K
      phipx1=phix(x[1],x[3],1.0,kernel.fp[k]);
      phipy1=phiy(x[2],x[3],1.0,kernel.fp[k]);
      phipx2=phix(x[1],x[3],-1.0,kernel.fp[k]);
      phipy2=phiy(x[2],x[3],-1.0,kernel.fp[k]);
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=phipx1[i]*phipy1[j]+phipx2[i]*phipy2[j];
          l+=1;
        end
      end
    end
    return v;
  end
  function d1phiVect(m::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy*kernel.K);
    l=1;
    if m==1
      for k in 1:kernel.K
        #
        phipy1=phiy(x[2],x[3],1.0,kernel.fp[k]);
        phipy2=phiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d1xphipx1=d1xphix(x[1],x[3],1.0,kernel.fp[k]);
        d1xphipx2=d1xphix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d1xphipx1[i]*phipy1[j]+d1xphipx2[i]*phipy2[j];
            l+=1;
          end
        end
      end
      return v;
    elseif m==2
      for k in 1:kernel.K
        #
        phipx1=phix(x[1],x[3],1.0,kernel.fp[k]);
        phipx2=phix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        d1yphipy1=d1yphiy(x[2],x[3],1.0,kernel.fp[k]);
        d1yphipy2=d1yphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=phipx1[i]*d1yphipy1[j]+phipx2[i]*d1yphipy2[j];
            l+=1;
          end
        end
      end
      return v;
    else
      for k in 1:kernel.K
        #
        phipx1=phix(x[1],x[3],1.0,kernel.fp[k]);
        phipy1=phiy(x[2],x[3],1.0,kernel.fp[k]);
        phipx2=phix(x[1],x[3],-1.0,kernel.fp[k]);
        phipy2=phiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d1zphipx1=d1zphix(x[1],x[3],1.0,kernel.fp[k]);
        d1zphipy1=d1zphiy(x[2],x[3],1.0,kernel.fp[k]);
        d1zphipx2=d1zphix(x[1],x[3],-1.0,kernel.fp[k]);
        d1zphipy2=d1zphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=(d1zphipx1[i]*phipy1[j]+phipx1[i]*d1zphipy1[j])+(d1zphipx2[i]*phipy2[j]+phipx2[i]*d1zphipy2[j]);
            l+=1;
          end
        end
      end
      return v;
    end
  end
  function d11phiVect(m::Int64,n::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy*kernel.K);
    l=1;
    if m==1 && n==2 || m==2 && n==1
      for k in 1:kernel.K
        #
        d1xphipx1=d1xphix(x[1],x[3],1.0,kernel.fp[k]);
        d1yphipy1=d1yphiy(x[2],x[3],1.0,kernel.fp[k]);
        d1xphipx2=d1xphix(x[1],x[3],-1.0,kernel.fp[k]);
        d1yphipy2=d1yphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d1xphipx1[i]*d1yphipy1[j]+d1xphipx2[i]*d1yphipy2[j];
            l+=1;
          end
        end
      end
      return v;
    elseif m==1 && n==3 || m==3 && n==1
      for k in 1:kernel.K
        #
        phipy1=phiy(x[2],x[3],1.0,kernel.fp[k]);
        phipy2=phiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d1xphipx1=d1xphix(x[1],x[3],1.0,kernel.fp[k]);
        d1zphipy1=d1zphiy(x[2],x[3],1.0,kernel.fp[k]);
        d1xphipx2=d1xphix(x[1],x[3],-1.0,kernel.fp[k]);
        d1zphipy2=d1zphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d11xzphipx1=d11xzphix(x[1],x[3],1.0,kernel.fp[k]);
        d11xzphipx2=d11xzphix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=(d11xzphipx1[i]*phipy1[j]+d1xphipx1[i]*d1zphipy1[j])+(d11xzphipx2[i]*phipy2[j]+d1xphipx2[i]*d1zphipy2[j]);
            l+=1;
          end
        end
      end
      return v;
    elseif m==2 && n==3 || m==3 && n==2
      for k in 1:kernel.K
        #
        phipx1=phix(x[1],x[3],1.0,kernel.fp[k]);
        phipx2=phix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        d1yphipy1=d1yphiy(x[2],x[3],1.0,kernel.fp[k]);
        d1zphipx1=d1zphix(x[1],x[3],1.0,kernel.fp[k]);
        d1yphipy2=d1yphiy(x[2],x[3],-1.0,kernel.fp[k]);
        d1zphipx2=d1zphix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        d11yzphipy1=d11yzphiy(x[2],x[3],1.0,kernel.fp[k]);
        d11yzphipy2=d11yzphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=(phipx1[i]*d11yzphipy1[j]+d1zphipx1[i]*d1yphipy1[j])+(phipx2[i]*d11yzphipy2[j]+d1zphipx2[i]*d1yphipy2[j]);
            l+=1;
          end
        end
      end
      return v;
    end
  end
  function d2phiVect(m::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy*kernel.K);
    l=1;
    if m==1
      for k in 1:kernel.K
        #
        phipy1=phiy(x[2],x[3],1.0,kernel.fp[k]);
        phipy2=phiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d2xphipx1=d2xphix(x[1],x[3],1.0,kernel.fp[k]);
        d2xphipx2=d2xphix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=(d2xphipx1[i]*phipy1[j])+(d2xphipx2[i]*phipy2[j]);
            l+=1;
          end
        end
      end
      return v;
    elseif m==2
      for k in 1:kernel.K
        #
        phipx1=phix(x[1],x[3],1.0,kernel.fp[k]);
        phipx2=phix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        d2yphipy1=d2yphiy(x[2],x[3],1.0,kernel.fp[k]);
        d2yphipy2=d2yphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=(phipx1[i]*d2yphipy1[j])+(phipx2[i]*d2yphipy2[j]);
            l+=1;
          end
        end
      end
      return v;
    else
      for k in 1:kernel.K
        #
        phipx1=phix(x[1],x[3],1.0,kernel.fp[k]);
        phipy1=phiy(x[2],x[3],1.0,kernel.fp[k]);
        phipx2=phix(x[1],x[3],-1.0,kernel.fp[k]);
        phipy2=phiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d1zphipx1=d1zphix(x[1],x[3],1.0,kernel.fp[k]);
        d1zphipy1=d1zphiy(x[2],x[3],1.0,kernel.fp[k]);
        d1zphipx2=d1zphix(x[1],x[3],-1.0,kernel.fp[k]);
        d1zphipy2=d1zphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d2zphipx1=d2zphix(x[1],x[3],1.0,kernel.fp[k]);
        d2zphipy1=d2zphiy(x[2],x[3],1.0,kernel.fp[k]);
        d2zphipx2=d2zphix(x[1],x[3],-1.0,kernel.fp[k]);
        d2zphipy2=d2zphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=(d2zphipx1[i]*phipy1[j]+phipx1[i]*d2zphipy1[j]+2*d1zphipx1[i]*d1zphipy1[j])+(d2zphipx2[i]*phipy2[j]+phipx2[i]*d2zphipy2[j]+2*d1zphipx2[i]*d1zphipy2[j]);
            l+=1;
          end
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
  y=copy(blasso.poisson.((y./maximum(sum([y[1+(k-1)*kernel.Npx*kernel.Npy : k*kernel.Npx*kernel.Npy] for k in 1:kernel.K])))*nPhoton)./nPhoton);
  y=y+w;
  normObs=.5*dot(y,y);

  PhisY=zeros(prod(kernel.nbpointsgrid));
  l=1;
  for k in 1:kernel.nbpointsgrid[3]
    for i in 1:kernel.nbpointsgrid[1]
      for j in 1:kernel.nbpointsgrid[2]
        v=zeros(kernel.Npx*kernel.Npy*kernel.K);
        lp=1;
        for kp in 1:kernel.K
          for jp in 1:kernel.Npy
            for ip in 1:kernel.Npx
              v[lp]=Phix[1][kp][k][i][ip]*Phiy[1][kp][k][j][jp]+Phix[2][kp][k][i][ip]*Phiy[2][kp][k][j][jp];
              lp+=1;
            end
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


  function correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Array{Float64,1},1},1},1},1})
    a11=[[dot(phix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    b11=[[dot(phiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    a12=[[dot(phix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    b12=[[dot(phiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    a21=[[dot(phix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    b21=[[dot(phiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    a22=[[dot(phix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    b22=[[dot(phiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    return sum([dot(a11[i],b11[i])+dot(a12[i],b12[i])+dot(a21[i],b21[i])+dot(a22[i],b22[i]) for i in 1:kernel.K])-ob(x);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Array{Float64,1},1},1},1},1})
    d1c=zeros(kernel.dim);
    a11=[[dot(phix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    b11=[[dot(phiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    a12=[[dot(phix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    b12=[[dot(phiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    a21=[[dot(phix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    b21=[[dot(phiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    a22=[[dot(phix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    b22=[[dot(phiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];

    da11=[[dot(d1xphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    da12=[[dot(d1xphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    db11=[[dot(d1yphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    db12=[[dot(d1yphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dc11=[[dot(d1zphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dc12=[[dot(d1zphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    dd11=[[dot(d1zphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    dd12=[[dot(d1zphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    da22=[[dot(d1xphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    da21=[[dot(d1xphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    db22=[[dot(d1yphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    db21=[[dot(d1yphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    dc22=[[dot(d1zphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    dc21=[[dot(d1zphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dd22=[[dot(d1zphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dd21=[[dot(d1zphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];

    d1c[1]=sum([dot(da11[i],b11[i])+dot(da12[i],b12[i])+dot(da22[i],b22[i])+dot(da21[i],b21[i]) for i in 1:kernel.K])-d1ob(1,x);
    d1c[2]=sum([dot(a11[i],db11[i])+dot(a12[i],db12[i])+dot(a22[i],db22[i])+dot(a21[i],db21[i]) for i in 1:kernel.K])-d1ob(2,x);
    d1c[3]=sum([(dot(dc11[i],b11[i])+dot(a11[i],dd11[i]))+(dot(dc22[i],b22[i])+dot(a22[i],dd22[i]))+(dot(dc12[i],b12[i])+dot(a12[i],dd12[i]))+(dot(dc21[i],b21[i])+dot(a21[i],dd21[i])) for i in 1:kernel.K])-d1ob(3,x);
    return d1c
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Array{Float64,1},1},1},1},1})
    d2c=zeros(kernel.dim,kernel.dim);
    a11=[[dot(phix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    b11=[[dot(phiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    a12=[[dot(phix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    b12=[[dot(phiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    a21=[[dot(phix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    b21=[[dot(phiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    a22=[[dot(phix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    b22=[[dot(phiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];

    da11=[[dot(d1xphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    da12=[[dot(d1xphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    db11=[[dot(d1yphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    db12=[[dot(d1yphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dc11=[[dot(d1zphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dc12=[[dot(d1zphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    dd11=[[dot(d1zphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    dd12=[[dot(d1zphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    da22=[[dot(d1xphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    da21=[[dot(d1xphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    db22=[[dot(d1yphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    db21=[[dot(d1yphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    dc22=[[dot(d1zphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    dc21=[[dot(d1zphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dd22=[[dot(d1zphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dd21=[[dot(d1zphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];


    dac11=[[dot(d11xzphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dac12=[[dot(d11xzphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    dbd11=[[dot(d11yzphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    dbd12=[[dot(d11yzphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dda11=[[dot(d2xphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dda12=[[dot(d2xphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    ddb11=[[dot(d2yphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    ddb12=[[dot(d2yphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    ddc11=[[dot(d2zphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    ddc12=[[dot(d2zphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    ddd11=[[dot(d2zphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    ddd12=[[dot(d2zphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];

    dac22=[[dot(d11xzphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    dbd22=[[dot(d11yzphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dda22=[[dot(d2xphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    ddb22=[[dot(d2yphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    ddc22=[[dot(d2zphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    ddd22=[[dot(d2zphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dac21=[[dot(d11xzphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dbd21=[[dot(d11yzphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    dda21=[[dot(d2xphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    ddb21=[[dot(d2yphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    ddc21=[[dot(d2zphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    ddd21=[[dot(d2zphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];

    d2c[1,2]=sum([dot(da11[i],db11[i])+dot(da22[i],db22[i])+dot(da12[i],db12[i])+dot(da21[i],db21[i]) for i in 1:kernel.K])-dot(d11phiVect(1,2,x),y);
    d2c[1,3]=sum([(dot(da11[i],dd11[i])+dot(dac11[i],b11[i]))+(dot(da22[i],dd22[i])+dot(dac22[i],b22[i]))+(dot(da12[i],dd12[i])+dot(dac12[i],b12[i]))+(dot(da21[i],dd21[i])+dot(dac21[i],b21[i])) for i in 1:kernel.K])-dot(d11phiVect(1,3,x),y);
    d2c[2,3]=sum([(dot(dc11[i],db11[i])+dot(a11[i],dbd11[i]))+(dot(dc22[i],db22[i])+dot(a22[i],dbd22[i]))+(dot(dc12[i],db12[i])+dot(a12[i],dbd12[i]))+(dot(dc21[i],db21[i])+dot(a21[i],dbd21[i])) for i in 1:kernel.K])-dot(d11phiVect(2,3,x),y);
    d2c=d2c+d2c';
    d2c[1,1]=sum([dot(dda11[i],b11[i])+dot(dda22[i],b22[i])+dot(dda12[i],b12[i])+dot(dda21[i],b21[i]) for i in 1:kernel.K])-d2ob(1,x);
    d2c[2,2]=sum([dot(a11[i],ddb11[i])+dot(a22[i],ddb22[i])+dot(a12[i],ddb12[i])+dot(a21[i],ddb21[i]) for i in 1:kernel.K])-d2ob(2,x);
    d2c[3,3]=sum([(dot(ddc11[i],b11[i])+dot(a11[i],ddd11[i])+2*dot(dc11[i],dd11[i]))+(dot(ddc22[i],b22[i])+dot(a22[i],ddd22[i])+2*dot(dc22[i],dd22[i]))+(dot(ddc12[i],b12[i])+dot(a12[i],ddd12[i])+2*dot(dc12[i],dd12[i]))+(dot(ddc21[i],b21[i])+dot(a21[i],ddd21[i])+2*dot(dc21[i],dd21[i])) for i in 1:kernel.K])-d2ob(3,x);
    return(d2c)
  end

  gaussconv2DDoubleHelixMultiFP(typeof(kernel),kernel.dim,kernel.bounds,kernel.K,kernel.fp,normObs,Phix,Phiy,PhisY,phix,phiy,phiVect,d1phiVect,d11phiVect,d2phiVect,y,c,d10c,d01c,d11c,d20c,d02c,ob,d1ob,d11ob,d2ob,correl,d1correl,d2correl);
end

function setoperator(kernel::doubleHelixMultiFP,y::Array{Float64,1})
  function phiDx(x::Float64,z::Float64,xi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2);
    return .5*(erf(alphap/kernel.sigmax)-erf(alpham/kernel.sigmax));
  end
  function phiDy(y::Float64,z::Float64,yi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2);
    return .5*(erf(alphap/kernel.sigmay)-erf(alpham/kernel.sigmay));
  end

  function d1xphiDx(x::Float64,z::Float64,xi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmax)*(exp(-(alphap/kernel.sigmax)^2) - exp(-(alpham/kernel.sigmax)^2));
  end
  function d1yphiDy(y::Float64,z::Float64,yi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmay)*(exp(-(alphap/kernel.sigmay)^2) - exp(-(alpham/kernel.sigmay)^2));
  end

  function d11xzphiDx(x::Float64,z::Float64,xi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2);
    return -s*kernel.d1rotx(z,fp)/(sqrt(pi)*kernel.sigmax^3)*(alphap*exp(-(alphap/kernel.sigmax)^2) - alpham*exp(-(alpham/kernel.sigmax)^2));
  end
  function d11yzphiDy(y::Float64,z::Float64,yi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2);
    return -s*kernel.d1roty(z,fp)/(sqrt(pi)*kernel.sigmay^3)*(alphap*exp(-(alphap/kernel.sigmay)^2) - alpham*exp(-(alpham/kernel.sigmay)^2));
  end

  function d2xphiDx(x::Float64,z::Float64,xi::Float64,s::Float64,fp::Float64)
    return -1/(sqrt(pi)*kernel.sigmax^2)*( ((xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/(sqrt(2)*kernel.sigmax))*exp(-((xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/(sqrt(2)*kernel.sigmax))^2) - ((xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/(sqrt(2)*kernel.sigmax))*exp(-((xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/(sqrt(2)*kernel.sigmax))^2) );
  end
  function d2yphiDy(y::Float64,z::Float64,yi::Float64,s::Float64,fp::Float64)
    return -1/(sqrt(pi)*kernel.sigmay^2)*( ((yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/(sqrt(2)*kernel.sigmay))*exp(-((yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/(sqrt(2)*kernel.sigmay))^2) - ((yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/(sqrt(2)*kernel.sigmay))*exp(-((yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/(sqrt(2)*kernel.sigmay))^2) );
  end

  function d1zphiDx(x::Float64,z::Float64,xi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2);
    return -s*kernel.d1rotx(z,fp)/(sqrt(2*pi)*kernel.sigmax)*(exp(-(alphap/kernel.sigmax)^2) - exp(-(alpham/kernel.sigmax)^2));
  end
  function d1zphiDy(y::Float64,z::Float64,yi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2);
    return -s*kernel.d1roty(z,fp)/(sqrt(2*pi)*kernel.sigmay)*(exp(-(alphap/kernel.sigmay)^2) - exp(-(alpham/kernel.sigmay)^2));
  end

  function d2zphiDx(x::Float64,z::Float64,xi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2),(xi-kernel.Dpx/2-x-s*kernel.rotx(z,fp))/sqrt(2);
    return -s*kernel.d2rotx(z,fp)/(sqrt(2*pi)*kernel.sigmax)*(exp(-(alphap/kernel.sigmax)^2) - exp(-(alpham/kernel.sigmax)^2)) - s^2*kernel.d1rotx(z,fp)^2/(sqrt(pi)*kernel.sigmax^3)*(alphap*exp(-(alphap/kernel.sigmax)^2) - alpham*exp(-(alpham/kernel.sigmax)^2));
  end
  function d2zphiDy(y::Float64,z::Float64,yi::Float64,s::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2),(yi-kernel.Dpy/2-y-s*kernel.roty(z,fp))/sqrt(2);
    return -s*kernel.d2roty(z,fp)/(sqrt(2*pi)*kernel.sigmay)*(exp(-(alphap/kernel.sigmay)^2) - exp(-(alpham/kernel.sigmay)^2)) - s^2*kernel.d1roty(z,fp)^2/(sqrt(pi)*kernel.sigmay^3)*(alphap*exp(-(alphap/kernel.sigmay)^2) - alpham*exp(-(alpham/kernel.sigmay)^2));
  end
  ##

  ##
  function phix(x::Float64,z::Float64,s::Float64,fp::Float64)
    phipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      phipx[i]=phiDx(x,z,kernel.px[i],s,fp);
    end
    return phipx;
  end
  #
  function phiy(y::Float64,z::Float64,s::Float64,fp::Float64)
    phipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      phipy[i]=phiDy(y,z,kernel.py[i],s,fp);
    end
    return phipy;
  end
  #

  Phix=[Array{Array{Array{Array{Float64,1},1},1}}(kernel.K),Array{Array{Array{Array{Float64,1},1},1}}(kernel.K)];
  Phiy=[Array{Array{Array{Array{Float64,1},1},1}}(kernel.K),Array{Array{Array{Array{Float64,1},1},1}}(kernel.K)];
  for l in 1:kernel.K
    Phix[1][l]=Array{Array{Array{Float64,1},1},1}(kernel.nbpointsgrid[3]);
    Phiy[1][l]=Array{Array{Array{Float64,1},1},1}(kernel.nbpointsgrid[3]);
    Phix[2][l]=Array{Array{Array{Float64,1},1},1}(kernel.nbpointsgrid[3]);
    Phiy[2][l]=Array{Array{Array{Float64,1},1},1}(kernel.nbpointsgrid[3]);
    for k in 1:kernel.nbpointsgrid[3]
      Phix[1][l][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[1]);
      Phiy[1][l][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[2]);
      Phix[2][l][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[1]);
      Phiy[2][l][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[2]);
      for i in 1:kernel.nbpointsgrid[1]
        Phix[1][l][k][i]=phix(kernel.grid[1][i],kernel.grid[3][k],1.0,kernel.fp[l]);
        Phix[2][l][k][i]=phix(kernel.grid[1][i],kernel.grid[3][k],-1.0,kernel.fp[l]);
      end
      for j in 1:kernel.nbpointsgrid[2]
        Phiy[1][l][k][j]=phiy(kernel.grid[2][j],kernel.grid[3][k],1.0,kernel.fp[l]);
        Phiy[2][l][k][j]=phiy(kernel.grid[2][j],kernel.grid[3][k],-1.0,kernel.fp[l]);
      end
    end
  end

  #
  function d1xphix(x::Float64,z::Float64,s::Float64,fp::Float64)
    d1xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1xphipx[i]=d1xphiDx(x,z,kernel.px[i],s,fp);
    end
    return d1xphipx;
  end
  #
  function d1yphiy(y::Float64,z::Float64,s::Float64,fp::Float64)
    d1yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1yphipy[i]=d1yphiDy(y,z,kernel.py[i],s,fp);
    end
    return d1yphipy;
  end
  #
  function d1zphix(x::Float64,z::Float64,s::Float64,fp::Float64)
    d1zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1zphipx[i]=d1zphiDx(x,z,kernel.px[i],s,fp);
    end
    return d1zphipx;
  end
  #
  function d1zphiy(y::Float64,z::Float64,s::Float64,fp::Float64)
    d1zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1zphipy[i]=d1zphiDy(y,z,kernel.py[i],s,fp);
    end
    return d1zphipy;
  end
  #

  #
  function d11xzphix(x::Float64,z::Float64,s::Float64,fp::Float64)
    d11xzphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d11xzphipx[i]=d11xzphiDx(x,z,kernel.px[i],s,fp);
    end
    return d11xzphipx
  end
  #
  function d11yzphiy(y::Float64,z::Float64,s::Float64,fp::Float64)
    d11yzphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d11yzphipy[i]=d11yzphiDy(y,z,kernel.py[i],s,fp);
    end
    return d11yzphipy
  end
  #

  #
  function d2xphix(x::Float64,z::Float64,s::Float64,fp::Float64)
    d2xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2xphipx[i]=d2xphiDx(x,z,kernel.px[i],s,fp);
    end
    return d2xphipx;
  end
  function d2yphiy(y::Float64,z::Float64,s::Float64,fp::Float64)
    d2yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2yphipy[i]=d2yphiDy(y,z,kernel.py[i],s,fp);
    end
    return d2yphipy;
  end
  #
  function d2zphix(x::Float64,z::Float64,s::Float64,fp::Float64)
    d2zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2zphipx[i]=d2zphiDx(x,z,kernel.px[i],s,fp);
    end
    return d2zphipx;
  end
  function d2zphiy(y::Float64,z::Float64,s::Float64,fp::Float64)
    d2zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2zphipy[i]=d2zphiDy(y,z,kernel.py[i],s,fp);
    end
    return d2zphipy;
  end
  #
  ##

  ##
  function phiVect(x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy*kernel.K);
    l=1;
    for k in 1:kernel.K
      phipx1=phix(x[1],x[3],1.0,kernel.fp[k]);
      phipy1=phiy(x[2],x[3],1.0,kernel.fp[k]);
      phipx2=phix(x[1],x[3],-1.0,kernel.fp[k]);
      phipy2=phiy(x[2],x[3],-1.0,kernel.fp[k]);
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=phipx1[i]*phipy1[j]+phipx2[i]*phipy2[j];
          l+=1;
        end
      end
    end
    return v;
  end
  function d1phiVect(m::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy*kernel.K);
    l=1;
    if m==1
      for k in 1:kernel.K
        #
        phipy1=phiy(x[2],x[3],1.0,kernel.fp[k]);
        phipy2=phiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d1xphipx1=d1xphix(x[1],x[3],1.0,kernel.fp[k]);
        d1xphipx2=d1xphix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d1xphipx1[i]*phipy1[j]+d1xphipx2[i]*phipy2[j];
            l+=1;
          end
        end
      end
      return v;
    elseif m==2
      for k in 1:kernel.K
        #
        phipx1=phix(x[1],x[3],1.0,kernel.fp[k]);
        phipx2=phix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        d1yphipy1=d1yphiy(x[2],x[3],1.0,kernel.fp[k]);
        d1yphipy2=d1yphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=phipx1[i]*d1yphipy1[j]+phipx2[i]*d1yphipy2[j];
            l+=1;
          end
        end
      end
      return v;
    else
      for k in 1:kernel.K
        #
        phipx1=phix(x[1],x[3],1.0,kernel.fp[k]);
        phipy1=phiy(x[2],x[3],1.0,kernel.fp[k]);
        phipx2=phix(x[1],x[3],-1.0,kernel.fp[k]);
        phipy2=phiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d1zphipx1=d1zphix(x[1],x[3],1.0,kernel.fp[k]);
        d1zphipy1=d1zphiy(x[2],x[3],1.0,kernel.fp[k]);
        d1zphipx2=d1zphix(x[1],x[3],-1.0,kernel.fp[k]);
        d1zphipy2=d1zphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=(d1zphipx1[i]*phipy1[j]+phipx1[i]*d1zphipy1[j])+(d1zphipx2[i]*phipy2[j]+phipx2[i]*d1zphipy2[j]);
            l+=1;
          end
        end
      end
      return v;
    end
  end
  function d11phiVect(m::Int64,n::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy*kernel.K);
    l=1;
    if m==1 && n==2 || m==2 && n==1
      for k in 1:kernel.K
        #
        d1xphipx1=d1xphix(x[1],x[3],1.0,kernel.fp[k]);
        d1yphipy1=d1yphiy(x[2],x[3],1.0,kernel.fp[k]);
        d1xphipx2=d1xphix(x[1],x[3],-1.0,kernel.fp[k]);
        d1yphipy2=d1yphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d1xphipx1[i]*d1yphipy1[j]+d1xphipx2[i]*d1yphipy2[j];
            l+=1;
          end
        end
      end
      return v;
    elseif m==1 && n==3 || m==3 && n==1
      for k in 1:kernel.K
        #
        phipy1=phiy(x[2],x[3],1.0,kernel.fp[k]);
        phipy2=phiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d1xphipx1=d1xphix(x[1],x[3],1.0,kernel.fp[k]);
        d1zphipy1=d1zphiy(x[2],x[3],1.0,kernel.fp[k]);
        d1xphipx2=d1xphix(x[1],x[3],-1.0,kernel.fp[k]);
        d1zphipy2=d1zphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d11xzphipx1=d11xzphix(x[1],x[3],1.0,kernel.fp[k]);
        d11xzphipx2=d11xzphix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=(d11xzphipx1[i]*phipy1[j]+d1xphipx1[i]*d1zphipy1[j])+(d11xzphipx2[i]*phipy2[j]+d1xphipx2[i]*d1zphipy2[j]);
            l+=1;
          end
        end
      end
      return v;
    elseif m==2 && n==3 || m==3 && n==2
      for k in 1:kernel.K
        #
        phipx1=phix(x[1],x[3],1.0,kernel.fp[k]);
        phipx2=phix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        d1yphipy1=d1yphiy(x[2],x[3],1.0,kernel.fp[k]);
        d1zphipx1=d1zphix(x[1],x[3],1.0,kernel.fp[k]);
        d1yphipy2=d1yphiy(x[2],x[3],-1.0,kernel.fp[k]);
        d1zphipx2=d1zphix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        d11yzphipy1=d11yzphiy(x[2],x[3],1.0,kernel.fp[k]);
        d11yzphipy2=d11yzphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=(phipx1[i]*d11yzphipy1[j]+d1zphipx1[i]*d1yphipy1[j])+(phipx2[i]*d11yzphipy2[j]+d1zphipx2[i]*d1yphipy2[j]);
            l+=1;
          end
        end
      end
      return v;
    end
  end
  function d2phiVect(m::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy*kernel.K);
    l=1;
    if m==1
      for k in 1:kernel.K
        #
        phipy1=phiy(x[2],x[3],1.0,kernel.fp[k]);
        phipy2=phiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d2xphipx1=d2xphix(x[1],x[3],1.0,kernel.fp[k]);
        d2xphipx2=d2xphix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=(d2xphipx1[i]*phipy1[j])+(d2xphipx2[i]*phipy2[j]);
            l+=1;
          end
        end
      end
      return v;
    elseif m==2
      for k in 1:kernel.K
        #
        phipx1=phix(x[1],x[3],1.0,kernel.fp[k]);
        phipx2=phix(x[1],x[3],-1.0,kernel.fp[k]);
        #
        d2yphipy1=d2yphiy(x[2],x[3],1.0,kernel.fp[k]);
        d2yphipy2=d2yphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=(phipx1[i]*d2yphipy1[j])+(phipx2[i]*d2yphipy2[j]);
            l+=1;
          end
        end
      end
      return v;
    else
      for k in 1:kernel.K
        #
        phipx1=phix(x[1],x[3],1.0,kernel.fp[k]);
        phipy1=phiy(x[2],x[3],1.0,kernel.fp[k]);
        phipx2=phix(x[1],x[3],-1.0,kernel.fp[k]);
        phipy2=phiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d1zphipx1=d1zphix(x[1],x[3],1.0,kernel.fp[k]);
        d1zphipy1=d1zphiy(x[2],x[3],1.0,kernel.fp[k]);
        d1zphipx2=d1zphix(x[1],x[3],-1.0,kernel.fp[k]);
        d1zphipy2=d1zphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        d2zphipx1=d2zphix(x[1],x[3],1.0,kernel.fp[k]);
        d2zphipy1=d2zphiy(x[2],x[3],1.0,kernel.fp[k]);
        d2zphipx2=d2zphix(x[1],x[3],-1.0,kernel.fp[k]);
        d2zphipy2=d2zphiy(x[2],x[3],-1.0,kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=(d2zphipx1[i]*phipy1[j]+phipx1[i]*d2zphipy1[j]+2*d1zphipx1[i]*d1zphipy1[j])+(d2zphipx2[i]*phipy2[j]+phipx2[i]*d2zphipy2[j]+2*d1zphipx2[i]*d1zphipy2[j]);
            l+=1;
          end
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
        v=zeros(kernel.Npx*kernel.Npy*kernel.K);
        lp=1;
        for kp in 1:kernel.K
          for jp in 1:kernel.Npy
            for ip in 1:kernel.Npx
              v[lp]=Phix[1][kp][k][i][ip]*Phiy[1][kp][k][j][jp]+Phix[2][kp][k][i][ip]*Phiy[2][kp][k][j][jp];
              lp+=1;
            end
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


  function correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Array{Float64,1},1},1},1},1})
    a11=[[dot(phix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    b11=[[dot(phiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    a12=[[dot(phix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    b12=[[dot(phiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    a21=[[dot(phix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    b21=[[dot(phiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    a22=[[dot(phix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    b22=[[dot(phiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    return sum([dot(a11[i],b11[i])+dot(a12[i],b12[i])+dot(a21[i],b21[i])+dot(a22[i],b22[i]) for i in 1:kernel.K])-ob(x);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Array{Float64,1},1},1},1},1})
    d1c=zeros(kernel.dim);
    a11=[[dot(phix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    b11=[[dot(phiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    a12=[[dot(phix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    b12=[[dot(phiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    a21=[[dot(phix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    b21=[[dot(phiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    a22=[[dot(phix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    b22=[[dot(phiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];

    da11=[[dot(d1xphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    da12=[[dot(d1xphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    db11=[[dot(d1yphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    db12=[[dot(d1yphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dc11=[[dot(d1zphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dc12=[[dot(d1zphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    dd11=[[dot(d1zphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    dd12=[[dot(d1zphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    da22=[[dot(d1xphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    da21=[[dot(d1xphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    db22=[[dot(d1yphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    db21=[[dot(d1yphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    dc22=[[dot(d1zphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    dc21=[[dot(d1zphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dd22=[[dot(d1zphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dd21=[[dot(d1zphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];

    d1c[1]=sum([dot(da11[i],b11[i])+dot(da12[i],b12[i])+dot(da22[i],b22[i])+dot(da21[i],b21[i]) for i in 1:kernel.K])-d1ob(1,x);
    d1c[2]=sum([dot(a11[i],db11[i])+dot(a12[i],db12[i])+dot(a22[i],db22[i])+dot(a21[i],db21[i]) for i in 1:kernel.K])-d1ob(2,x);
    d1c[3]=sum([(dot(dc11[i],b11[i])+dot(a11[i],dd11[i]))+(dot(dc22[i],b22[i])+dot(a22[i],dd22[i]))+(dot(dc12[i],b12[i])+dot(a12[i],dd12[i]))+(dot(dc21[i],b21[i])+dot(a21[i],dd21[i])) for i in 1:kernel.K])-d1ob(3,x);
    return d1c
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Array{Float64,1},1},1},1},1})
    d2c=zeros(kernel.dim,kernel.dim);
    a11=[[dot(phix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    b11=[[dot(phiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    a12=[[dot(phix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    b12=[[dot(phiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    a21=[[dot(phix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    b21=[[dot(phiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    a22=[[dot(phix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    b22=[[dot(phiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];

    da11=[[dot(d1xphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    da12=[[dot(d1xphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    db11=[[dot(d1yphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    db12=[[dot(d1yphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dc11=[[dot(d1zphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dc12=[[dot(d1zphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    dd11=[[dot(d1zphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    dd12=[[dot(d1zphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    da22=[[dot(d1xphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    da21=[[dot(d1xphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    db22=[[dot(d1yphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    db21=[[dot(d1yphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    dc22=[[dot(d1zphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    dc21=[[dot(d1zphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dd22=[[dot(d1zphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dd21=[[dot(d1zphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];


    dac11=[[dot(d11xzphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dac12=[[dot(d11xzphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    dbd11=[[dot(d11yzphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    dbd12=[[dot(d11yzphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dda11=[[dot(d2xphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dda12=[[dot(d2xphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    ddb11=[[dot(d2yphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    ddb12=[[dot(d2yphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    ddc11=[[dot(d2zphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    ddc12=[[dot(d2zphix(x[1],x[3],1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    ddd11=[[dot(d2zphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    ddd12=[[dot(d2zphiy(x[2],x[3],1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];

    dac22=[[dot(d11xzphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    dbd22=[[dot(d11yzphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dda22=[[dot(d2xphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    ddb22=[[dot(d2yphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    ddc22=[[dot(d2zphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[2][1][j][i]) for i in 1:length(Phiu[2][1][j])] for j in 1:kernel.K];
    ddd22=[[dot(d2zphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[2][2][j][i]) for i in 1:length(Phiu[2][2][j])] for j in 1:kernel.K];
    dac21=[[dot(d11xzphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    dbd21=[[dot(d11yzphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    dda21=[[dot(d2xphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    ddb21=[[dot(d2yphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];
    ddc21=[[dot(d2zphix(x[1],x[3],-1.0,kernel.fp[j]),Phiu[1][1][j][i]) for i in 1:length(Phiu[1][1][j])] for j in 1:kernel.K];
    ddd21=[[dot(d2zphiy(x[2],x[3],-1.0,kernel.fp[j]),Phiu[1][2][j][i]) for i in 1:length(Phiu[1][2][j])] for j in 1:kernel.K];

    d2c[1,2]=sum([dot(da11[i],db11[i])+dot(da22[i],db22[i])+dot(da12[i],db12[i])+dot(da21[i],db21[i]) for i in 1:kernel.K])-dot(d11phiVect(1,2,x),y);
    d2c[1,3]=sum([(dot(da11[i],dd11[i])+dot(dac11[i],b11[i]))+(dot(da22[i],dd22[i])+dot(dac22[i],b22[i]))+(dot(da12[i],dd12[i])+dot(dac12[i],b12[i]))+(dot(da21[i],dd21[i])+dot(dac21[i],b21[i])) for i in 1:kernel.K])-dot(d11phiVect(1,3,x),y);
    d2c[2,3]=sum([(dot(dc11[i],db11[i])+dot(a11[i],dbd11[i]))+(dot(dc22[i],db22[i])+dot(a22[i],dbd22[i]))+(dot(dc12[i],db12[i])+dot(a12[i],dbd12[i]))+(dot(dc21[i],db21[i])+dot(a21[i],dbd21[i])) for i in 1:kernel.K])-dot(d11phiVect(2,3,x),y);
    d2c=d2c+d2c';
    d2c[1,1]=sum([dot(dda11[i],b11[i])+dot(dda22[i],b22[i])+dot(dda12[i],b12[i])+dot(dda21[i],b21[i]) for i in 1:kernel.K])-d2ob(1,x);
    d2c[2,2]=sum([dot(a11[i],ddb11[i])+dot(a22[i],ddb22[i])+dot(a12[i],ddb12[i])+dot(a21[i],ddb21[i]) for i in 1:kernel.K])-d2ob(2,x);
    d2c[3,3]=sum([(dot(ddc11[i],b11[i])+dot(a11[i],ddd11[i])+2*dot(dc11[i],dd11[i]))+(dot(ddc22[i],b22[i])+dot(a22[i],ddd22[i])+2*dot(dc22[i],dd22[i]))+(dot(ddc12[i],b12[i])+dot(a12[i],ddd12[i])+2*dot(dc12[i],dd12[i]))+(dot(ddc21[i],b21[i])+dot(a21[i],ddd21[i])+2*dot(dc21[i],dd21[i])) for i in 1:kernel.K])-d2ob(3,x);
    return(d2c)
  end

  gaussconv2DDoubleHelixMultiFP(typeof(kernel),kernel.dim,kernel.bounds,kernel.K,kernel.fp,normObs,Phix,Phiy,PhisY,phix,phiy,phiVect,d1phiVect,d11phiVect,d2phiVect,y,c,d10c,d01c,d11c,d20c,d02c,ob,d1ob,d11ob,d2ob,correl,d1correl,d2correl);
end

function computePhiu(u::Array{Float64,1},op::blasso.gaussconv2DDoubleHelixMultiFP)
  a,X=blasso.decompAmpPos(u,d=op.dim);
  Phiux1=[[a[i]*op.phix(X[i][1],X[i][3],1.0,op.fp[j]) for i in 1:length(a)] for j in 1:op.K];
  Phiuy1=[[op.phiy(X[i][2],X[i][3],1.0,op.fp[j]) for i in 1:length(a)] for j in 1:op.K];
  Phiux2=[[a[i]*op.phix(X[i][1],X[i][3],-1.0,op.fp[j]) for i in 1:length(a)] for j in 1:op.K];
  Phiuy2=[[op.phiy(X[i][2],X[i][3],-1.0,op.fp[j]) for i in 1:length(a)] for j in 1:op.K];
  return [[Phiux1,Phiuy1],[Phiux2,Phiuy2]];
end

# Compute the argmin and min of the correl on the grid.
function minCorrelOnGrid(Phiu::Array{Array{Array{Array{Array{Float64,1},1},1},1},1},kernel::blasso.doubleHelixMultiFP,op::blasso.operator,positivity::Bool=true)
  correl_min,argmin=Inf,zeros(op.dim);
  l=1;
  for k in 1:kernel.nbpointsgrid[3]
    for i in 1:kernel.nbpointsgrid[1]
      a11=[[dot(op.Phix[1][kp][k][i],Phiu[1][1][kp][m]) for m in 1:length(Phiu[1][1][kp])] for kp in 1:kernel.K];
      a12=[[dot(op.Phix[1][kp][k][i],Phiu[2][1][kp][m]) for m in 1:length(Phiu[2][1][kp])] for kp in 1:kernel.K];
      a22=[[dot(op.Phix[2][kp][k][i],Phiu[2][1][kp][m]) for m in 1:length(Phiu[2][1][kp])] for kp in 1:kernel.K];
      a21=[[dot(op.Phix[2][kp][k][i],Phiu[1][1][kp][m]) for m in 1:length(Phiu[1][1][kp])] for kp in 1:kernel.K];
      for j in 1:kernel.nbpointsgrid[2]
        b11=[[dot(op.Phiy[1][kp][k][j],Phiu[1][2][kp][m]) for m in 1:length(Phiu[1][2][kp])] for kp in 1:kernel.K];
        b12=[[dot(op.Phiy[1][kp][k][j],Phiu[2][2][kp][m]) for m in 1:length(Phiu[2][2][kp])] for kp in 1:kernel.K];
        b22=[[dot(op.Phiy[2][kp][k][j],Phiu[2][2][kp][m]) for m in 1:length(Phiu[2][2][kp])] for kp in 1:kernel.K];
        b21=[[dot(op.Phiy[2][kp][k][j],Phiu[1][2][kp][m]) for m in 1:length(Phiu[1][2][kp])] for kp in 1:kernel.K];
        buffer=sum([dot(a11[kp],b11[kp])+dot(a22[kp],b22[kp])+dot(a12[kp],b12[kp])+dot(a21[kp],b21[kp]) for kp in 1:kernel.K])-op.PhisY[l];
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

function setbounds(op::blasso.gaussconv2DDoubleHelixMultiFP,positivity::Bool=true,ampbounds::Bool=true)
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
