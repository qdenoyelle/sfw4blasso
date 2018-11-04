mutable struct astigmatism3DMultiFP <: discrete
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

  sigmax::Function
  sigmay::Function
  d1sigmax::Function
  d1sigmay::Function
  d2sigmax::Function
  d2sigmay::Function
  bounds::Array{Array{Float64,1},1}
end

function setKernel(Npx::Int64,Npy::Int64,Dpx::Float64,Dpy::Float64,lamb::Float64,NA::Float64,fp::Array{Float64,1},
  ni::Float64,bounds::Array{Array{Float64,1},1})

  px=collect(Dpx/2+range(0,stop=(Npx-1)*Dpx,length=Npx));
  py=collect(Dpy/2+range(0,stop=(Npy-1)*Dpy,length=Npy));

  dim=3;
  a,b=bounds[1],bounds[2];
  p=Array{Array{Float64,1}}(undef,0);
  for fpi in fp
    for pyi in py
      for pxi in px
        append!(p,[[pxi,pyi,fpi]]);
      end
    end
  end

  sigma0=2*.21*lamb/NA;
  alpha=-.79;
  beta=.2;
  d=(.5*lamb*ni)/NA^2;

  sigmax(z::Float64,fpi::Float64)=sigma0*sqrt(1+(alpha*(z-fpi)-beta)^2/d^2);
  sigmay(z::Float64,fpi::Float64)=sigmax(-z+2*fpi,fpi);
  d1sigmax(z::Float64,fpi::Float64)=sigma0/d^2*alpha*(alpha*(z-fpi)-beta)*(1+(alpha*(z-fpi)-beta)^2/d^2)^(-.5);
  d1sigmay(z::Float64,fpi::Float64)=-d1sigmax(-z+2*fpi,fpi);
  d2sigmax(z::Float64,fpi::Float64)=alpha^2*sigma0/d^2*( (1+(alpha*(z-fpi)-beta)^2/d^2)^(-.5) - 1/d^2*(alpha*(z-fpi)-beta)^2*(1+(alpha*(z-fpi)-beta)^2/d^2)^(-1.5) );
  d2sigmay(z::Float64,fpi::Float64)=d2sigmax(-z+2*fpi,fpi);

  coeff=.25;
  lSample=[coeff/sigmax(fp[1],fp[1]),coeff/sigmay(fp[1],fp[1]),12];#6
  nbpointsgrid=[convert(Int64,round(5*lSample[i]*abs(b[i]-a[i]);digits=0)) for i in 1:3];
  g=Array{Array{Float64,1}}(undef,dim);
  for i in 1:3
    g[i]=collect(range(a[i],stop=b[i],length=nbpointsgrid[i]));
  end

  return astigmatism3DMultiFP(dim,px,py,p,Npx,Npy,Dpx,Dpy,length(fp),fp,nbpointsgrid,g,sigmax,sigmay,d1sigmax,d1sigmay,d2sigmax,d2sigmay,bounds)
end

mutable struct gaussconv2DAstigmatismMultiFP <: operator
    ker::DataType
    dim::Int64
    bounds::Array{Array{Float64,1},1}
    K::Int64
    fp::Array{Float64,1}

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
function setoperator(kernel::astigmatism3DMultiFP,a0::Array{Float64,1},x0::Array{Array{Float64,1},1})
  function phiDx(x::Float64,z::Float64,xi::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return .5*(erf(alphap/kernel.sigmax(z,fp))-erf(alpham/kernel.sigmax(z,fp)));
  end
  function phiDy(y::Float64,z::Float64,yi::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return .5*(erf(alphap/kernel.sigmay(z,fp))-erf(alpham/kernel.sigmay(z,fp)));
  end
  function d1xphiDx(x::Float64,z::Float64,xi::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmax(z,fp))*(exp(-(alphap/kernel.sigmax(z,fp))^2) - exp(-(alpham/kernel.sigmax(z,fp))^2));
  end


  function d1yphiDy(y::Float64,z::Float64,yi::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmay(z,fp))*(exp(-(alphap/kernel.sigmay(z,fp))^2) - exp(-(alpham/kernel.sigmay(z,fp))^2));
  end
  function d11xzphiDx(x::Float64,z::Float64,xi::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return kernel.d1sigmax(z,fp)/(sqrt(2*pi)*kernel.sigmax(z,fp)^2)*( (exp(-(alphap/kernel.sigmax(z,fp))^2) - exp(-(alpham/kernel.sigmax(z,fp))^2)) - 2/kernel.sigmax(z,fp)^2*(alphap^2*exp(-(alphap/kernel.sigmax(z,fp))^2)-alpham^2*exp(-(alpham/kernel.sigmax(z,fp))^2)) );
  end
  function d11yzphiDy(y::Float64,z::Float64,yi::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return kernel.d1sigmay(z,fp)/(sqrt(2*pi)*kernel.sigmay(z,fp)^2)*( (exp(-(alphap/kernel.sigmay(z,fp))^2) - exp(-(alpham/kernel.sigmay(z,fp))^2)) - 2/kernel.sigmay(z,fp)^2*(alphap^2*exp(-(alphap/kernel.sigmay(z,fp))^2)-alpham^2*exp(-(alpham/kernel.sigmay(z,fp))^2)) );
  end
  function d2xphiDx(x::Float64,z::Float64,xi::Float64,fp::Float64)
    return -1/(sqrt(pi)*kernel.sigmax(z,fp)^2)*( ((xi+kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z,fp)))*exp(-((xi+kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z,fp)))^2) - ((xi-kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z,fp)))*exp(-((xi-kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z,fp)))^2) );
  end
  function d2yphiDy(y::Float64,z::Float64,yi::Float64,fp::Float64)
    return -1/(sqrt(pi)*kernel.sigmay(z,fp)^2)*( ((yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z,fp)))*exp(-((yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z,fp)))^2) - ((yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z,fp)))*exp(-((yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z,fp)))^2) );
  end

  function d1zphiDx(x::Float64,z::Float64,xi::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return -kernel.d1sigmax(z,fp)/(sqrt(pi)*kernel.sigmax(z,fp)^2)*( alphap*exp(-(alphap/kernel.sigmax(z,fp))^2) - alpham*exp(-(alpham/kernel.sigmax(z,fp))^2) );
  end
  function d1zphiDy(y::Float64,z::Float64,yi::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return -kernel.d1sigmay(z,fp)/(sqrt(pi)*kernel.sigmay(z,fp)^2)*( alphap*exp(-(alphap/kernel.sigmay(z,fp))^2) - alpham*exp(-(alpham/kernel.sigmay(z,fp))^2) );
  end

  function d2zphiDx(x::Float64,z::Float64,xi::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return -1/(sqrt(pi)*kernel.sigmax(z,fp)^2)*( alphap*exp(-(alphap/kernel.sigmax(z,fp))^2) - alpham*exp(-(alpham/kernel.sigmax(z,fp))^2) )*(kernel.d2sigmax(z,fp)-2*kernel.d1sigmax(z,fp)^2/kernel.sigmax(z,fp)) - 2*kernel.d1sigmax(z,fp)^2/(sqrt(pi)*kernel.sigmax(z,fp)^5)*(alphap^3*exp(-(alphap/kernel.sigmax(z,fp))^2)-alpham^3*exp(-(alpham/kernel.sigmax(z,fp))^2));
  end
  function d2zphiDy(y::Float64,z::Float64,yi::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return -1/(sqrt(pi)*kernel.sigmay(z,fp)^2)*( alphap*exp(-(alphap/kernel.sigmay(z,fp))^2) - alpham*exp(-(alpham/kernel.sigmay(z,fp))^2) )*(kernel.d2sigmay(z,fp)-2*kernel.d1sigmay(z,fp)^2/kernel.sigmay(z,fp)) - 2*kernel.d1sigmay(z,fp)^2/(sqrt(pi)*kernel.sigmay(z,fp)^5)*(alphap^3*exp(-(alphap/kernel.sigmay(z,fp))^2)-alpham^3*exp(-(alpham/kernel.sigmay(z,fp))^2));
  end

  ##
  function phix(x::Float64,z::Float64,fp::Float64)
    phipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      phipx[i]=phiDx(x,z,kernel.px[i],fp);
    end
    return phipx;
  end
  function phiy(y::Float64,z::Float64,fp::Float64)
    phipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      phipy[i]=phiDy(y,z,kernel.py[i],fp);
    end
    return phipy;
  end

  Phix=Array{Array{Array{Array{Float64,1},1},1}}(kernel.K);
  Phiy=Array{Array{Array{Array{Float64,1},1},1}}(kernel.K);
  for l in 1:kernel.K
    Phix[l]=Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3]);
    Phiy[l]=Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3]);
    for k in 1:kernel.nbpointsgrid[3]
      Phix[l][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[1]);
      Phiy[l][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[2]);
      for i in 1:kernel.nbpointsgrid[1]
        Phix[l][k][i]=phix(kernel.grid[1][i],kernel.grid[3][k],kernel.fp[l]);
      end
      for j in 1:kernel.nbpointsgrid[2]
        Phiy[l][k][j]=phiy(kernel.grid[2][j],kernel.grid[3][k],kernel.fp[l]);
      end
    end
  end

  #
  function d1xphix(x::Float64,z::Float64,fp::Float64)
    d1xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1xphipx[i]=d1xphiDx(x,z,kernel.px[i],fp);
    end
    return d1xphipx;
  end
  function d1yphiy(y::Float64,z::Float64,fp::Float64)
    d1yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1yphipy[i]=d1yphiDy(y,z,kernel.py[i],fp);
    end
    return d1yphipy;
  end
  function d1zphix(x::Float64,z::Float64,fp::Float64)
    d1zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1zphipx[i]=d1zphiDx(x,z,kernel.px[i],fp);
    end
    return d1zphipx;
  end
  function d1zphiy(y::Float64,z::Float64,fp::Float64)
    d1zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1zphipy[i]=d1zphiDy(y,z,kernel.py[i],fp);
    end
    return d1zphipy;
  end
  #

  #
  function d11xzphix(x::Float64,z::Float64,fp::Float64)
    d11xzphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d11xzphipx[i]=d11xzphiDx(x,z,kernel.px[i],fp);
    end
    return d11xzphipx
  end
  function d11yzphiy(y::Float64,z::Float64,fp::Float64)
    d11yzphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d11yzphipy[i]=d11yzphiDy(y,z,kernel.py[i],fp);
    end
    return d11yzphipy
  end
  #

  #
  function d2xphix(x::Float64,z::Float64,fp::Float64)
    d2xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2xphipx[i]=d2xphiDx(x,z,kernel.px[i],fp);
    end
    return d2xphipx;
  end
  function d2yphiy(y::Float64,z::Float64,fp::Float64)
    d2yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2yphipy[i]=d2yphiDy(y,z,kernel.py[i],fp);
    end
    return d2yphipy;
  end
  #

  #
  function d2zphix(x::Float64,z::Float64,fp::Float64)
    d2zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2zphipx[i]=d2zphiDx(x,z,kernel.px[i],fp);
    end
    return d2zphipx;
  end
  function d2zphiy(y::Float64,z::Float64,fp::Float64)
    d2zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2zphipy[i]=d2zphiDy(y,z,kernel.py[i],fp);
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
      phipx=phix(x[1],x[3],kernel.fp[k]);
      phipy=phiy(x[2],x[3],kernel.fp[k]);
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=phipx[i]*phipy[j];
          l+=1;
        end
      end
    end
    return v;
  end
  function d1phiVect(m::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy*kernel.K);
    if m==1
      l=1;
      for k in 1:kernel.K
        #
        phipy=phiy(x[2],x[3],kernel.fp[k]);
        d1xphipx=d1xphix(x[1],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d1xphipx[i]*phipy[j];
            l+=1;
          end
        end
      end # end for k
      return v;
    elseif m==2
      l=1;
      for k in 1:kernel.K
        #
        phipx=phix(x[1],x[3],kernel.fp[k]);
        d1yphipy=d1yphiy(x[2],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=phipx[i]*d1yphipy[j];
            l+=1;
          end
        end
      end # end for k
      return v;
    else
      l=1;
      for k in 1:kernel.K
        #
        phipx=phix(x[1],x[3],kernel.fp[k]);
        phipy=phiy(x[2],x[3],kernel.fp[k]);
        #
        d1zphipx=d1zphix(x[1],x[3],kernel.fp[k]);
        d1zphipy=d1zphiy(x[2],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d1zphipx[i]*phipy[j]+phipx[i]*d1zphipy[j];
            l+=1;
          end
        end
      end #end for k
      return v;
    end
  end
  function d11phiVect(m::Int64,n::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy*kernel.K);
    l=1;
    if m==1 && n==2 || m==2 && n==1
      for k in 1:kernel.K
        #
        d1xphipx=d1xphix(x[1],x[3],kernel.fp[k]);
        d1yphipy=d1yphiy(x[2],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d1xphipx[i]*d1yphipy[j];
            l+=1;
          end
        end
      end #end for k
      return v;
    elseif m==1 && n==3 || m==3 && n==1
      for k in 1:kernel.K
        #
        phipy=phiy(x[2],x[3],kernel.fp[k]);
        #
        d1xphipx=d1xphix(x[1],x[3],kernel.fp[k]);
        d1zphipy=d1zphiy(x[2],x[3],kernel.fp[k]);
        #
        d11xzphipx=d11xzphix(x[1],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d11xzphipx[i]*phipy[j]+d1xphipx[i]*d1zphipy[j];
            l+=1;
          end
        end
      end
      return v;
    elseif m==2 && n==3 || m==3 && n==2
      for k in 1:kernel.K
        #
        phipx=phix(x[1],x[3],kernel.fp[k]);
        #
        d1yphipy=d1yphiy(x[2],x[3],kernel.fp[k]);
        d1zphipx=d1zphix(x[1],x[3],kernel.fp[k]);
        #
        d11yzphipy=d11yzphiy(x[2],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=phipx[i]*d11yzphipy[j]+d1zphipx[i]*d1yphipy[j];
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
        phipy=phiy(x[2],x[3],kernel.fp[k]);
        #
        d2xphipx=d2xphix(x[1],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d2xphipx[i]*phipy[j];
            l+=1;
          end
        end
      end #end for k
      return v;
    elseif m==2
      for k in 1:kernel.K
        #
        phipx=phix(x[1],x[3],kernel.fp[k]);
        #
        d2yphipy=d2yphiy(x[2],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=phipx[i]*d2yphipy[j];
            l+=1;
          end
        end
      end #end for k
      return v;
    else
      for k in 1:kernel.K
        #
        phipx=phix(x[1],x[3],kernel.fp[k]);
        phipy=phiy(x[2],x[3],kernel.fp[k]);
        #
        d1zphipx=d1zphix(x[1],x[3],kernel.fp[k]);
        d1zphipy=d1zphiy(x[2],x[3],kernel.fp[k]);
        #
        d2zphipx=d2zphix(x[1],x[3],kernel.fp[k]);
        d2zphipy=d2zphiy(x[2],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d2zphipx[i]*phipy[j]+phipx[i]*d2zphipy[j]+2*d1zphipx[i]*d1zphipy[j];
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
              v[lp]=Phix[kp][k][i][ip]*Phiy[kp][k][j][jp];
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


  function correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Float64,1},1},1},1})
    a=[[dot(phix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    b=[[dot(phiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    return sum([dot(a[i],b[i]) for i in 1:kernel.K])-ob(x);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Float64,1},1},1},1})
    d1c=zeros(kernel.dim);
    a=[[dot(phix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    b=[[dot(phiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    da=[[dot(d1xphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    db=[[dot(d1yphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    dc=[[dot(d1zphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    dd=[[dot(d1zphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    d1c[1]=sum([dot(da[j],b[j]) for j in 1:kernel.K])-d1ob(1,x);
    d1c[2]=sum([dot(a[j],db[j]) for j in 1:kernel.K])-d1ob(2,x);
    d1c[3]=sum([dot(dc[j],b[j]) + dot(a[j],dd[j]) for j in 1:kernel.K]) - d1ob(3,x);
    return d1c
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Float64,1},1},1},1})
    d2c=zeros(kernel.dim,kernel.dim);
    a=[[dot(phix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    b=[[dot(phiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    da=[[dot(d1xphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    db=[[dot(d1yphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    dc=[[dot(d1zphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    dd=[[dot(d1zphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    dac=[[dot(d11xzphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    dbd=[[dot(d11yzphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    dda=[[dot(d2xphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    ddb=[[dot(d2yphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    ddc=[[dot(d2zphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    ddd=[[dot(d2zphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];

    d2c[1,2]=sum([dot(da[j],db[j]) for j in 1:kernel.K])-dot(d11phiVect(1,2,x),y);
    d2c[1,3]=sum([dot(da[j],dd[j]) + dot(dac[j],b[j]) for j in 1:kernel.K]) - dot(d11phiVect(1,3,x),y);
    d2c[2,3]=sum([dot(dc[j],db[j]) + dot(a[j],dbd[j]) for j in 1:kernel.K]) - dot(d11phiVect(2,3,x),y);
    d2c=d2c+d2c';
    d2c[1,1]=sum([dot(dda[j],b[j]) for j in 1:kernel.K])-d2ob(1,x);
    d2c[2,2]=sum([dot(a[j],ddb[j]) for j in 1:kernel.K])-d2ob(2,x);
    d2c[3,3]=sum([dot(ddc[j],b[j]) + dot(a[j],ddd[j]) + 2*dot(dc[j],dd[j]) for j in 1:kernel.K]) - d2ob(3,x);
    return(d2c)
  end

  gaussconv2DAstigmatismMultiFP(typeof(kernel),kernel.dim,kernel.bounds,kernel.K,kernel.fp,normObs,Phix,Phiy,PhisY,phix,phiy,phiVect,d1phiVect,d11phiVect,d2phiVect,y,c,d10c,d01c,d11c,d20c,d02c,ob,d1ob,d11ob,d2ob,correl,d1correl,d2correl);
end

#
function setoperator(kernel::astigmatism3DMultiFP,a0::Array{Float64,1},x0::Array{Array{Float64,1},1},w::Array{Float64,1})
  function phiDx(x::Float64,z::Float64,xi::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return .5*(erf(alphap/kernel.sigmax(z,fp))-erf(alpham/kernel.sigmax(z,fp)));
  end
  function phiDy(y::Float64,z::Float64,yi::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return .5*(erf(alphap/kernel.sigmay(z,fp))-erf(alpham/kernel.sigmay(z,fp)));
  end
  function d1xphiDx(x::Float64,z::Float64,xi::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmax(z,fp))*(exp(-(alphap/kernel.sigmax(z,fp))^2) - exp(-(alpham/kernel.sigmax(z,fp))^2));
  end


  function d1yphiDy(y::Float64,z::Float64,yi::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmay(z,fp))*(exp(-(alphap/kernel.sigmay(z,fp))^2) - exp(-(alpham/kernel.sigmay(z,fp))^2));
  end
  function d11xzphiDx(x::Float64,z::Float64,xi::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return kernel.d1sigmax(z,fp)/(sqrt(2*pi)*kernel.sigmax(z,fp)^2)*( (exp(-(alphap/kernel.sigmax(z,fp))^2) - exp(-(alpham/kernel.sigmax(z,fp))^2)) - 2/kernel.sigmax(z,fp)^2*(alphap^2*exp(-(alphap/kernel.sigmax(z,fp))^2)-alpham^2*exp(-(alpham/kernel.sigmax(z,fp))^2)) );
  end
  function d11yzphiDy(y::Float64,z::Float64,yi::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return kernel.d1sigmay(z,fp)/(sqrt(2*pi)*kernel.sigmay(z,fp)^2)*( (exp(-(alphap/kernel.sigmay(z,fp))^2) - exp(-(alpham/kernel.sigmay(z,fp))^2)) - 2/kernel.sigmay(z,fp)^2*(alphap^2*exp(-(alphap/kernel.sigmay(z,fp))^2)-alpham^2*exp(-(alpham/kernel.sigmay(z,fp))^2)) );
  end
  function d2xphiDx(x::Float64,z::Float64,xi::Float64,fp::Float64)
    return -1/(sqrt(pi)*kernel.sigmax(z,fp)^2)*( ((xi+kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z,fp)))*exp(-((xi+kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z,fp)))^2) - ((xi-kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z,fp)))*exp(-((xi-kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z,fp)))^2) );
  end
  function d2yphiDy(y::Float64,z::Float64,yi::Float64,fp::Float64)
    return -1/(sqrt(pi)*kernel.sigmay(z,fp)^2)*( ((yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z,fp)))*exp(-((yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z,fp)))^2) - ((yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z,fp)))*exp(-((yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z,fp)))^2) );
  end

  function d1zphiDx(x::Float64,z::Float64,xi::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return -kernel.d1sigmax(z,fp)/(sqrt(pi)*kernel.sigmax(z,fp)^2)*( alphap*exp(-(alphap/kernel.sigmax(z,fp))^2) - alpham*exp(-(alpham/kernel.sigmax(z,fp))^2) );
  end
  function d1zphiDy(y::Float64,z::Float64,yi::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return -kernel.d1sigmay(z,fp)/(sqrt(pi)*kernel.sigmay(z,fp)^2)*( alphap*exp(-(alphap/kernel.sigmay(z,fp))^2) - alpham*exp(-(alpham/kernel.sigmay(z,fp))^2) );
  end

  function d2zphiDx(x::Float64,z::Float64,xi::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return -1/(sqrt(pi)*kernel.sigmax(z,fp)^2)*( alphap*exp(-(alphap/kernel.sigmax(z,fp))^2) - alpham*exp(-(alpham/kernel.sigmax(z,fp))^2) )*(kernel.d2sigmax(z,fp)-2*kernel.d1sigmax(z,fp)^2/kernel.sigmax(z,fp)) - 2*kernel.d1sigmax(z,fp)^2/(sqrt(pi)*kernel.sigmax(z,fp)^5)*(alphap^3*exp(-(alphap/kernel.sigmax(z,fp))^2)-alpham^3*exp(-(alpham/kernel.sigmax(z,fp))^2));
  end
  function d2zphiDy(y::Float64,z::Float64,yi::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return -1/(sqrt(pi)*kernel.sigmay(z,fp)^2)*( alphap*exp(-(alphap/kernel.sigmay(z,fp))^2) - alpham*exp(-(alpham/kernel.sigmay(z,fp))^2) )*(kernel.d2sigmay(z,fp)-2*kernel.d1sigmay(z,fp)^2/kernel.sigmay(z,fp)) - 2*kernel.d1sigmay(z,fp)^2/(sqrt(pi)*kernel.sigmay(z,fp)^5)*(alphap^3*exp(-(alphap/kernel.sigmay(z,fp))^2)-alpham^3*exp(-(alpham/kernel.sigmay(z,fp))^2));
  end

  ##
  function phix(x::Float64,z::Float64,fp::Float64)
    phipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      phipx[i]=phiDx(x,z,kernel.px[i],fp);
    end
    return phipx;
  end
  function phiy(y::Float64,z::Float64,fp::Float64)
    phipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      phipy[i]=phiDy(y,z,kernel.py[i],fp);
    end
    return phipy;
  end

  Phix=Array{Array{Array{Array{Float64,1},1},1}}(kernel.K);
  Phiy=Array{Array{Array{Array{Float64,1},1},1}}(kernel.K);
  for l in 1:kernel.K
    Phix[l]=Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3]);
    Phiy[l]=Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3]);
    for k in 1:kernel.nbpointsgrid[3]
      Phix[l][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[1]);
      Phiy[l][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[2]);
      for i in 1:kernel.nbpointsgrid[1]
        Phix[l][k][i]=phix(kernel.grid[1][i],kernel.grid[3][k],kernel.fp[l]);
      end
      for j in 1:kernel.nbpointsgrid[2]
        Phiy[l][k][j]=phiy(kernel.grid[2][j],kernel.grid[3][k],kernel.fp[l]);
      end
    end
  end

  #
  function d1xphix(x::Float64,z::Float64,fp::Float64)
    d1xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1xphipx[i]=d1xphiDx(x,z,kernel.px[i],fp);
    end
    return d1xphipx;
  end
  function d1yphiy(y::Float64,z::Float64,fp::Float64)
    d1yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1yphipy[i]=d1yphiDy(y,z,kernel.py[i],fp);
    end
    return d1yphipy;
  end
  function d1zphix(x::Float64,z::Float64,fp::Float64)
    d1zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1zphipx[i]=d1zphiDx(x,z,kernel.px[i],fp);
    end
    return d1zphipx;
  end
  function d1zphiy(y::Float64,z::Float64,fp::Float64)
    d1zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1zphipy[i]=d1zphiDy(y,z,kernel.py[i],fp);
    end
    return d1zphipy;
  end
  #

  #
  function d11xzphix(x::Float64,z::Float64,fp::Float64)
    d11xzphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d11xzphipx[i]=d11xzphiDx(x,z,kernel.px[i],fp);
    end
    return d11xzphipx
  end
  function d11yzphiy(y::Float64,z::Float64,fp::Float64)
    d11yzphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d11yzphipy[i]=d11yzphiDy(y,z,kernel.py[i],fp);
    end
    return d11yzphipy
  end
  #

  #
  function d2xphix(x::Float64,z::Float64,fp::Float64)
    d2xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2xphipx[i]=d2xphiDx(x,z,kernel.px[i],fp);
    end
    return d2xphipx;
  end
  function d2yphiy(y::Float64,z::Float64,fp::Float64)
    d2yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2yphipy[i]=d2yphiDy(y,z,kernel.py[i],fp);
    end
    return d2yphipy;
  end
  #

  #
  function d2zphix(x::Float64,z::Float64,fp::Float64)
    d2zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2zphipx[i]=d2zphiDx(x,z,kernel.px[i],fp);
    end
    return d2zphipx;
  end
  function d2zphiy(y::Float64,z::Float64,fp::Float64)
    d2zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2zphipy[i]=d2zphiDy(y,z,kernel.py[i],fp);
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
      phipx=phix(x[1],x[3],kernel.fp[k]);
      phipy=phiy(x[2],x[3],kernel.fp[k]);
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=phipx[i]*phipy[j];
          l+=1;
        end
      end
    end
    return v;
  end
  function d1phiVect(m::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy*kernel.K);
    if m==1
      l=1;
      for k in 1:kernel.K
        #
        phipy=phiy(x[2],x[3],kernel.fp[k]);
        d1xphipx=d1xphix(x[1],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d1xphipx[i]*phipy[j];
            l+=1;
          end
        end
      end # end for k
      return v;
    elseif m==2
      l=1;
      for k in 1:kernel.K
        #
        phipx=phix(x[1],x[3],kernel.fp[k]);
        d1yphipy=d1yphiy(x[2],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=phipx[i]*d1yphipy[j];
            l+=1;
          end
        end
      end # end for k
      return v;
    else
      l=1;
      for k in 1:kernel.K
        #
        phipx=phix(x[1],x[3],kernel.fp[k]);
        phipy=phiy(x[2],x[3],kernel.fp[k]);
        #
        d1zphipx=d1zphix(x[1],x[3],kernel.fp[k]);
        d1zphipy=d1zphiy(x[2],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d1zphipx[i]*phipy[j]+phipx[i]*d1zphipy[j];
            l+=1;
          end
        end
      end #end for k
      return v;
    end
  end
  function d11phiVect(m::Int64,n::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy*kernel.K);
    l=1;
    if m==1 && n==2 || m==2 && n==1
      for k in 1:kernel.K
        #
        d1xphipx=d1xphix(x[1],x[3],kernel.fp[k]);
        d1yphipy=d1yphiy(x[2],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d1xphipx[i]*d1yphipy[j];
            l+=1;
          end
        end
      end #end for k
      return v;
    elseif m==1 && n==3 || m==3 && n==1
      for k in 1:kernel.K
        #
        phipy=phiy(x[2],x[3],kernel.fp[k]);
        #
        d1xphipx=d1xphix(x[1],x[3],kernel.fp[k]);
        d1zphipy=d1zphiy(x[2],x[3],kernel.fp[k]);
        #
        d11xzphipx=d11xzphix(x[1],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d11xzphipx[i]*phipy[j]+d1xphipx[i]*d1zphipy[j];
            l+=1;
          end
        end
      end
      return v;
    elseif m==2 && n==3 || m==3 && n==2
      for k in 1:kernel.K
        #
        phipx=phix(x[1],x[3],kernel.fp[k]);
        #
        d1yphipy=d1yphiy(x[2],x[3],kernel.fp[k]);
        d1zphipx=d1zphix(x[1],x[3],kernel.fp[k]);
        #
        d11yzphipy=d11yzphiy(x[2],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=phipx[i]*d11yzphipy[j]+d1zphipx[i]*d1yphipy[j];
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
        phipy=phiy(x[2],x[3],kernel.fp[k]);
        #
        d2xphipx=d2xphix(x[1],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d2xphipx[i]*phipy[j];
            l+=1;
          end
        end
      end #end for k
      return v;
    elseif m==2
      for k in 1:kernel.K
        #
        phipx=phix(x[1],x[3],kernel.fp[k]);
        #
        d2yphipy=d2yphiy(x[2],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=phipx[i]*d2yphipy[j];
            l+=1;
          end
        end
      end #end for k
      return v;
    else
      for k in 1:kernel.K
        #
        phipx=phix(x[1],x[3],kernel.fp[k]);
        phipy=phiy(x[2],x[3],kernel.fp[k]);
        #
        d1zphipx=d1zphix(x[1],x[3],kernel.fp[k]);
        d1zphipy=d1zphiy(x[2],x[3],kernel.fp[k]);
        #
        d2zphipx=d2zphix(x[1],x[3],kernel.fp[k]);
        d2zphipy=d2zphiy(x[2],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d2zphipx[i]*phipy[j]+phipx[i]*d2zphipy[j]+2*d1zphipx[i]*d1zphipy[j];
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
              v[lp]=Phix[kp][k][i][ip]*Phiy[kp][k][j][jp];
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


  function correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Float64,1},1},1},1})
    a=[[dot(phix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    b=[[dot(phiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    return sum([dot(a[i],b[i]) for i in 1:kernel.K])-ob(x);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Float64,1},1},1},1})
    d1c=zeros(kernel.dim);
    a=[[dot(phix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    b=[[dot(phiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    da=[[dot(d1xphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    db=[[dot(d1yphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    dc=[[dot(d1zphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    dd=[[dot(d1zphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    d1c[1]=sum([dot(da[j],b[j]) for j in 1:kernel.K])-d1ob(1,x);
    d1c[2]=sum([dot(a[j],db[j]) for j in 1:kernel.K])-d1ob(2,x);
    d1c[3]=sum([dot(dc[j],b[j]) + dot(a[j],dd[j]) for j in 1:kernel.K]) - d1ob(3,x);
    return d1c
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Float64,1},1},1},1})
    d2c=zeros(kernel.dim,kernel.dim);
    a=[[dot(phix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    b=[[dot(phiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    da=[[dot(d1xphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    db=[[dot(d1yphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    dc=[[dot(d1zphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    dd=[[dot(d1zphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    dac=[[dot(d11xzphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    dbd=[[dot(d11yzphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    dda=[[dot(d2xphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    ddb=[[dot(d2yphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    ddc=[[dot(d2zphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    ddd=[[dot(d2zphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];

    d2c[1,2]=sum([dot(da[j],db[j]) for j in 1:kernel.K])-dot(d11phiVect(1,2,x),y);
    d2c[1,3]=sum([dot(da[j],dd[j]) + dot(dac[j],b[j]) for j in 1:kernel.K]) - dot(d11phiVect(1,3,x),y);
    d2c[2,3]=sum([dot(dc[j],db[j]) + dot(a[j],dbd[j]) for j in 1:kernel.K]) - dot(d11phiVect(2,3,x),y);
    d2c=d2c+d2c';
    d2c[1,1]=sum([dot(dda[j],b[j]) for j in 1:kernel.K])-d2ob(1,x);
    d2c[2,2]=sum([dot(a[j],ddb[j]) for j in 1:kernel.K])-d2ob(2,x);
    d2c[3,3]=sum([dot(ddc[j],b[j]) + dot(a[j],ddd[j]) + 2*dot(dc[j],dd[j]) for j in 1:kernel.K]) - d2ob(3,x);
    return(d2c)
  end

  gaussconv2DAstigmatismMultiFP(typeof(kernel),kernel.dim,kernel.bounds,kernel.K,kernel.fp,normObs,Phix,Phiy,PhisY,phix,phiy,phiVect,d1phiVect,d11phiVect,d2phiVect,y,c,d10c,d01c,d11c,d20c,d02c,ob,d1ob,d11ob,d2ob,correl,d1correl,d2correl);
end

#
function setoperator(kernel::astigmatism3DMultiFP,a0::Array{Float64,1},x0::Array{Array{Float64,1},1},w::Array{Float64,1},nPhoton::Float64)
  function phiDx(x::Float64,z::Float64,xi::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return .5*(erf(alphap/kernel.sigmax(z,fp))-erf(alpham/kernel.sigmax(z,fp)));
  end
  function phiDy(y::Float64,z::Float64,yi::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return .5*(erf(alphap/kernel.sigmay(z,fp))-erf(alpham/kernel.sigmay(z,fp)));
  end
  function d1xphiDx(x::Float64,z::Float64,xi::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmax(z,fp))*(exp(-(alphap/kernel.sigmax(z,fp))^2) - exp(-(alpham/kernel.sigmax(z,fp))^2));
  end


  function d1yphiDy(y::Float64,z::Float64,yi::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmay(z,fp))*(exp(-(alphap/kernel.sigmay(z,fp))^2) - exp(-(alpham/kernel.sigmay(z,fp))^2));
  end
  function d11xzphiDx(x::Float64,z::Float64,xi::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return kernel.d1sigmax(z,fp)/(sqrt(2*pi)*kernel.sigmax(z,fp)^2)*( (exp(-(alphap/kernel.sigmax(z,fp))^2) - exp(-(alpham/kernel.sigmax(z,fp))^2)) - 2/kernel.sigmax(z,fp)^2*(alphap^2*exp(-(alphap/kernel.sigmax(z,fp))^2)-alpham^2*exp(-(alpham/kernel.sigmax(z,fp))^2)) );
  end
  function d11yzphiDy(y::Float64,z::Float64,yi::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return kernel.d1sigmay(z,fp)/(sqrt(2*pi)*kernel.sigmay(z,fp)^2)*( (exp(-(alphap/kernel.sigmay(z,fp))^2) - exp(-(alpham/kernel.sigmay(z,fp))^2)) - 2/kernel.sigmay(z,fp)^2*(alphap^2*exp(-(alphap/kernel.sigmay(z,fp))^2)-alpham^2*exp(-(alpham/kernel.sigmay(z,fp))^2)) );
  end
  function d2xphiDx(x::Float64,z::Float64,xi::Float64,fp::Float64)
    return -1/(sqrt(pi)*kernel.sigmax(z,fp)^2)*( ((xi+kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z,fp)))*exp(-((xi+kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z,fp)))^2) - ((xi-kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z,fp)))*exp(-((xi-kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z,fp)))^2) );
  end
  function d2yphiDy(y::Float64,z::Float64,yi::Float64,fp::Float64)
    return -1/(sqrt(pi)*kernel.sigmay(z,fp)^2)*( ((yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z,fp)))*exp(-((yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z,fp)))^2) - ((yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z,fp)))*exp(-((yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z,fp)))^2) );
  end

  function d1zphiDx(x::Float64,z::Float64,xi::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return -kernel.d1sigmax(z,fp)/(sqrt(pi)*kernel.sigmax(z,fp)^2)*( alphap*exp(-(alphap/kernel.sigmax(z,fp))^2) - alpham*exp(-(alpham/kernel.sigmax(z,fp))^2) );
  end
  function d1zphiDy(y::Float64,z::Float64,yi::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return -kernel.d1sigmay(z,fp)/(sqrt(pi)*kernel.sigmay(z,fp)^2)*( alphap*exp(-(alphap/kernel.sigmay(z,fp))^2) - alpham*exp(-(alpham/kernel.sigmay(z,fp))^2) );
  end

  function d2zphiDx(x::Float64,z::Float64,xi::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return -1/(sqrt(pi)*kernel.sigmax(z,fp)^2)*( alphap*exp(-(alphap/kernel.sigmax(z,fp))^2) - alpham*exp(-(alpham/kernel.sigmax(z,fp))^2) )*(kernel.d2sigmax(z,fp)-2*kernel.d1sigmax(z,fp)^2/kernel.sigmax(z,fp)) - 2*kernel.d1sigmax(z,fp)^2/(sqrt(pi)*kernel.sigmax(z,fp)^5)*(alphap^3*exp(-(alphap/kernel.sigmax(z,fp))^2)-alpham^3*exp(-(alpham/kernel.sigmax(z,fp))^2));
  end
  function d2zphiDy(y::Float64,z::Float64,yi::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return -1/(sqrt(pi)*kernel.sigmay(z,fp)^2)*( alphap*exp(-(alphap/kernel.sigmay(z,fp))^2) - alpham*exp(-(alpham/kernel.sigmay(z,fp))^2) )*(kernel.d2sigmay(z,fp)-2*kernel.d1sigmay(z,fp)^2/kernel.sigmay(z,fp)) - 2*kernel.d1sigmay(z,fp)^2/(sqrt(pi)*kernel.sigmay(z,fp)^5)*(alphap^3*exp(-(alphap/kernel.sigmay(z,fp))^2)-alpham^3*exp(-(alpham/kernel.sigmay(z,fp))^2));
  end

  ##
  function phix(x::Float64,z::Float64,fp::Float64)
    phipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      phipx[i]=phiDx(x,z,kernel.px[i],fp);
    end
    return phipx;
  end
  function phiy(y::Float64,z::Float64,fp::Float64)
    phipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      phipy[i]=phiDy(y,z,kernel.py[i],fp);
    end
    return phipy;
  end

  Phix=Array{Array{Array{Array{Float64,1},1},1}}(kernel.K);
  Phiy=Array{Array{Array{Array{Float64,1},1},1}}(kernel.K);
  for l in 1:kernel.K
    Phix[l]=Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3]);
    Phiy[l]=Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3]);
    for k in 1:kernel.nbpointsgrid[3]
      Phix[l][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[1]);
      Phiy[l][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[2]);
      for i in 1:kernel.nbpointsgrid[1]
        Phix[l][k][i]=phix(kernel.grid[1][i],kernel.grid[3][k],kernel.fp[l]);
      end
      for j in 1:kernel.nbpointsgrid[2]
        Phiy[l][k][j]=phiy(kernel.grid[2][j],kernel.grid[3][k],kernel.fp[l]);
      end
    end
  end

  #
  function d1xphix(x::Float64,z::Float64,fp::Float64)
    d1xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1xphipx[i]=d1xphiDx(x,z,kernel.px[i],fp);
    end
    return d1xphipx;
  end
  function d1yphiy(y::Float64,z::Float64,fp::Float64)
    d1yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1yphipy[i]=d1yphiDy(y,z,kernel.py[i],fp);
    end
    return d1yphipy;
  end
  function d1zphix(x::Float64,z::Float64,fp::Float64)
    d1zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1zphipx[i]=d1zphiDx(x,z,kernel.px[i],fp);
    end
    return d1zphipx;
  end
  function d1zphiy(y::Float64,z::Float64,fp::Float64)
    d1zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1zphipy[i]=d1zphiDy(y,z,kernel.py[i],fp);
    end
    return d1zphipy;
  end
  #

  #
  function d11xzphix(x::Float64,z::Float64,fp::Float64)
    d11xzphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d11xzphipx[i]=d11xzphiDx(x,z,kernel.px[i],fp);
    end
    return d11xzphipx
  end
  function d11yzphiy(y::Float64,z::Float64,fp::Float64)
    d11yzphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d11yzphipy[i]=d11yzphiDy(y,z,kernel.py[i],fp);
    end
    return d11yzphipy
  end
  #

  #
  function d2xphix(x::Float64,z::Float64,fp::Float64)
    d2xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2xphipx[i]=d2xphiDx(x,z,kernel.px[i],fp);
    end
    return d2xphipx;
  end
  function d2yphiy(y::Float64,z::Float64,fp::Float64)
    d2yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2yphipy[i]=d2yphiDy(y,z,kernel.py[i],fp);
    end
    return d2yphipy;
  end
  #

  #
  function d2zphix(x::Float64,z::Float64,fp::Float64)
    d2zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2zphipx[i]=d2zphiDx(x,z,kernel.px[i],fp);
    end
    return d2zphipx;
  end
  function d2zphiy(y::Float64,z::Float64,fp::Float64)
    d2zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2zphipy[i]=d2zphiDy(y,z,kernel.py[i],fp);
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
      phipx=phix(x[1],x[3],kernel.fp[k]);
      phipy=phiy(x[2],x[3],kernel.fp[k]);
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=phipx[i]*phipy[j];
          l+=1;
        end
      end
    end
    return v;
  end
  function d1phiVect(m::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy*kernel.K);
    if m==1
      l=1;
      for k in 1:kernel.K
        #
        phipy=phiy(x[2],x[3],kernel.fp[k]);
        d1xphipx=d1xphix(x[1],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d1xphipx[i]*phipy[j];
            l+=1;
          end
        end
      end # end for k
      return v;
    elseif m==2
      l=1;
      for k in 1:kernel.K
        #
        phipx=phix(x[1],x[3],kernel.fp[k]);
        d1yphipy=d1yphiy(x[2],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=phipx[i]*d1yphipy[j];
            l+=1;
          end
        end
      end # end for k
      return v;
    else
      l=1;
      for k in 1:kernel.K
        #
        phipx=phix(x[1],x[3],kernel.fp[k]);
        phipy=phiy(x[2],x[3],kernel.fp[k]);
        #
        d1zphipx=d1zphix(x[1],x[3],kernel.fp[k]);
        d1zphipy=d1zphiy(x[2],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d1zphipx[i]*phipy[j]+phipx[i]*d1zphipy[j];
            l+=1;
          end
        end
      end #end for k
      return v;
    end
  end
  function d11phiVect(m::Int64,n::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy*kernel.K);
    l=1;
    if m==1 && n==2 || m==2 && n==1
      for k in 1:kernel.K
        #
        d1xphipx=d1xphix(x[1],x[3],kernel.fp[k]);
        d1yphipy=d1yphiy(x[2],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d1xphipx[i]*d1yphipy[j];
            l+=1;
          end
        end
      end #end for k
      return v;
    elseif m==1 && n==3 || m==3 && n==1
      for k in 1:kernel.K
        #
        phipy=phiy(x[2],x[3],kernel.fp[k]);
        #
        d1xphipx=d1xphix(x[1],x[3],kernel.fp[k]);
        d1zphipy=d1zphiy(x[2],x[3],kernel.fp[k]);
        #
        d11xzphipx=d11xzphix(x[1],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d11xzphipx[i]*phipy[j]+d1xphipx[i]*d1zphipy[j];
            l+=1;
          end
        end
      end
      return v;
    elseif m==2 && n==3 || m==3 && n==2
      for k in 1:kernel.K
        #
        phipx=phix(x[1],x[3],kernel.fp[k]);
        #
        d1yphipy=d1yphiy(x[2],x[3],kernel.fp[k]);
        d1zphipx=d1zphix(x[1],x[3],kernel.fp[k]);
        #
        d11yzphipy=d11yzphiy(x[2],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=phipx[i]*d11yzphipy[j]+d1zphipx[i]*d1yphipy[j];
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
        phipy=phiy(x[2],x[3],kernel.fp[k]);
        #
        d2xphipx=d2xphix(x[1],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d2xphipx[i]*phipy[j];
            l+=1;
          end
        end
      end #end for k
      return v;
    elseif m==2
      for k in 1:kernel.K
        #
        phipx=phix(x[1],x[3],kernel.fp[k]);
        #
        d2yphipy=d2yphiy(x[2],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=phipx[i]*d2yphipy[j];
            l+=1;
          end
        end
      end #end for k
      return v;
    else
      for k in 1:kernel.K
        #
        phipx=phix(x[1],x[3],kernel.fp[k]);
        phipy=phiy(x[2],x[3],kernel.fp[k]);
        #
        d1zphipx=d1zphix(x[1],x[3],kernel.fp[k]);
        d1zphipy=d1zphiy(x[2],x[3],kernel.fp[k]);
        #
        d2zphipx=d2zphix(x[1],x[3],kernel.fp[k]);
        d2zphipy=d2zphiy(x[2],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d2zphipx[i]*phipy[j]+phipx[i]*d2zphipy[j]+2*d1zphipx[i]*d1zphipy[j];
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
              v[lp]=Phix[kp][k][i][ip]*Phiy[kp][k][j][jp];
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


  function correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Float64,1},1},1},1})
    a=[[dot(phix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    b=[[dot(phiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    return sum([dot(a[i],b[i]) for i in 1:kernel.K])-ob(x);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Float64,1},1},1},1})
    d1c=zeros(kernel.dim);
    a=[[dot(phix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    b=[[dot(phiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    da=[[dot(d1xphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    db=[[dot(d1yphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    dc=[[dot(d1zphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    dd=[[dot(d1zphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    d1c[1]=sum([dot(da[j],b[j]) for j in 1:kernel.K])-d1ob(1,x);
    d1c[2]=sum([dot(a[j],db[j]) for j in 1:kernel.K])-d1ob(2,x);
    d1c[3]=sum([dot(dc[j],b[j]) + dot(a[j],dd[j]) for j in 1:kernel.K]) - d1ob(3,x);
    return d1c
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Float64,1},1},1},1})
    d2c=zeros(kernel.dim,kernel.dim);
    a=[[dot(phix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    b=[[dot(phiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    da=[[dot(d1xphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    db=[[dot(d1yphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    dc=[[dot(d1zphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    dd=[[dot(d1zphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    dac=[[dot(d11xzphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    dbd=[[dot(d11yzphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    dda=[[dot(d2xphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    ddb=[[dot(d2yphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    ddc=[[dot(d2zphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    ddd=[[dot(d2zphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];

    d2c[1,2]=sum([dot(da[j],db[j]) for j in 1:kernel.K])-dot(d11phiVect(1,2,x),y);
    d2c[1,3]=sum([dot(da[j],dd[j]) + dot(dac[j],b[j]) for j in 1:kernel.K]) - dot(d11phiVect(1,3,x),y);
    d2c[2,3]=sum([dot(dc[j],db[j]) + dot(a[j],dbd[j]) for j in 1:kernel.K]) - dot(d11phiVect(2,3,x),y);
    d2c=d2c+d2c';
    d2c[1,1]=sum([dot(dda[j],b[j]) for j in 1:kernel.K])-d2ob(1,x);
    d2c[2,2]=sum([dot(a[j],ddb[j]) for j in 1:kernel.K])-d2ob(2,x);
    d2c[3,3]=sum([dot(ddc[j],b[j]) + dot(a[j],ddd[j]) + 2*dot(dc[j],dd[j]) for j in 1:kernel.K]) - d2ob(3,x);
    return(d2c)
  end

  gaussconv2DAstigmatismMultiFP(typeof(kernel),kernel.dim,kernel.bounds,kernel.K,kernel.fp,normObs,Phix,Phiy,PhisY,phix,phiy,phiVect,d1phiVect,d11phiVect,d2phiVect,y,c,d10c,d01c,d11c,d20c,d02c,ob,d1ob,d11ob,d2ob,correl,d1correl,d2correl);
end

# gaussian convolution in 2D when only y is given
function setoperator(kernel::astigmatism3DMultiFP,y::Array{Float64,1})
  function phiDx(x::Float64,z::Float64,xi::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return .5*(erf(alphap/kernel.sigmax(z,fp))-erf(alpham/kernel.sigmax(z,fp)));
  end
  function phiDy(y::Float64,z::Float64,yi::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return .5*(erf(alphap/kernel.sigmay(z,fp))-erf(alpham/kernel.sigmay(z,fp)));
  end
  function d1xphiDx(x::Float64,z::Float64,xi::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmax(z,fp))*(exp(-(alphap/kernel.sigmax(z,fp))^2) - exp(-(alpham/kernel.sigmax(z,fp))^2));
  end


  function d1yphiDy(y::Float64,z::Float64,yi::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return -1/(sqrt(2*pi)*kernel.sigmay(z,fp))*(exp(-(alphap/kernel.sigmay(z,fp))^2) - exp(-(alpham/kernel.sigmay(z,fp))^2));
  end
  function d11xzphiDx(x::Float64,z::Float64,xi::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return kernel.d1sigmax(z,fp)/(sqrt(2*pi)*kernel.sigmax(z,fp)^2)*( (exp(-(alphap/kernel.sigmax(z,fp))^2) - exp(-(alpham/kernel.sigmax(z,fp))^2)) - 2/kernel.sigmax(z,fp)^2*(alphap^2*exp(-(alphap/kernel.sigmax(z,fp))^2)-alpham^2*exp(-(alpham/kernel.sigmax(z,fp))^2)) );
  end
  function d11yzphiDy(y::Float64,z::Float64,yi::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return kernel.d1sigmay(z,fp)/(sqrt(2*pi)*kernel.sigmay(z,fp)^2)*( (exp(-(alphap/kernel.sigmay(z,fp))^2) - exp(-(alpham/kernel.sigmay(z,fp))^2)) - 2/kernel.sigmay(z,fp)^2*(alphap^2*exp(-(alphap/kernel.sigmay(z,fp))^2)-alpham^2*exp(-(alpham/kernel.sigmay(z,fp))^2)) );
  end
  function d2xphiDx(x::Float64,z::Float64,xi::Float64,fp::Float64)
    return -1/(sqrt(pi)*kernel.sigmax(z,fp)^2)*( ((xi+kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z,fp)))*exp(-((xi+kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z,fp)))^2) - ((xi-kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z,fp)))*exp(-((xi-kernel.Dpx/2-x)/(sqrt(2)*kernel.sigmax(z,fp)))^2) );
  end
  function d2yphiDy(y::Float64,z::Float64,yi::Float64,fp::Float64)
    return -1/(sqrt(pi)*kernel.sigmay(z,fp)^2)*( ((yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z,fp)))*exp(-((yi+kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z,fp)))^2) - ((yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z,fp)))*exp(-((yi-kernel.Dpy/2-y)/(sqrt(2)*kernel.sigmay(z,fp)))^2) );
  end

  function d1zphiDx(x::Float64,z::Float64,xi::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return -kernel.d1sigmax(z,fp)/(sqrt(pi)*kernel.sigmax(z,fp)^2)*( alphap*exp(-(alphap/kernel.sigmax(z,fp))^2) - alpham*exp(-(alpham/kernel.sigmax(z,fp))^2) );
  end
  function d1zphiDy(y::Float64,z::Float64,yi::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return -kernel.d1sigmay(z,fp)/(sqrt(pi)*kernel.sigmay(z,fp)^2)*( alphap*exp(-(alphap/kernel.sigmay(z,fp))^2) - alpham*exp(-(alpham/kernel.sigmay(z,fp))^2) );
  end

  function d2zphiDx(x::Float64,z::Float64,xi::Float64,fp::Float64)
    alphap,alpham=(xi+kernel.Dpx/2-x)/sqrt(2),(xi-kernel.Dpx/2-x)/sqrt(2);
    return -1/(sqrt(pi)*kernel.sigmax(z,fp)^2)*( alphap*exp(-(alphap/kernel.sigmax(z,fp))^2) - alpham*exp(-(alpham/kernel.sigmax(z,fp))^2) )*(kernel.d2sigmax(z,fp)-2*kernel.d1sigmax(z,fp)^2/kernel.sigmax(z,fp)) - 2*kernel.d1sigmax(z,fp)^2/(sqrt(pi)*kernel.sigmax(z,fp)^5)*(alphap^3*exp(-(alphap/kernel.sigmax(z,fp))^2)-alpham^3*exp(-(alpham/kernel.sigmax(z,fp))^2));
  end
  function d2zphiDy(y::Float64,z::Float64,yi::Float64,fp::Float64)
    alphap,alpham=(yi+kernel.Dpy/2-y)/sqrt(2),(yi-kernel.Dpy/2-y)/sqrt(2);
    return -1/(sqrt(pi)*kernel.sigmay(z,fp)^2)*( alphap*exp(-(alphap/kernel.sigmay(z,fp))^2) - alpham*exp(-(alpham/kernel.sigmay(z,fp))^2) )*(kernel.d2sigmay(z,fp)-2*kernel.d1sigmay(z,fp)^2/kernel.sigmay(z,fp)) - 2*kernel.d1sigmay(z,fp)^2/(sqrt(pi)*kernel.sigmay(z,fp)^5)*(alphap^3*exp(-(alphap/kernel.sigmay(z,fp))^2)-alpham^3*exp(-(alpham/kernel.sigmay(z,fp))^2));
  end

  ##
  function phix(x::Float64,z::Float64,fp::Float64)
    phipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      phipx[i]=phiDx(x,z,kernel.px[i],fp);
    end
    return phipx;
  end
  function phiy(y::Float64,z::Float64,fp::Float64)
    phipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      phipy[i]=phiDy(y,z,kernel.py[i],fp);
    end
    return phipy;
  end

  Phix=Array{Array{Array{Array{Float64,1},1},1}}(kernel.K);
  Phiy=Array{Array{Array{Array{Float64,1},1},1}}(kernel.K);
  for l in 1:kernel.K
    Phix[l]=Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3]);
    Phiy[l]=Array{Array{Array{Float64,1},1}}(kernel.nbpointsgrid[3]);
    for k in 1:kernel.nbpointsgrid[3]
      Phix[l][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[1]);
      Phiy[l][k]=Array{Array{Float64,1},1}(kernel.nbpointsgrid[2]);
      for i in 1:kernel.nbpointsgrid[1]
        Phix[l][k][i]=phix(kernel.grid[1][i],kernel.grid[3][k],kernel.fp[l]);
      end
      for j in 1:kernel.nbpointsgrid[2]
        Phiy[l][k][j]=phiy(kernel.grid[2][j],kernel.grid[3][k],kernel.fp[l]);
      end
    end
  end

  #
  function d1xphix(x::Float64,z::Float64,fp::Float64)
    d1xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1xphipx[i]=d1xphiDx(x,z,kernel.px[i],fp);
    end
    return d1xphipx;
  end
  function d1yphiy(y::Float64,z::Float64,fp::Float64)
    d1yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1yphipy[i]=d1yphiDy(y,z,kernel.py[i],fp);
    end
    return d1yphipy;
  end
  function d1zphix(x::Float64,z::Float64,fp::Float64)
    d1zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d1zphipx[i]=d1zphiDx(x,z,kernel.px[i],fp);
    end
    return d1zphipx;
  end
  function d1zphiy(y::Float64,z::Float64,fp::Float64)
    d1zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d1zphipy[i]=d1zphiDy(y,z,kernel.py[i],fp);
    end
    return d1zphipy;
  end
  #

  #
  function d11xzphix(x::Float64,z::Float64,fp::Float64)
    d11xzphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d11xzphipx[i]=d11xzphiDx(x,z,kernel.px[i],fp);
    end
    return d11xzphipx
  end
  function d11yzphiy(y::Float64,z::Float64,fp::Float64)
    d11yzphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d11yzphipy[i]=d11yzphiDy(y,z,kernel.py[i],fp);
    end
    return d11yzphipy
  end
  #

  #
  function d2xphix(x::Float64,z::Float64,fp::Float64)
    d2xphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2xphipx[i]=d2xphiDx(x,z,kernel.px[i],fp);
    end
    return d2xphipx;
  end
  function d2yphiy(y::Float64,z::Float64,fp::Float64)
    d2yphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2yphipy[i]=d2yphiDy(y,z,kernel.py[i],fp);
    end
    return d2yphipy;
  end
  #

  #
  function d2zphix(x::Float64,z::Float64,fp::Float64)
    d2zphipx=zeros(kernel.Npx);
    for i in 1:kernel.Npx
      d2zphipx[i]=d2zphiDx(x,z,kernel.px[i],fp);
    end
    return d2zphipx;
  end
  function d2zphiy(y::Float64,z::Float64,fp::Float64)
    d2zphipy=zeros(kernel.Npy);
    for i in 1:kernel.Npy
      d2zphipy[i]=d2zphiDy(y,z,kernel.py[i],fp);
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
      phipx=phix(x[1],x[3],kernel.fp[k]);
      phipy=phiy(x[2],x[3],kernel.fp[k]);
      for j in 1:kernel.Npy
        for i in 1:kernel.Npx
          v[l]=phipx[i]*phipy[j];
          l+=1;
        end
      end
    end
    return v;
  end
  function d1phiVect(m::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy*kernel.K);
    if m==1
      l=1;
      for k in 1:kernel.K
        #
        phipy=phiy(x[2],x[3],kernel.fp[k]);
        d1xphipx=d1xphix(x[1],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d1xphipx[i]*phipy[j];
            l+=1;
          end
        end
      end # end for k
      return v;
    elseif m==2
      l=1;
      for k in 1:kernel.K
        #
        phipx=phix(x[1],x[3],kernel.fp[k]);
        d1yphipy=d1yphiy(x[2],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=phipx[i]*d1yphipy[j];
            l+=1;
          end
        end
      end # end for k
      return v;
    else
      l=1;
      for k in 1:kernel.K
        #
        phipx=phix(x[1],x[3],kernel.fp[k]);
        phipy=phiy(x[2],x[3],kernel.fp[k]);
        #
        d1zphipx=d1zphix(x[1],x[3],kernel.fp[k]);
        d1zphipy=d1zphiy(x[2],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d1zphipx[i]*phipy[j]+phipx[i]*d1zphipy[j];
            l+=1;
          end
        end
      end #end for k
      return v;
    end
  end
  function d11phiVect(m::Int64,n::Int64,x::Array{Float64,1})
    v=zeros(kernel.Npx*kernel.Npy*kernel.K);
    l=1;
    if m==1 && n==2 || m==2 && n==1
      for k in 1:kernel.K
        #
        d1xphipx=d1xphix(x[1],x[3],kernel.fp[k]);
        d1yphipy=d1yphiy(x[2],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d1xphipx[i]*d1yphipy[j];
            l+=1;
          end
        end
      end #end for k
      return v;
    elseif m==1 && n==3 || m==3 && n==1
      for k in 1:kernel.K
        #
        phipy=phiy(x[2],x[3],kernel.fp[k]);
        #
        d1xphipx=d1xphix(x[1],x[3],kernel.fp[k]);
        d1zphipy=d1zphiy(x[2],x[3],kernel.fp[k]);
        #
        d11xzphipx=d11xzphix(x[1],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d11xzphipx[i]*phipy[j]+d1xphipx[i]*d1zphipy[j];
            l+=1;
          end
        end
      end
      return v;
    elseif m==2 && n==3 || m==3 && n==2
      for k in 1:kernel.K
        #
        phipx=phix(x[1],x[3],kernel.fp[k]);
        #
        d1yphipy=d1yphiy(x[2],x[3],kernel.fp[k]);
        d1zphipx=d1zphix(x[1],x[3],kernel.fp[k]);
        #
        d11yzphipy=d11yzphiy(x[2],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=phipx[i]*d11yzphipy[j]+d1zphipx[i]*d1yphipy[j];
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
        phipy=phiy(x[2],x[3],kernel.fp[k]);
        #
        d2xphipx=d2xphix(x[1],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d2xphipx[i]*phipy[j];
            l+=1;
          end
        end
      end #end for k
      return v;
    elseif m==2
      for k in 1:kernel.K
        #
        phipx=phix(x[1],x[3],kernel.fp[k]);
        #
        d2yphipy=d2yphiy(x[2],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=phipx[i]*d2yphipy[j];
            l+=1;
          end
        end
      end #end for k
      return v;
    else
      for k in 1:kernel.K
        #
        phipx=phix(x[1],x[3],kernel.fp[k]);
        phipy=phiy(x[2],x[3],kernel.fp[k]);
        #
        d1zphipx=d1zphix(x[1],x[3],kernel.fp[k]);
        d1zphipy=d1zphiy(x[2],x[3],kernel.fp[k]);
        #
        d2zphipx=d2zphix(x[1],x[3],kernel.fp[k]);
        d2zphipy=d2zphiy(x[2],x[3],kernel.fp[k]);
        #
        for j in 1:kernel.Npy
          for i in 1:kernel.Npx
            v[l]=d2zphipx[i]*phipy[j]+phipx[i]*d2zphipy[j]+2*d1zphipx[i]*d1zphipy[j];
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
              v[lp]=Phix[kp][k][i][ip]*Phiy[kp][k][j][jp];
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


  function correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Float64,1},1},1},1})
    a=[[dot(phix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    b=[[dot(phiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    return sum([dot(a[i],b[i]) for i in 1:kernel.K])-ob(x);
  end
  function d1correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Float64,1},1},1},1})
    d1c=zeros(kernel.dim);
    a=[[dot(phix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    b=[[dot(phiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    da=[[dot(d1xphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    db=[[dot(d1yphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    dc=[[dot(d1zphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    dd=[[dot(d1zphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    d1c[1]=sum([dot(da[j],b[j]) for j in 1:kernel.K])-d1ob(1,x);
    d1c[2]=sum([dot(a[j],db[j]) for j in 1:kernel.K])-d1ob(2,x);
    d1c[3]=sum([dot(dc[j],b[j]) + dot(a[j],dd[j]) for j in 1:kernel.K]) - d1ob(3,x);
    return d1c
  end
  function d2correl(x::Array{Float64,1},Phiu::Array{Array{Array{Array{Float64,1},1},1},1})
    d2c=zeros(kernel.dim,kernel.dim);
    a=[[dot(phix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    b=[[dot(phiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    da=[[dot(d1xphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    db=[[dot(d1yphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    dc=[[dot(d1zphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    dd=[[dot(d1zphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    dac=[[dot(d11xzphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    dbd=[[dot(d11yzphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    dda=[[dot(d2xphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    ddb=[[dot(d2yphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];
    ddc=[[dot(d2zphix(x[1],x[3],kernel.fp[j]),Phiu[1][j][i]) for i in 1:length(Phiu[1][j])] for j in 1:kernel.K];
    ddd=[[dot(d2zphiy(x[2],x[3],kernel.fp[j]),Phiu[2][j][i]) for i in 1:length(Phiu[2][j])] for j in 1:kernel.K];

    d2c[1,2]=sum([dot(da[j],db[j]) for j in 1:kernel.K])-dot(d11phiVect(1,2,x),y);
    d2c[1,3]=sum([dot(da[j],dd[j]) + dot(dac[j],b[j]) for j in 1:kernel.K]) - dot(d11phiVect(1,3,x),y);
    d2c[2,3]=sum([dot(dc[j],db[j]) + dot(a[j],dbd[j]) for j in 1:kernel.K]) - dot(d11phiVect(2,3,x),y);
    d2c=d2c+d2c';
    d2c[1,1]=sum([dot(dda[j],b[j]) for j in 1:kernel.K])-d2ob(1,x);
    d2c[2,2]=sum([dot(a[j],ddb[j]) for j in 1:kernel.K])-d2ob(2,x);
    d2c[3,3]=sum([dot(ddc[j],b[j]) + dot(a[j],ddd[j]) + 2*dot(dc[j],dd[j]) for j in 1:kernel.K]) - d2ob(3,x);
    return(d2c)
  end

  gaussconv2DAstigmatismMultiFP(typeof(kernel),kernel.dim,kernel.bounds,kernel.K,kernel.fp,normObs,Phix,Phiy,PhisY,phix,phiy,phiVect,d1phiVect,d11phiVect,d2phiVect,y,c,d10c,d01c,d11c,d20c,d02c,ob,d1ob,d11ob,d2ob,correl,d1correl,d2correl);
end


function computePhiu(u::Array{Float64,1},op::blasso.gaussconv2DAstigmatismMultiFP)
  a,X=blasso.decompAmpPos(u,d=op.dim);
  Phiux=[[a[i]*op.phix(X[i][1],X[i][3],op.fp[j]) for i in 1:length(a)] for j in 1:op.K];
  Phiuy=[[op.phiy(X[i][2],X[i][3],op.fp[j]) for i in 1:length(a)] for j in 1:op.K];
  return [Phiux,Phiuy];
end

# Compute the argmin and min of the correl on the grid.
function minCorrelOnGrid(Phiu::Array{Array{Array{Array{Float64,1},1},1},1},kernel::blasso.astigmatism3DMultiFP,op::blasso.operator,positivity::Bool=true)
  correl_min,argmin=Inf,zeros(op.dim);
  l=1;
  for k in 1:kernel.nbpointsgrid[3]
    for i in 1:kernel.nbpointsgrid[1]
      a=[[dot(op.Phix[kp][k][i],Phiu[1][kp][m]) for m in 1:length(Phiu[1][kp])] for kp in 1:kernel.K];
      for j in 1:kernel.nbpointsgrid[2]
        b=[[dot(op.Phiy[kp][k][j],Phiu[2][kp][m]) for m in 1:length(Phiu[2][kp])] for kp in 1:kernel.K];
        buffer=sum([dot(a[kp],b[kp]) for kp in 1:kernel.K])-op.PhisY[l];
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

function setbounds(op::blasso.gaussconv2DAstigmatismMultiFP,positivity::Bool=true,ampbounds::Bool=true)
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
