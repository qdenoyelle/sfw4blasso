

module blasso

using LinearAlgebra, PyPlot

#####################################################################################

# Abstract parent type for the different kernels
abstract type kernel end
abstract type discrete <: kernel end
abstract type continuous <: kernel end
abstract type dLaplace <: discrete end
abstract type cLaplace <: continuous end


# Abstract parent type for the different operators
abstract type operator end

include("forward-models/dirichlet.jl");
include("forward-models/gaussian.jl");
include("forward-models/gaussian2D.jl");
include("forward-models/gaussian2DLaplace.jl");
include("forward-models/astigmatism3D.jl");
include("forward-models/doubleHelix.jl");
include("forward-models/doubleHelixMultiFP.jl");
include("forward-models/astigmatism3DMultiFP.jl");
include("forward-models/dnlaplace.jl");


function checkOp(op::blasso.operator)
  if op.dim == 1
    x=op.bounds[1]+rand()*(op.bounds[2]-op.bounds[1]);
    y=op.bounds[1]+rand()*(op.bounds[2]-op.bounds[1]);
    eps_d1=1e-7;
    eps_d2=1e-3;
    tol=1e-4;

    a,b=(op.c(x+eps_d1,y)-op.c(x-eps_d1,y))/(2*eps_d1),op.d10c(x,y)
    if abs(a-b)/abs(b)<tol
      message="Ok";
    else
      message="Error";
    end
    println("Approx d10c(x,y)=",a,", d10c(x,y)=",b,", ",message);

    a,b=(op.c(x,y+eps_d1)-op.c(x,y-eps_d1))/(2*eps_d1),op.d01c(x,y);
    if abs(a-b)/abs(b)<tol
      message="Ok";
    else
      message="Error";
    end
    println("Approx d01c(x,y)=",a,", d01c(x,y)=",b,", ",message);

    a,b=(op.d10c(x+eps_d1,y)-op.d10c(x-eps_d1,y))/(2*eps_d1),op.d20c(x,y);
    if abs(a-b)/abs(b)<tol
      message="Ok";
    else
      message="Error";
    end
    println("Approx d20c(x,y)=",a,", d20c(x,y)=",b,", ",message);

    a,b=(op.d01c(x,y+eps_d1)-op.d01c(x,y-eps_d1))/(2*eps_d1),op.d02c(x,y);
    if abs(a-b)/abs(b)<tol
      message="Ok";
    else
      message="Error";
    end
    println("Approx d02c(x,y)=",a,", d02c(x,y)=",b,", ",message);

    a,b=(op.d10c(x,y+eps_d1)-op.d10c(x,y-eps_d1))/(2*eps_d1),op.d11c(x,y);
    if abs(a-b)/abs(b)<tol
      message="Ok";
    else
      message="Error";
    end
    println("Approx d11c(x,y)=",a,", d11c(x,y)=",b,", ",message);

    a,b=(op.ob(x+eps_d1)-op.ob(x-eps_d1))/(2*eps_d1),op.d1ob(x);
    if abs(a-b)/abs(b)<tol
      message="Ok";
    else
      message="Error";
    end
    println("Approx d1ob(x)=",a,", d1ob(x)=",b,", ",message);

    a,b=(op.d1ob(x+eps_d1)-op.d1ob(x-eps_d1))/(2*eps_d1),op.d2ob(x);
    if abs(a-b)/abs(b)<tol
      message="Ok";
    else
      message="Error";
    end
    println("Approx d2ob(x)=",a,", d2ob(x)=",b,", ",message);
  else
    x=op.bounds[1]+rand()*(op.bounds[2]-op.bounds[1]);
    y=op.bounds[1]+rand()*(op.bounds[2]-op.bounds[1]);
    eps_d1=1e-5;
    eps_d2=1e-5;
    tol=1e-4;
    e=Array{Array{Float64,1}}(undef,op.dim);
    for i in 1:op.dim
      e[i]=zeros(op.dim);
      e[i][i]=1.0;
    end

    for i in 1:op.dim
      println("Coordinates : ",i);
      a,b=(op.c(x+eps_d1*e[i],y)-op.c(x-eps_d1*e[i],y))/(2*eps_d1),op.d10c(i,x,y);
      if abs(a-b)/abs(b)<tol
        message="Ok";
      else
        message="Error";
      end
      println("Approx d10c(x,y)=",a,", d10c(x,y)=",b,", ",message, " : ",abs(a-b)/abs(b));

      a,b=(op.c(x,y+eps_d1*e[i])-op.c(x,y-eps_d1*e[i]))/(2*eps_d1),op.d01c(i,x,y);
      if abs(a-b)/abs(b)<tol
        message="Ok";
      else
        message="Error";
      end
      println("Approx d01c(x,y)=",a,", d01c(x,y)=",b,", ",message, " : ",abs(a-b)/abs(b));

      a,b=(op.d10c(i,x+eps_d1*e[i],y)-op.d10c(i,x-eps_d1*e[i],y))/(2*eps_d1),op.d20c(i,x,y);
      if abs(a-b)/abs(b)<tol
        message="Ok";
      else
        message="Error";
      end
      println("Approx d20c(x,y)=",a,", d20c(x,y)=",b,", ",message, " : ",abs(a-b)/abs(b));

      a,b=(op.d01c(i,x,y+eps_d1*e[i])-op.d01c(i,x,y-eps_d1*e[i]))/(2*eps_d1),op.d02c(i,x,y);
      if abs(a-b)/abs(b)<tol
        message="Ok";
      else
        message="Error";
      end
      println("Approx d02c(x,y)=",a,", d02c(x,y)=",b,", ",message, " : ",abs(a-b)/abs(b));

      a,b=(op.ob(x+eps_d1*e[i])-op.ob(x-eps_d1*e[i]))/(2*eps_d1),op.d1ob(i,x);
      if abs(a-b)/abs(b)<tol
        message="Ok";
      else
        message="Error";
      end
      println("Approx d1ob(x)=",a,", d1ob(x)=",b,", ",message, " : ",abs(a-b)/abs(b));

      a,b=(op.d1ob(i,x+eps_d1*e[i])-op.d1ob(i,x-eps_d1*e[i]))/(2*eps_d1),op.d2ob(i,x);
      if abs(a-b)/abs(b)<tol
        message="Ok";
      else
        message="Error";
      end
      println("Approx d2ob(x)=",a,", d2ob(x)=",b,", ",message, " : ",abs(a-b)/abs(b));
    end

    if :phi in fieldnames(typeof(op))
      N=5;
      a=rand(N);
      x=Array{Float64}(undef,0);
      for i in 1:op.dim
        append!(x,op.bounds[1][i]+rand(N)*(op.bounds[2][i]-op.bounds[1][i]));
      end
      u=vcat(a,x);
      Phiu=blasso.computePhiu(u,op);
      correl(x::Array{Float64,1})=op.correl(x,Phiu);
      d1correl(x::Array{Float64,1})=op.d1correl(x,Phiu);
      d2correl(x::Array{Float64,1})=op.d2correl(x,Phiu);
      e=Array{Array{Float64,1}}(undef,(op.dim));
      x,y=blasso.pointsInDomain(op);
      for i in 1:(op.dim)
          e[i]=zeros((op.dim));
          e[i][i]=1.0;
      end

      g_approx=zeros((op.dim));
      for i in 1:(op.dim)
          g_approx[i]=(correl(x+eps_d1*e[i])-correl(x-eps_d1*e[i]))/(2*eps_d1);
      end

      g=d1correl(x);
      if maximum(g_approx-g)<tol
          message="Ok";
      else
          message="Error";
      end
      println("\nApprox d1correl(x)=",g_approx,", d1correl(x)=",g,", ",message, " : ",maximum(g_approx-g));

      h_approx=zeros((op.dim),(op.dim));
      for i in 1:(op.dim)
          h_approx[:,i]=(d1correl(x+eps_d1*e[i])-d1correl(x-eps_d1*e[i]))./(2*eps_d1);
      end
      t=time();
      h=d2correl(x);
      if maximum(h_approx-h)<tol
          message="Ok";
      else
          message="Error";
      end
      println("Approx d2correl(x)=",h_approx,", d2correl(x)=",h,", ",message, " : ",maximum(h_approx-h));
    end

  end
end

# Plot observations
function plotobservation(op::operator;kwargs...)
  if op.dim == 1
    u=range(op.bounds[1],stop=op.bounds[2],length=500);
    obser=Array{Float64}(undef,length(u));
    for i in 1:length(u)
        obser[i]=op.ob(u[i]);
    end
    figure(figsize=(4,3));
    plot(u,obser)
    show()
  elseif op.dim == 2
    u=range(op.bounds[1][1],stop=op.bounds[2][1],length=100);
    v=range(op.bounds[1][1],stop=op.bounds[2][2],length=100);
    U=Array{Float64}(undef,0);
    V=Array{Float64}(undef,0);
    Ob=zeros(length(u),length(v));
    for i in 1:length(u)
      for j in 1:length(v)
        append!(U,u[i]);
        append!(V,v[j]);
        Ob[j,i]=op.ob([u[i],v[j]]); #i column -> x axis
      end
    end

    cm=ColorMap("coolwarm");
    cs=contourf(u,v,Ob,200,cmap=cm,interpolation="bicubic",linestyles="None")
    cb = colorbar(cs)
    show()
  elseif op.dim == 3
    key_kw=[kwargs[i][1] for i in 1:length(kwargs)];
    if :ngrid in key_kw
      ngrid=kwargs[find(key_kw.==:ngrid)][1][2];
    else
      ngrid=20;
    end
    if :z in key_kw
      z=kwargs[find(key_kw.==:z)][1][2];
      u=range(op.bounds[1][1],stop=op.bounds[2][1],length=ngrid);
      v=range(op.bounds[1][2],stop=op.bounds[2][2],length=ngrid);
    elseif :y in key_kw
      y=kwargs[find(key_kw.==:y)][1][2];
      u=range(op.bounds[1][1],stop=op.bounds[2][1],length=ngrid);
      v=range(op.bounds[1][3],stop=op.bounds[2][3],length=ngrid);
    elseif :x in key_kw
      x=kwargs[find(key_kw.==:x)][1][2];
      u=range(op.bounds[1][2],stop=op.bounds[2][2],length=ngrid);
      v=range(op.bounds[1][3],stop=op.bounds[2][3],length=ngrid);
    else
      z=(op.bounds[1][3]+op.bounds[2][3])/2;
      u=range(op.bounds[1][1],stop=op.bounds[2][1],length=ngrid);
      v=range(op.bounds[1][2],stop=op.bounds[2][2],length=ngrid);
    end


    U=Array{Float64}(undef,0);
    V=Array{Float64}(undef,0);
    Ob=zeros(length(v),length(u));
    for i in 1:length(u)
      for j in 1:length(v)
        append!(U,u[i]);
        append!(V,v[j]);
        if :x in key_kw
          Ob[j,i]=op.ob([x,u[i],v[j]]); #i column -> x axis
        elseif :y in key_kw
          Ob[j,i]=op.ob([u[i],y,v[j]]);
        else
          Ob[j,i]=op.ob([u[i],v[j],z]);
        end
      end
    end

    cm=ColorMap("coolwarm");
    cs=contourf(u,v,Ob,50,cmap=cm,interpolation="bicubic",linestyles="None")
    cb = colorbar(cs)
    if :clim in key_kw
      clim=kwargs[find(key_kw.==:clim)][1][2];
      cs[:set_clim](clim[1], clim[2])
    end
    show()
  end
end
# Plot observations
function plotobservation(u::Array{Float64,1},op::operator;kwargs...)
  Phiu=blasso.computePhiu(u,op);
  correl(x::Array{Float64,1})=op.correl(x,Phiu);
  if op.dim == 1
    u=range(op.bounds[1],stop=op.bounds[2],length=500);
    obser=Array{Float64}(undef,length(u));
    for i in 1:length(u)
        obser[i]=correl([u[i]]);
    end
    figure(figsize=(4,3));
    plot(u,obser)
    show()
  elseif op.dim == 2
    u=range(op.bounds[1][1],stop=op.bounds[2][1],length=100);
    v=range(op.bounds[1][1],stop=op.bounds[2][2],length=100);
    U=Array{Float64}(undef,0);
    V=Array{Float64}(undef,0);
    Ob=zeros(length(u),length(v));
    for i in 1:length(u)
      for j in 1:length(v)
        append!(U,u[i]);
        append!(V,v[j]);
        Ob[j,i]=correl([u[i],v[j]]); #i column -> x axis
      end
    end

    cm=ColorMap("coolwarm");
    cs=contourf(u,v,Ob,200,cmap=cm,interpolation="bicubic",linestyles="None")
    cb = colorbar(cs)
    show()
  elseif op.dim == 3
    key_kw=[kwargs[i][1] for i in 1:length(kwargs)];
    if :ngrid in key_kw
      ngrid=kwargs[find(key_kw.==:ngrid)][1][2];
    else
      ngrid=20;
    end
    if :z in key_kw
      z=kwargs[find(key_kw.==:z)][1][2];
      u=range(op.bounds[1][1],stop=op.bounds[2][1],length=ngrid);
      v=range(op.bounds[1][2],stop=op.bounds[2][2],length=ngrid);
    elseif :y in key_kw
      y=kwargs[find(key_kw.==:y)][1][2];
      u=range(op.bounds[1][1],stop=op.bounds[2][1],length=ngrid);
      v=range(op.bounds[1][3],stop=op.bounds[2][3],length=ngrid);
    elseif :x in key_kw
      x=kwargs[find(key_kw.==:x)][1][2];
      u=range(op.bounds[1][2],stop=op.bounds[2][2],length=ngrid);
      v=range(op.bounds[1][3],stop=op.bounds[2][3],length=ngrid);
    else
      z=(op.bounds[1][3]+op.bounds[2][3])/2;
      u=range(op.bounds[1][1],stop=op.bounds[2][1],length=ngrid);
      v=range(op.bounds[1][2],stop=op.bounds[2][2],length=ngrid);
    end

    U=Array{Float64}(undef,0);
    V=Array{Float64}(undef,0);
    Ob=zeros(length(v),length(u));
    for i in 1:length(u)
      for j in 1:length(v)
        append!(U,u[i]);
        append!(V,v[j]);
        if :x in key_kw
          Ob[j,i]=correl([x,u[i],v[j]]); #i column -> x axis
        elseif :y in key_kw
          Ob[j,i]=correl([u[i],y,v[j]]);
        else
          Ob[j,i]=correl([u[i],v[j],z]);
        end
      end
    end

    cm=ColorMap("coolwarm");
    cs=contourf(u,v,Ob,50,cmap=cm,interpolation="bicubic",linestyles="None")
    cb = colorbar(cs)
    show()
  end
end

# Objective function (and its gradient and hessian)
struct fobj{}
  lambda::Real

  f::Function
  g::Function
  h::Function
end

function setfobj(op::operator,lambda::Float64)
  function energy(u::Array{Float64,1},op::operator,lambda::Real)
    a,x=blasso.decompAmpPos(u,d=op.dim);
    N=length(a);
    if :phi in fieldnames(typeof(op))
      vect=true;
      Phi=sum([a[i]*op.phi(x[i]) for i in 1:length(a)]);
    else
      vect=false;
    end

    if vect
      residual=Phi-op.y;
      return .5*dot(residual,residual) + lambda*sum(abs.(a));
    else
      C=Array{Float64}(undef,(N,N));

      @simd for i in 1:N
          @simd for j in 1:N
              @inbounds C[i,j]=op.c(x[i],x[j]);
          end
      end
      return op.normObs + .5*dot(At_mul_B(a,C),a) - sum([a[i]*op.ob(x[i]) for i in 1:N]) + lambda*sum(abs(a));
    end
  end
  function computeGradientEnergy(u::Array{Float64,1},op::blasso.operator=op,lambda::Real=lambda)
    a,x=decompAmpPos(u,d=op.dim);
    N=length(a);

    if :phi in fieldnames(typeof(op))
      vect=true;
      Phia=sum([a[i]*op.phi(x[i]) for i in 1:length(a)]);
      p=Phia-op.y;
      K=length(p);
      if op.dim==1
        d1Phia=zeros(K,N);
        for i in 1:N
          d1Phia[:,i]=a[i]*op.d1phi(x[i]);
        end
      else
        d1Phia=zeros(K,op.dim*N);
        for i in 1:N
          for k in 0:(op.dim-1)
            d1Phia[:,i+k*N]=a[i]*op.d1phi(k+1,x[i]);
          end
        end
      end
      daf,dxf=zeros(N),zeros(op.dim*N);
      for i in 1:N
        daf[i]=dot(op.phi(x[i]),p) + lambda*sign(a[i]);
      end
      dxf=d1Phia'*p;
      return vcat(daf,dxf);
    else ##
      O,buffer=Array{Float64}(undef,(1+op.dim)*N),Array{Float64}(undef,(1+op.dim)*N);
      C=zeros(Float64,((1+op.dim)*N,(1+op.dim)*N));
      if op.dim==1
        @simd for i in 1:N
            @simd for j in 1:N
                @inbounds C[i,j]=op.c(x[i],x[j]);
                @inbounds C[i+N,j+N]=0.5*(op.d10c(x[i],x[j])+op.d01c(x[j],x[i]));
            end
            @inbounds O[i]=-op.ob(x[i]);
            @inbounds O[i+N]=-op.d1ob(x[i]);
        end
        mul!(buffer,C,vcat(a,a));
        return(vcat(ones(N),a).*(buffer+O+lambda*vcat(sign.(a),zeros(N))));
      else
        @simd for i in 1:N
            @simd for j in 1:N
                @inbounds C[i,j]=op.c(x[i],x[j]);
                for k in 1:op.dim
                  @inbounds C[i+k*N,j+k*N]=0.5*(op.d10c(k,x[i],x[j])+op.d01c(k,x[j],x[i]));
                end
            end
            @inbounds O[i]=-op.ob(x[i]);
            for k in 1:op.dim
              @inbounds O[i+k*N]=-op.d1ob(k,x[i]);
            end
        end
        if op.dim==2
          mul!(buffer,C,vcat(a,a,a));
          return(vcat(ones(N),a,a).*(buffer+O+lambda*vcat(sign.(a),zeros(N),zeros(N))));
        elseif op.dim==3
          mul!(buffer,C,vcat(a,a,a,a));
          return(vcat(ones(N),a,a,a).*(buffer+O+lambda*vcat(sign.(a),zeros(N),zeros(N),zeros(N))));
        end
      end
    end
  end
  function computeGradientEnergy(u::Array{Float64,1},n::Array{Int64,1},op::blasso.operator=op,lambda::Real=lambda)
    a,x=decompAmpPos(u,d=op.dim);
    N,K=length(a),length(n);

    if :phi in fieldnames(typeof(op))
      Phia=sum([a[i]*op.phi(x[i]) for i in 1:length(a)]);
      p=Phia-op.y;
      Kp=length(p);
      if op.dim==1
        d1Phia=zeros(Kp,K);
        for i in 1:K
          d1Phia[:,i]=a[n[i]]*op.d1phi(x[n[i]]);
        end
      else
        d1Phia=zeros(Kp,op.dim*K);
        for i in 1:K
          for k in 0:(op.dim-1)
            d1Phia[:,i+k*K]=a[n[i]]*op.d1phi(k+1,x[n[i]]);
          end
        end
      end
      daf,dxf=zeros(K),zeros(op.dim*K);
      for i in 1:K
        daf[i]=dot(op.phi(x[n[i]]),p) + lambda*sign(a[n[i]]);
      end
      dxf=d1Phia'*p;
      return vcat(daf,dxf);
    end
  end

  function computeHessianEnergy(u::Array{Float64,1},op::operator)
    a,x=decompAmpPos(u,d=op.dim);
    N=length(a);
    h=zeros(Float64,((1+op.dim)*N,(1+op.dim)*N));
    d1C,d11C,d2C=Array{Float64}(undef,N),Array{Float64}(undef,N),Array{Float64}(undef,N);

    if op.dim==1
      @simd for i in 1:2N-1
        @simd for j in i+1:2N
          if i<=N && j<=N
            I,J=i,j;
            @inbounds h[i,j]=op.c(x[I],x[J]);
          end
          if i<=N && j>N
            I,J=i,j-N;
            for K in 1:N
              @inbounds d1C[K]=op.d10c(x[I],x[K])+op.d01c(x[K],x[I]);
            end
            @inbounds h[i,j]=0.5*a[J]*(op.d10c(x[J],x[I])+op.d01c(x[I],x[J])) + (I==J)*(.5*dot(a,d1C) - op.d1ob(x[I]));
          end

          if i>N
            I,J=i-N,j-N;
            @inbounds h[i,j]=0.5*a[J]*a[I]*(op.d11c(x[J],x[I])+op.d11c(x[I],x[J]));
          end
        end
      end

      h=h+h';

      @simd for i in 1:2N
        if i<=N
          I=i;
          @inbounds h[i,i]=op.c(x[I],x[I]);
        else
          I=i-N;
          @simd for K in 1:N
            @inbounds d2C[K]=op.d20c(x[I],x[K])+op.d02c(x[K],x[I]);
          end
          @inbounds h[i,i]=a[I]^2*op.d11c(x[I],x[I])-a[I]*op.d2ob(x[I])+.5*a[I]*dot(a,d2C);
        end
      end

    elseif op.dim==2 #op.dim>1
      @simd for i in 1:N-1
        @simd for j in i+1:N
          I,J=i,j;
          @inbounds h[i,j]=op.c(x[I],x[J]);
        end
      end
      @simd for i in 1:N
        @simd for j in 1:N
          @simd for k in 1:op.dim
            I,J=i,j;
            @simd for K in 1:N
              @inbounds d1C[K]=op.d10c(k,x[I],x[K])+op.d01c(k,x[K],x[I]);
            end
            @inbounds h[i,j+k*N]=0.5*a[J]*(op.d10c(k,x[J],x[I])+op.d01c(k,x[I],x[J])) + (I==J)*(.5*dot(a,d1C) - op.d1ob(k,x[I]));
          end
        end
      end
      @simd for j in 2:N ###
        @simd for i in 1:j-1
          I,J=i,j;
          for k in 1:op.dim
            @inbounds h[i+k*N,j+k*N]=.5*a[I]*a[J]*( op.d11c((k-1)*op.dim+1,(k-1)*op.dim+2,x[I],x[J]) + op.d11c((k-1)*op.dim+2,(k-1)*op.dim+1,x[J],x[I]) );
          end
        end
      end
      @simd for k in 1:(op.dim-1)
        @simd for l in k+1:op.dim
          @simd for i in 1:N
            @simd for j in 1:N
              I,J=i+k*N,j+l*N;
              @simd for m in 1:N
                d11C[m]=(op.d11c((k-1)*op.dim+1,(l-1)*op.dim+1,x[i],x[m])+op.d11c((k-1)*op.dim+2,(l-1)*op.dim+2,x[m],x[i]));
              end
              #@inbounds h[I,J]=0.5*a[i]*a[j]*(op.d11c((k-1)*op.dim+1,(l)*op.dim,x[i],x[j])+op.d11c((l)*op.dim,(k-1)*op.dim+1,x[j],x[i]))+1*(i==j)*(.5*a[i]*dot(a,d11C) - a[i]*op.d11ob(x[i]));
              @inbounds h[I,J]=0.5*a[i]*a[j]*(op.d11c((k-1)*op.dim+2,(l-1)*op.dim+1,x[i],x[j])+op.d11c((k-1)*op.dim+1,(l-1)*op.dim+2,x[j],x[i]))+1*(i==j)*(.5*a[i]*dot(a,d11C) - a[i]*op.d11ob((k-1)*op.dim+1,(l-1)*op.dim+1,x[i]));
            end
          end
        end
      end

      h=h+h';

      @simd for i in 1:2N
        if i<=N
          I=i;
          @inbounds h[I,I]=op.c(x[I],x[I]);
        else
          I=i-N;
          for k in 1:op.dim
            @simd for K in 1:N
              @inbounds d2C[K]=op.d20c(k,x[I],x[K])+op.d02c(k,x[K],x[I]);
            end
            @inbounds h[I+k*N,I+k*N]=a[I]^2*op.d11c((k-1)*op.dim+1,(k-1)*op.dim+2,x[I],x[I])-a[I]*op.d2ob(k,x[I])+.5*a[I]*dot(a,d2C);
          end
        end
      end


    elseif op.dim==3 #op.dim>1
      @simd for i in 1:N-1
        @simd for j in i+1:N
          I,J=i,j;
          @inbounds h[i,j]=op.c(x[I],x[J]);
        end
      end
      @simd for i in 1:N
        @simd for j in 1:N
          @simd for k in 1:op.dim
            I,J=i,j;
            @simd for K in 1:N
              @inbounds d1C[K]=op.d10c(k,x[I],x[K])+op.d01c(k,x[K],x[I]);
            end
            @inbounds h[i,j+k*N]=0.5*a[J]*(op.d10c(k,x[J],x[I])+op.d01c(k,x[I],x[J])) + (I==J)*(.5*dot(a,d1C) - op.d1ob(k,x[I]));
          end
        end
      end
      @simd for j in 2:N ###
        @simd for i in 1:j-1
          I,J=i,j;
          for k in 1:op.dim
            @inbounds h[i+k*N,j+k*N]=.5*a[I]*a[J]*( op.d11c(2*(k-1)+1,2*(k-1)+2,x[I],x[J]) + op.d11c(2*(k-1)+2,2*(k-1)+1,x[J],x[I]) );
          end
        end
      end
      @simd for k in 1:(op.dim-1)
        @simd for l in k+1:op.dim
          @simd for i in 1:N
            @simd for j in 1:N
              I,J=i+k*N,j+l*N;
              @simd for m in 1:N
                d11C[m]=(op.d11c(2*(k-1)+1,2*(l-1)+1,x[i],x[m])+op.d11c(2*(k-1)+2,2*(l-1)+2,x[m],x[i]));
              end
              #@inbounds h[I,J]=0.5*a[i]*a[j]*(op.d11c((k-1)*op.dim+1,(l)*op.dim,x[i],x[j])+op.d11c((l)*op.dim,(k-1)*op.dim+1,x[j],x[i]))+1*(i==j)*(.5*a[i]*dot(a,d11C) - a[i]*op.d11ob(x[i]));
              @inbounds h[I,J]=0.5*a[i]*a[j]*(op.d11c(2*(k-1)+2,2*(l-1)+1,x[i],x[j])+op.d11c(2*(k-1)+1,2*(l-1)+2,x[j],x[i]))+1*(i==j)*(.5*a[i]*dot(a,d11C) - a[i]*op.d11ob(2*(k-1)+1,2*(l-1)+1,x[i]));
            end
          end
        end
      end

      h=h+h';

      @simd for i in 1:2N
        if i<=N
          I=i;
          @inbounds h[I,I]=op.c(x[I],x[I]);
        else
          I=i-N;
          for k in 1:op.dim
            @simd for K in 1:N
              @inbounds d2C[K]=op.d20c(k,x[I],x[K])+op.d02c(k,x[K],x[I]);
            end
            @inbounds h[I+k*N,I+k*N]=a[I]^2*op.d11c(2*(k-1)+1,2*(k-1)+2,x[I],x[I])-a[I]*op.d2ob(k,x[I])+.5*a[I]*dot(a,d2C);
          end
        end
      end

    end #if op.dim


    return h;
  end

  f(u::Array{Float64,1})=energy(u,op,lambda);
  g(u::Array{Float64,1})=computeGradientEnergy(u,op,lambda);
  g(u::Array{Float64,1},n::Array{Int64,1})=computeGradientEnergy(u,n,op,lambda);
  h(u::Array{Float64,1})=computeHessianEnergy(u,op);

  fobj(lambda,f,g,h)
end #function

function checkfobj(fobj::blasso.fobj,op::blasso.operator;N::Int64=3)
  a=rand(N);
  x=Array{Float64}(undef,0);
  for i in 1:op.dim
    append!(x,op.bounds[1][i]+rand(N)*(op.bounds[2][i]-op.bounds[1][i]));
  end
  u=vcat(a,x);
  eps_d1=1e-5;
  eps_d2=1e-5;
  tol=1e-6;

  e=Array{Array{Float64,1}}(undef,(1+op.dim)*N);
  for i in 1:(1+op.dim)*N
    e[i]=zeros((1+op.dim)*N);
    e[i][i]=1.0;
  end

  g_approx=zeros((1+op.dim)*N);
  for i in 1:(1+op.dim)*N
    g_approx[i]=(fobj.f(u+eps_d1*e[i])-fobj.f(u-eps_d1*e[i]))/(2*eps_d1);
  end

  t=time();
  g=fobj.g(u);
  println("Time to compute fobj.g(u) : ",time()-t," s");
  if maximum(g_approx-g)<tol
  #if norm(g_approx-g)/norm(g)<tol
    message="Ok";
  else
    message="Error";
  end
  println("Approx g(u)=",g_approx,", g(u)=",g,", ",message, " : ",maximum(g_approx-g));

  h_approx=zeros((1+op.dim)*N,(1+op.dim)*N);
  for i in 1:(1+op.dim)*N
    h_approx[:,i]=(fobj.g(u+eps_d1*e[i])-fobj.g(u-eps_d1*e[i]))./(2*eps_d1);
  end
  t=time();
  h=fobj.h(u);
  println("Time to compute fobj.h(u) : ",time()-t," s");
  if maximum(h_approx-h)<tol
  #if norm(h_approx-h)/norm(h)<tol
    message="Ok";
  else
    message="Error";
  end
  println("Approx h(u)=",h_approx,", h(u)=",h,", ",message, " : ",maximum(h_approx-h));
end

function lassoFISTA(x::Array{Float64,1},a_previous::Array{Float64,1},op::operator,lambda::Real,tol::Real=1e-3,positive::Bool=true,max_iter::Integer=5000000)
  N=length(x);
  o,C=Array{Float64}(undef,N),Array{Float64}(undef,(N,N));
  a,buffer=zeros(N),zeros(N);

  @simd for i in 1:N
    @simd for j in 1:N
      @inbounds C[i,j]=op.c(x[i],x[j]);
    end
    @inbounds o[i]=op.ob(x[i]);
  end
  L=norm(C);
  alpha=1/L;

  A_previous=copy(a_previous);
  A=copy(a);
  t_previous=1.0;

  i=0;
  convergence=false;
  while !convergence
    mul!(buffer,C,A_previous);
    a=copy(A_previous-alpha*(buffer-o));
    if positive
      a=max.(blasso.softthreshold(a,alpha*lambda),0.0);
    else
      a=blasso.softthreshold(a,alpha*lambda);
    end

    t=(1+sqrt(1+4*t_previous^2))/2;
    mu=(t_previous-1)/t;

    A=copy(a+mu*(a-a_previous));
    i+=1;
    if i>=max_iter || norm(A-A_previous)<tol
      convergence=true;
    end
    a_previous=copy(a);
    A_previous=copy(A);
    t_previous=t;
  end

  return(A_previous);
end
function lassoFISTA(x::Array{Array{Float64,1},1},a_previous::Array{Float64,1},op::operator,lambda::Real,tol::Real=1e-3,positive::Bool=true,max_iter::Integer=5000000)
  N=length(x);
  o,C=Array{Float64}(undef,N),Array{Float64}(undef,(N,N));
  a,buffer=zeros(N),zeros(N);

  @simd for i in 1:N
    @simd for j in 1:N
      @inbounds C[i,j]=op.c(x[i],x[j]);
    end
    @inbounds o[i]=op.ob(x[i]);
  end
  L=norm(C);
  alpha=1/L;

  A_previous=copy(a_previous);
  A=copy(a);
  t_previous=1.0;

  i=0;
  convergence=false;
  while !convergence
    mul!(buffer,C,A_previous);
    a=copy(A_previous-alpha*(buffer-o));
    if positive
      a=max.(blasso.softthreshold(a,alpha*lambda),0.0);
    else
      a=blasso.softthreshold(a,alpha*lambda);
    end

    t=(1+sqrt(1+4*t_previous^2))/2;
    mu=(t_previous-1)/t;

    A=copy(a+mu*(a-a_previous));
    i+=1;
    if i>=max_iter || norm(A-A_previous)<tol
      convergence=true;
      if i>=max_iter
        #println("# Max iter in lassoIST #, delta_a=",norm(a-a_previous));
      end
    end
    a_previous=copy(a);
    A_previous=copy(A);
    t_previous=t;
  end

  return(A_previous);
end
function softthreshold(v::Array{Float64,1},t::Real)
  N=length(v);

  @fastmath @inbounds @simd for i in 1:N
    if v[i]>t
      v[i]=v[i]-t;
    elseif v[i]<-t
      v[i]=v[i]+t;
    else
      v[i]=0.0;
    end
  end

  return(v);
end

function remainInDomain(u::Array{Float64,1},bounds::Array{Float64,1})
  N=convert(Integer,length(u)/2);
  u_domain=Array{Float64}(undef,2N);
  z=Array{Float64}(undef,N);
  u_domain[1:N]=u[1:N];
  z=u[N+1:2N];

  if z[argmin(z)]<bounds[1] || z[argmax(z)]>bounds[2]
    println("Out of Domain!");
    for i in 1:N
      if z[i]<bounds[1]
        z[i]=bounds[1];
      end
      if z[i]>bounds[2]
        z[i]=bounds[2];
      end
    end
  end
  u_domain[N+1:2N]=z;

  return u_domain;
end

function computeMinSepDist(x::Array{Float64},ker::DataType)
  N=length(x);

  if N==1
    Delta=Inf;
  else
    dx=0.0;
    p=sortperm(x);
    x=x[p];

    if ker<:blasso.dirichlet
      Delta=1.0;
    else
      Delta=x[2]-x[1];
    end

    for i in 1:N-1
      dx=x[i+1]-x[i];
      if dx < Delta
        Delta=dx;
      end
    end

    if ker==:dirichlet
      dx=x[1]-(x[N]-1);
      if dx<Delta;
        Delta=dx;
      end
    end
  end
  return Delta
end

function sortspikes!(a::Array{Float64,1},x::Array{Float64,1})
  N=length(a);
  not_sorted=true;
  while not_sorted
    not_sorted=false;
    for i in 1:N-1
      if a[i]<a[i+1]
        a_buffer=a[i];
        x_buffer=x[i];
        a[i],x[i]=a[i+1],x[i+1];
        a[i+1],x[i+1]=a_buffer,x_buffer;
        not_sorted=true;
      end
    end
  end
  return a,x
end

function gennoise(a::Array{Float64},level::Float64,number_spikes::Int64,bounds::Array{Float64,1})
  min_amp=a[argmin(a)];

  x_noise=bounds[1]+(bounds[2]-bounds[1])*rand(number_spikes);
  a_noise=level*min_amp*rand(number_spikes);

  return a_noise,x_noise
end

function addnoise(a::Array{Float64},x::Array{Float64},a_noise::Array{Float64},x_noise::Array{Float64})
  return vcat(a,a_noise),vcat(x,x_noise);
end

function plotSpikes(u::Array{Float64,1})
  N=convert(Integer,length(u)/2);
  a,x=u[1:N],u[N+1:2N];

  number_points=100;
  barU,posBarU=zeros(N,number_points),zeros(N,number_points);

  for i in 1:N
      barU[i,:]=collect(range(0,stop=a[i],length=number_points));
      posBarU[i,:]=x[i]*ones(number_points);
  end

  figure(figsize=(4,3))
  for i in 1:N
      plot(reshape(posBarU[i,:],number_points),reshape(barU[i,:],number_points),"--",color="red",linewidth=2.);
      plot(x,a,".",color="red",markersize=10.);
  end
  axis([0.0,x[N]+1.0,0.0,maximum(a)+.2]);
  show()
end
function plotSpikes(u0::Array{Float64,1},u::Array{Float64,1},op::operator;save::Bool=false,show_obser=false,show_obser_est=false,titl::String="")
  N0=convert(Integer,length(u0)/2);
  N=convert(Integer,length(u)/2);
  a,b=op.bounds[1],op.bounds[2];
  ya,yb=min(0.0,u[argmin(u[1:N])],u0[argmin(u0[1:N0])]),max(0.0,u[argmax(u[1:N])],u0[argmax(u0[1:N0])]);

  number_points=100;
  barU0,barU=zeros(N0,number_points),zeros(N,number_points);
  posBarU0,posBarU=zeros(N0,number_points),zeros(N,number_points);
  s0=zeros(N0);
  s=zeros(N);

  for i in 1:N0
    s0[i]=sign(u0[i]);
    barU0[i,:]=range(0,stop=abs(u0[i]),length=number_points);
    posBarU0[i,:]=u0[i+N0]*ones(number_points);
  end
  for i in 1:N
    s[i]=sign(u[i]);
    barU[i,:]=range(0,stop=abs(u[i]),length=number_points);
    posBarU[i,:]=u[i+N]*ones(number_points);
  end

  #if display==false
  #  ioff()
  #end
  figure(figsize=(4,3))
  for i in 1:N0
    plot(reshape(posBarU0[i,:],number_points),s0[i]*reshape(barU0[i,:],number_points),"--",color="black",linewidth=1.);
  end
  for i in 1:N
    plot(reshape(posBarU[i,:],number_points),s[i]*reshape(barU[i,:],number_points),"--",color="red",linewidth=1.);
  end
  plot(u0[N0+1:2N0],u0[1:N0],".",color="black",markersize=7.,label=L"$m_{a_0,x_0}$");
  plot(u[N+1:2N],u[1:N],".",color="red",markersize=7.,label=L"$m_{a,x}$");
  plot(range(a,stop=b,length=1000),zeros(1000),color="black",linewidth=0.5);

  if show_obser || show_obser_est
    nbpointsgrid=blasso.nbpointsgrid(op);
    grid=collect(range(op.bounds[1],stop=op.bounds[2],length=nbpointsgrid));
  end
  if show_obser
    obser=Array{Float64}(undef,nbpointsgrid);
    for i in 1:nbpointsgrid
      obser[i]=op.ob(grid[i]);
    end
    ya,yb=min(ya,minimum(obser)),max(yb,maximum(obser));
    plot(grid,obser,color="black");
  end
  if show_obser_est
    obser_est=Array{Float64}(undef,nbpointsgrid);
    for i in 1:nbpointsgrid
      obser_est[i]=op.ob(grid[i],u[1:N],u[N+1:2N]);
    end
    ya,yb=min(ya,minimum(obser_est)),max(yb,maximum(obser_est));
    plot(grid,obser_est,color="red");
  end

  axis([a,b,ya-.2,yb+.2]);

  if op.ker==blasso.dirichlet
    titlestring=string(op.ker,", fc=",op.fc);
  elseif op.ker==blasso.gaussian2D
    titlestring=string(op.ker,", sigma=",op.sigma);
  elseif op.ker<:blasso.dLaplace || op.ker<:blasso.cLaplace
    titlestring=string(op.ker);
  else
    titlestring=string(op.ker);
  end

  #legend()

  #title(string(titlestring,titl));
  if save
    savefig(string("spikes.pdf"),dpi=100);
  end

  #if display==false
  #  ion()
  #  close("all")
  #end
end
function plotSpikes(u0::Array{Float64,1},u::Array{Vector{Float64},1},op::operator;save::Bool=false,show_obser=false,show_obser_est=false,titl::String="")
  N0=convert(Integer,length(u0)/2);
  a,b=op.bounds[1],op.bounds[2];
  ya,yb=min(0.0,u0[argmin(u0[1:N0])]),max(0.0,u0[argmax(u0[1:N0])]);
  number_points=100;

  c=Array{Vector{Float64}}(length(u));
  t=range(.1,stop=.9,length=length(u));
  for i in 1:length(u)
    c[i]=(1-t[i])*[1.0,0.0,0.0]+t[i]*[0.0,0.0,1.0];
  end


  figure(figsize=(4,3))
  for k in 1:length(u)
    N=convert(Integer,length(u[k])/2);
    ya,yb=min(ya,u[k][argmin(u[k][1:N])]),max(yb,u[k][argmax(u[k][1:N])]);

    barU0,barU=zeros(N0,number_points),zeros(N,number_points);
    posBarU0,posBarU=zeros(N0,number_points),zeros(N,number_points);
    s0=zeros(N0);
    s=zeros(N);

    for i in 1:N0
      s0[i]=sign(u0[i]);
      barU0[i,:]=collect(range(0,stop=abs(u0[i]),length=number_points));
      posBarU0[i,:]=u0[i+N0]*ones(number_points);
    end
    for i in 1:N
      s[i]=sign(u[k][i]);
      barU[i,:]=collect(range(0,stop=abs(u[k][i]),length=number_points));
      posBarU[i,:]=u[k][i+N]*ones(number_points);
    end

    #if display==false
    #  ioff()
    #end

    for i in 1:N0
      plot(reshape(posBarU0[i,:],number_points),s0[i]*reshape(barU0[i,:],number_points),"--",color="black",linewidth=3.);
    end
    for i in 1:N
      plot(reshape(posBarU[i,:],number_points),s[i]*reshape(barU[i,:],number_points),"--",color=c[k],linewidth=2.);
    end
    plot(u0[N0+1:2N0],u0[1:N0],".",color="black",markersize=12.);
    plot(u[k][N+1:2N],u[k][1:N],".",color=c[k],markersize=10.);
    plot(collect(range(a,stop=b,length=100)),zeros(100),color="black",linewidth=0.5);

  end
  axis([a,b,ya-.2,yb+.2]);

  if op.ker==blasso.dirichlet
    titlestring=string(op.ker,", fc=",op.fc);
  elseif op.ker==blasso.gaussian
    titlestring=string(op.ker,", sigma=",op.sigma);
  elseif op.ker<:blasso.dLaplace || op.ker<:blasso.cLaplace
    titlestring=string(op.ker);
  else
    titlestring=string(op.ker);
  end

    title(string(titlestring,titl));
    if save
      savefig(string("spikes.pdf"),dpi=100);
    end

  #if display==false
  #  ion()
  #  close("all")
  #end
end
function plotSpikes(u::Array{Float64,1},op::operator;iter::Integer=0,save::Bool=false,show_obser=false,show_obser_est=false)
  N=convert(Integer,length(u)/2);
  a,b=op.bounds[1],op.bounds[2];
  ya,yb=min(0.0,u[argmin(u[1:N])]),max(0.0,u[argmax(u[1:N])]);

  number_points=100;
  barU,posBarU=zeros(N,number_points),zeros(N,number_points);
  s=zeros(N);

  for i in 1:N
    s[i]=sign(u[i]);
    barU[i,:]=collect(range(0,stop=abs(u[i]),length=number_points));
    posBarU[i,:]=u[i+N]*ones(number_points);
  end

  #if display==false
  #  ioff()
  #end
  figure(figsize=(4,3))
  for i in 1:N
    plot(reshape(posBarU[i,:],number_points),s[i]*reshape(barU[i,:],number_points),"--",color="red",linewidth=2.);
  end
  plot(u[N+1:2N],u[1:N],".",color="red",markersize=10.);
  plot(collect(range(a,stop=b,length=1000)),zeros(1000),color="black",linewidth=0.5);

  if show_obser || show_obser_est
    nbpointsgrid=5*length(op.Phix);
    grid=collect(range(op.bounds[1],stop=op.bounds[2],length=nbpointsgrid));
  end
  if show_obser
    obser=Array{Float64}(undef,nbpointsgrid);
    for i in 1:nbpointsgrid
      obser[i]=op.ob(grid[i]);
    end
    ya,yb=min(ya,minimum(obser)),max(yb,maximum(obser));
    plot(grid,9*obser,color="black",lw=1.0);
  end
  if show_obser_est
    obser_est=Array{Float64}(undef,nbpointsgrid);
    for i in 1:nbpointsgrid
      obser_est[i]=op.ob(grid[i],u[1:N],u[N+1:2N]);
    end
    ya,yb=min(ya,minimum(obser_est)),max(yb,maximum(obser_est));
    plot(grid,obser_est,color="red");
  end

  axis([a,b,ya,yb+.2]);

  if iter!=0
    if iter < 10
      iter=string("0",iter);
    elseif 10<=iter<100
      iter=string(iter);
    else
      iter=string(99,"+");
    end

    if op.ker==blasso.dirichlet
      titlestring=string(op.ker,", fc=",op.fc,", Iter:",iter);
    elseif op.ker==blasso.gaussian
      titlestring=string(op.ker,", sigma=",op.sigma,", Iter:",iter);
    elseif op.ker<:blasso.dLaplace || op.ker<:blasso.cLaplace
      titlestring=string(op.ker,", Iter:",iter);
    else
      titlestring=string(op.ker);
    end
  else
    if op.ker==blasso.dirichlet
      titlestring=string(op.ker,", fc=",op.fc);
    elseif op.ker==blasso.gaussian
      titlestring=string(op.ker,", sigma=",op.sigma);
    elseif op.ker<:blasso.dLaplace || op.ker<:blasso.cLaplace
      titlestring=string(op.ker);
    else
      titlestring=string(op.ker);
    end
  end

  if save
    savefig(string("spikes",iter,".pdf"),dpi=100);
  end
end

function nbpointsgrid(op::blasso.operator)
  if op.dim==1
    a,b=op.bounds[1],op.bounds[2];
    if op.ker==blasso.dirichlet
      lSample=op.fc;
    elseif op.ker<:blasso.dLaplace || op.ker<:blasso.cLaplace
      lSample=11;#21 ok
    elseif op.ker==:gaussian
      a,b=op.bounds[1],op.bounds[2];
      lSample=convert(Integer,2*floor(1/op.sigma,0)+1);
    end
    n=[convert(Integer,5*(lSample-1)/2*max(convert(Integer,floor(abs(b[1]-a[1]))),1))];
  else
    a,b=op.bounds[1],op.bounds[2];
    if op.ker==blasso.gaussian2D
      coeff=.2;
      lSample=[coeff/op.sigmax,coeff/op.sigmay];
      n=[convert(Int64,round(5*lSample[i]*abs(b[i]-a[i]);digits=0)) for i in 1:op.dim];
    end
    if op.ker==blasso.gaussian2DLaplace
      coeff=.2;
      lSample=[coeff/op.sigmax,coeff/op.sigmay,11];
      n=[convert(Int64,round(5*lSample[i]*abs(b[i]-a[i]);digits=0)) for i in 1:op.dim];
    end
  end
  return n;
end

function gengrid(op::blasso.operator,nbpointsgrid::Array{Int64,1})
  b1,b2=op.bounds[1],op.bounds[2];

  grid=Array{Array{Float64,1}}(undef,prod(nbpointsgrid));
  g=Array{Array{Float64,1}}(undef,op.dim);
  for i in 1:op.dim
    g[i]=collect(range(b1[i],stop=b2[i],length=nbpointsgrid[i]));
  end
  if op.dim == 1
    for i in 1:length(g[1])
      grid[i]=[g[1][i]];
    end
  elseif op.dim == 2
    k=1;
    for i in 1:length(g[1])
      for j in 1:length(g[2])
        grid[k]=[g[1][i],g[2][j]];
        k+=1;
      end
    end
  elseif op.dim == 3
    l=1;
    for k in 1:length(g[3])
      for i in 1:length(g[1])
        for j in 1:length(g[2])
          grid[l]=[g[1][i],g[2][j],g[3][k]];
          l+=1;
        end
      end
    end
  end
  return grid
end

function pointsInDomain(op::blasso.operator)
  return op.bounds[1]+rand(op.dim).*(op.bounds[2]-op.bounds[1]),op.bounds[1]+rand(op.dim).*(op.bounds[2]-op.bounds[1])
end

function lengthMeasure(u::Array{Float64,1};d::Int64=1)
  return N=convert(Int64,length(u)/(1+d));
end

function decompAmpPos(u::Array{Float64,1};d::Int64=1)
    N=lengthMeasure(u,d=d);
    if d>1
      x=Array{Array{Float64,1}}(undef,N);
    else
      x=Array{Float64}(undef,N);
    end
    a=u[1:N];
    for i in 1:N
      if d>1
        x[i]=zeros(d);
        for j in 1:d
          x[i][j]=u[N+i+(j-1)*N];
        end
      else
        x[i]=u[i+N];
      end
    end

    return a,x
end

function poisson(lambda::Float64)
    # Generate a random variable following
    # Poisson distribution of parameter lambda
    # using Knuth method.
    L=e^(-lambda);
    k=0;p=1;
    while p>L
        p=p*rand();
        k+=1;
    end
    return maximum([k-1,0])
end

end # module
