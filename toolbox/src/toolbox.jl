module toolbox

#using Convex,SCS
using LinearAlgebra
using Statistics

function Min(u::Array{Float64})
    m=u[1];
    arg=1;
    for i in 1:length(u)-1
        U=u[i+1];
        if m>U
            m=U;
            arg=i;
        end
    end
    return arg,m
end

function Max(u::Array{Float64})
    M=u[1];
    arg=1;
    for i in 1:length(u)-1
        U=u[i+1];
        if M<U
            M=U;
            arg=i+1;
        end
    end
    return arg,M
end

function scientificwriting(x::Float64)
    x=abs(x);
    b=trunc(log10(x));
    if b<0.0
        if exp((log10(x)-b)*log(10)) < 1.0
            b-=1.0;
            a=floor(10.0*exp((log10(x)-trunc(log10(x)))*log(10)),1);
        else
            a=floor(exp((log10(x)-round(log10(x)))*log(10)),1);
        end
        b=convert(Int,b);
    else
        if exp((log10(x)-b)*log(10)) > 1.0
            a=floor(exp((log10(x)-trunc(log10(x)))*log(10)),1);
        else
            b-=1;
            a=floor(10.0*exp((log10(x)-round(log10(x)))*log(10)),1);
        end
        b=convert(Int,b);
    end
    return string(a,"e",b)
end

function conditionnement(A::Array{Float64,2})
  lmax=eigmax(A);
  lmin=eigmin(A);

  return abs(lmax/lmin)
end

function deleteZeroAmpSpikes(u::Array{Float64,1},show_warning=false;d::Int64=1)
    N=lengthMeasure(u,d=d);
    a,X=decompAmpPos(u,d=d);
    I=Array{Int64}(undef,0);
    j=1;
    for i in 1:N
        if a[j]==0.0
            if show_warning
              println("### Delete zero Amp spikes ###");
            end
            deleteat!(X,j);
            deleteat!(a,j);
            j-=1;
        end
        j+=1;
    end
    return recompAmpPos(a,X,d=d)
end

function pruneSpikes(u::Array{Float64,1};d::Int64=1)
    a,x=toolbox.decompAmpPos(u,d=d);
    new_a,new_x=Array{Float64}(undef,0),Array{typeof(x[1])}(undef,0);
    list_ind=Array{Int64}(undef,0);
    sum_a=0.0;
    N=length(a);
    pruned=false;

    while length(x)>0
        xt=[x[i]-x[1] for i in 1:length(x)];
        for i in 1:length(xt)
            if norm(xt[i])<1e-6
                sum_a+=a[i];
                append!(list_ind,i);
                if i>1
                  pruned=true;
                end
            end
        end
        append!(new_a,sum_a);
        append!(new_x,[mean([x[i] for i in list_ind])]);
        deleteat!(x,list_ind);
        deleteat!(a,list_ind);
        sum_a=0.0;
        list_ind=Array{Int64}(undef,0);
    end

    return recompAmpPos(new_a,new_x,d=d),pruned
end

function lengthMeasure(u::Array{Float64,1};d::Int64=1)
  return convert(Int64,length(u)/(1+d));
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

function recompAmpPos(a::Array{Float64,1},x::Array{Float64,1};d::Int64=1)
  return vcat(a,x)
end

function recompAmpPos(a::Array{Float64,1},x::Array{Array{Float64,1},1};d::Int64=1)
  N=length(a);
  u=Array{Float64}(undef,length(a)*(d+1));
  for i in 1:N
    u[i]=a[i];
  end
  for i in 1:d
    for j in 1:N
      u[N+j+(i-1)*N]=x[j][i];
    end
  end

  return u
end

function projgradient(g::Array{Float64,1},ind::Array{Int64,1})
  gproj=zeros(length(g));
  for i in 1:length(g)
      if i in ind
          gproj[i]=g[i];
      end
  end
  return gproj
end

function partition(N::Int64,n::Int64)
    c=collect(2:N);
    cpartition=Array{Vector{Int64}}(n);

    r=(N-1)%n;
    q=convert(Int64,((N-1)-r)/n);
    for i in 1:n
        if i==1
            if r>0
                cpartition[1]=2+collect((0:q));
            else
                cpartition[1]=2+collect((0:q-1));
            end
        else
            if i<=r
                cpartition[i]=cpartition[i-1][end]+1+(0:q);
            else
                cpartition[i]=cpartition[i-1][end]+1+(0:q-1);
            end
        end
    end

    return cpartition
end

function getMeanInterval(x::Array{Float64,1})
  m=mean(x);
  delta=x[end]-m;
  return m,delta
end

function histogram(x::Array{Float64};nbins=10)
  x=sort(x);
  bins=collect(linspace(x[1],x[end],nbins+1));
  h=zeros(nbins);
  k=1;
  i=1;
  while i <=length(x)
    if x[i]<=bins[k+1]
      h[k]+=1;
      i+=1;
    else
      k+=1;
    end
  end
  return bins,h
end

function stayInDomain(x::Array{Float64,1},x_low::Array{Float64,1},x_up::Array{Float64,1})
    outofbounds=false;
    for i in 1:length(x)
        if x[i]<x_low[i]
            x[i]=x_low[i];
            outofbounds=true;
        end
        if x[i]>x_up[i]
            x[i]=x_up[i];
            outofbounds=true;
        end
    end
    return x,outofbounds
end
function stayInDomain(x::Array{Float64,1},x_low::Float64,x_up::Float64)
    outofbounds=false;
    if x[1]<x_low
        x[1]=x_low;
        outofbounds=true;
    end
    if x[1]>x_up
        x[1]=x_up;
        outofbounds=true;
    end
    return x,outofbounds
end

function isOnFrontier(x::Array{Float64,1},x_low::Array{Float64,1},x_up::Array{Float64,1})
  onFrontier=true;
  for i in 1:length(x)
    if x[i]!=x_low[i] && x[i]!=x_up[i]
      onFrontier=false;
    end
  end
  return onFrontier
end
function isOnFrontier(x::Array{Float64,1},x_low::Float64,x_up::Float64)
  onFrontier=true;
  if x[1]!=x_low && x[1]!=x_up
    onFrontier=false;
  end
  return onFrontier
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

function selectInd(n::Array{Int64,1},N::Int64,d::Int64)
    K=length(n);
    Ii=zeros(Int64,(1+d)*K);
    k=1;
    for j in 1:1+d
        for i in 1:K
            Ii[k]=n[i]+(j-1)*N;
            k+=1;
        end
    end
    return Ii
end

function randOnList(list::Array{Int64,1},K::Int64)
    n=zeros(Int64,K);
    for i in 1:K
        j=rand(list);
        n[i]=j;
        deleteat!(list,find(list.==j));
    end
    return sort(n)
end

function clusterPoints(XY::Array{Array{Float64,1},1},d::Float64)
    cluster=Array{Array{Int64,1}}(length(XY));
    Ii=collect(1:length(XY));

    for i in 1:length(XY)
        cluster[i]=[i];
        for j in setdiff(Ii,[i])
            if norm(XY[i]-XY[j])<d
                append!(cluster[i],[j])
            end
        end
    end

    i=1;fusion=false;
    while i < length(cluster)
        for j in setdiff(collect(1:length(cluster)),[i])
            if length(intersect(cluster[i],cluster[j]))>0
                append!(cluster[i],cluster[j]);
                deleteat!(cluster,j);
                fusion=true;
                break;
            end
        end
        if !fusion
            i+=1;
        else
            fusion=false;
        end
    end

    c=Array{Int64}(undef,0);
    for i in 1:length(cluster)
        c=[cluster[i][1]];
        for j in 1:length(cluster[i])
            if !(cluster[i][j] in c)
                append!(c,[cluster[i][j]]);
            end
        end
        cluster[i]=sort(c);
    end
    return cluster
end


end # module
