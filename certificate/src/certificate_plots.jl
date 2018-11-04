function plotEtaL(u::Array{Float64,1},v::Array{Float64,1},op::blasso.operator,lambda::Real,show::Bool=true,save::Bool=true,iter::Integer=-1,positivity::Bool=true)
  etaL=computeEtaL(u,v,op,lambda);

  degenerate,degen_pos=testDegenCert(etaL,1e-3);
  if positivity
    degenerate=degen_pos;
  end

  if !show
    ioff()
  end
  figure(figsize=(4,3));
  if degenerate
    plot(v,etaL,"r",lw=1.0);
  else
    plot(v,etaL,"g",lw=1.0);
  end
  plot(collect(range(op.bounds[1],stop=op.bounds[2],length=200)),ones(200),"black",linestyle="--",lw=1.0)
  ax=gca();
  ax[:set_xlim]([op.bounds[1],op.bounds[2]]);

  if iter >=0
    if iter < 10
      iter=string("0",iter);
    elseif 10<=iter<100
      iter=string(iter);
    else
      iter=string(99,"+");
    end
  end

  if op.ker==blasso.dirichlet || op.ker==blasso.dirichletvec
    if iter>=0
      stringtitle=string(L"$\eta_{\lambda}$ for ",op.ker,L", $f_c$=",op.fc,", iter=",iter);
    else
      stringtitle=string(L"$\eta_{\lambda}$ for ",op.ker,L", $f_c$=",op.fc);
    end
  elseif op.ker<:blasso.dLaplace || op.ker<:blasso.cLaplace
    if iter>=0
      stringtitle=string(L"$\eta_{\lambda}$ for ",op.ker,", iter=",iter);
    else
      stringtitle=string(L"$\eta_{\lambda}$ for ",op.ker);
    end
  elseif op.ker==blasso.gaussian
    stringtitle=string(op.ker,", sigma=",op.sigma,", iter=",iter);
  end

  title(stringtitle);
  if save
    if iter>=0
      savefig(string("etaL",iter,".pdf"),dpi=100);
    else
      savefig(string("etaL.pdf"),dpi=100);
    end
  end
  if !show
    ion()
    close("all")
  end
end

function plotEtaV(etaV::Function,op::blasso.operator;kwargs...)
  key_kw=[kwargs[i][1] for i in 1:length(kwargs)];
  if :alpha in key_kw
      al=kwargs[find(key_kw.==:alpha)[1]][2];
  else
      al=.1;
  end
  if :elev in key_kw
      el=kwargs[find(key_kw.==:elev)[1]][2];
  else
      el=20.0;
  end
  if :azim in key_kw
      az=kwargs[find(key_kw.==:azim)[1]][2];
  else
      az=300.0;
  end
  if :save in key_kw
      save=kwargs[find(key_kw.==:save)[1]][2];
  else
      save=false;
  end
  if :fsize in key_kw
      fsize=kwargs[find(key_kw.==:fsize)[1]][2];
  else
      fsize=[6,6];
  end
  if :ngrid in key_kw
    ngrid=kwargs[find(key_kw.==:ngrid)][1][2];
  else
    ngrid=100;
  end
  if :xb in key_kw
    xb=kwargs[find(key_kw.==:xb)][1][2];
  else
    xb=[-Inf,Inf];
  end
  if :yb in key_kw
    yb=kwargs[find(key_kw.==:yb)][1][2];
  else
    yb=[-Inf,Inf];
  end
  if :zb in key_kw
    zb=kwargs[find(key_kw.==:zb)][1][2];
  else
    zb=[-Inf,Inf];
  end
  if :nlevels in key_kw
    nlevels=kwargs[find(key_kw.==:nlevels)][1][2];
  else
    nlevels=30;
  end
  if op.dim==3
    if :z in key_kw
      z=kwargs[find(key_kw.==:z)][1][2];
      u=range(maximum([op.bounds[1][1],xb[1]]),stop=minimum([op.bounds[2][1],xb[2]]),length=ngrid);
      v=range(maximum([op.bounds[1][2],yb[1]]),stop=minimum([op.bounds[2][2],yb[2]]),length=ngrid);
      xlab,ylab="x","y";
    elseif :y in key_kw
      y=kwargs[find(key_kw.==:y)][1][2];
      u=range(maximum([op.bounds[1][1],xb[1]]),stop=minimum([op.bounds[2][1],xb[2]]),length=ngrid);
      v=range(maximum([op.bounds[1][3],zb[1]]),stop=minimum([op.bounds[2][3],zb[2]]),length=ngrid);
      xlab,ylab="x","z";
    elseif :x in key_kw
      x=kwargs[find(key_kw.==:x)][1][2];
      u=range(maximum([op.bounds[1][2],yb[1]]),stop=minimum([op.bounds[2][2],yb[2]]),length=ngrid);
      v=range(maximum([op.bounds[1][3],zb[1]]),stop=minimum([op.bounds[2][3],zb[2]]),length=ngrid);
      xlab,ylab="y","z";
    else
      z=(op.bounds[1][3]+op.bounds[2][3])/2;
      u=range(maximum([op.bounds[1][1],xb[1]]),stop=minimum([op.bounds[2][1],xb[2]]),length=ngrid);
      v=range(maximum([op.bounds[1][2],yb[1]]),stop=minimum([op.bounds[2][2],yb[2]]),length=ngrid);
      xlab,ylab="x","y";
    end
  else
    u=range(maximum([op.bounds[1][1],xb[1]]),stop=minimum([op.bounds[2][1],xb[2]]),length=ngrid);
    v=range(maximum([op.bounds[1][2],yb[1]]),stop=minimum([op.bounds[2][2],yb[2]]),length=ngrid);
    xlab,ylab="x","y";
  end

  etaV_list=zeros(length(u),length(v));
  for i in 1:length(u)
    for j in 1:length(v)
      if op.dim==3
        if :x in key_kw
          etaV_list[j,i]=etaV([x,u[i],v[j]]); #i column -> x axis
        elseif :y in key_kw
          etaV_list[j,i]=etaV([u[i],y,v[j]]);
        else
          etaV_list[j,i]=etaV([u[i],v[j],z]);
        end
      else
        etaV_list[j,i]=etaV([u[i],v[j]]);
      end
    end
  end
  c=ColorMap("jet");
  fig=figure(figsize=(fsize[1],fsize[2]))
  U,V=toolbox.meshgrid(u,v);
  if false
    fig[:subplots_adjust](left=0, right=1, bottom=0, top=1)
    ax=Axes3D(fig)
    surf(U,V,etaV_list,cmap=c,alpha=al)
    ax[:view_init](elev=el, azim=az)
  else
    println(minimum(etaV_list),maximum(etaV_list))
    levels=collect(range(minimum(etaV_list),stop=maximum(etaV_list),length=nlevels));
    ax=gca();
    #ax[:set_aspect]("equal")
    sc=ax[:contourf](U,V,etaV_list,cmap=c,levels=levels,vmax=1.1,vmin=0.0);
    #surf(lme,lint,lfn_opt,cmap="coolwarm")
    #sc[:set_clim](minimum(etaV_list),maximum(etaV_list))
    sc[:set_clim](0.0,1.1)
    cb = colorbar(sc,ticks=[0.0,0.5, 0.99],fraction=0.05, pad=0.05,drawedges=false)
    cb[:ax][:set_yticklabels](["0", "0.5","1"])  # horizontal colorbar
  end
  xlabel(xlab);ylabel(ylab);
  if save
    savefig("etaV.pdf",dpi=100);
  end
  show()
end
