module sfw

using blasso,toolbox,certificate,PyPlot

#####################################################################################

type sfw_options{}
  positivity::Bool
  subblock_descent::Bool

  max_mainIter::Integer
  max_luIter::Integer
  max_subblock_iter::Integer
  dist_subblock::Float64

  x_tol::Float64
  a_tol::Float64
  u_tol::Float64
  f_tol::Float64
  g_tol::Float64
  eta_tol::Float64

  show_mainIter::Bool
  show_newPos::Bool
  keep_trace::Bool
  store_correl::Bool
  show_trace::Bool
  show_warning::Bool
  show_success::Bool

  show_time_newPos::Bool
  show_time_localUpdate::Bool
end

function sfw_options(;
  positivity::Bool=true,
  subblock_descent::Bool=false,
  max_mainIter::Integer=20,
  max_luIter::Integer=10000,
  max_subblock_iter::Integer=100,
  dist_subblock::Float64=0.0,
  x_tol::Real=1e-4,
  a_tol::Real=1e-10,
  u_tol::Real=1e-14,
  f_tol::Real=1e-15,
  g_tol::Real=1e-11,
  eta_tol::Real=1e-2,
  show_mainIter::Bool=true,
  show_newPos::Bool=true,
  keep_trace::Bool=false,
  store_correl::Bool=false,
  show_trace::Bool=false,
  show_warning::Bool=false,
  show_success::Bool=true,
  show_time_newPos::Bool=false,
  show_time_localUpdate::Bool=false)

  if show_trace
    keep_trace,show_mainIter=true,true;
  end
  sfw_options(positivity,subblock_descent,max_mainIter,max_luIter,max_subblock_iter,dist_subblock,x_tol,a_tol,u_tol,f_tol,g_tol,eta_tol,show_mainIter,show_newPos,keep_trace,store_correl,show_trace,show_warning,show_success,show_time_newPos,show_time_localUpdate)
end

type sfw_result
  newPos::Array{Array{Float64,1},1}
  f::Real
  f_previous::Real
  f_bfgs::Array{Float64,1}
  f_mainIter::Array{Float64,1}
  f_subblock_iter::Array{Vector{Float64},1}
  f_list::Array{Float64,1}
  u::Array{Float64,1}
  u_lasso::Array{Vector{Float64},1}
  u_bfgs::Array{Vector{Float64},1}
  u_mainIter::Array{Vector{Float64},1}
  u_list::Array{Vector{Float64},1}
  u_previous::Array{Float64,1}
  g::Array{Float64,1}
  g_bfgs::Array{Vector{Float64},1}
  g_mainIter::Array{Vector{Float64},1}
  g_list::Array{Vector{Float64},1}
  g_previous::Array{Float64,1}
  normG::Array{Float64,1}
  correl::Array{Vector{Float64},1}

  luIter::Array{Int64,1}
  subblock_iter::Array{Int64,1}
  max_etaL::Array{Float64,1}

  f_converged::Array{Bool,1}
  u_converged::Array{Bool,1}
  g_converged::Array{Bool,1}
  max_luIter_hits::Array{Bool,1}
  interpolation::Array{Float64,1}
  derivatives::Array{Float64,1}

  warning_newPos::Array{Int64,1}
  warning_step::Array{Int64,1}

  blasso_converged::Bool
end

function remove_result!(r::sfw_result)
  N=length(r.newPos);
  if N>=2
    deleteat!(r.u_mainIter,N);
    deleteat!(r.f_mainIter,N);
    deleteat!(r.g_mainIter,N);
    r.u,r.u_previous=r.u_mainIter[end],r.u_mainIter[end];
  end
end

function update_result!(r::sfw_result;kwargs...)
  for i in 1:length(kwargs)
    if !( kwargs[i][1] in [:f,:f_previous,:u,:u_previous,:g,:g_previous,:blasso_converged] )
      if length(getfield(r,kwargs[i][1]))==0
        setfield!(r,kwargs[i][1],[kwargs[i][2]]);
      elseif kwargs[i][1] in [:f,:f_previous]
        copy!(getfield(r,kwargs[i][1]),kwargs[i][2]);
      else
        append!(getfield(r,kwargs[i][1]),[kwargs[i][2]]);
      end
    else
      setfield!(r,kwargs[i][1],kwargs[i][2]);
    end
  end # for
end

function show_result(r::sfw_result,o::sfw.sfw_options)
  if r.blasso_converged
    println("###############################################");
    println("####### Solution of BLASSO in ", length(r.newPos),"/",o.max_mainIter," iter! #######");
    println("###############################################");
  else
    println("#### No solution of BLASSO in ", length(r.newPos)," iter! ####");
  end
  println(" - u         : ",r.u);
  println(" - fobj      : ",r.f_mainIter[end]);
  println(" - g         : ",r.g_mainIter[end]);
  println(" - max(etaL) : ",r.max_etaL);
  println(" - f_converged : (tol ", o.f_tol,") : ",r.f_converged[1:end]);
  println(" - u_converged : (tol ", o.u_tol,") : ",r.u_converged[:]);
  println(" - g_converged :  (tol ", o.g_tol,") : ",r.g_converged[:]);
  println(" - max lu iter hits : (max ", o.max_luIter,") : ",r.max_luIter_hits[:]);
  println(" - Nb iter local update          : ",r.luIter);
  println(" - Warning newPos : ",r.warning_newPos);
  println(" - Warning step   : ",r.warning_step);
end

function init_result(op::blasso.operator)
  u_start=vcat([0.0],op.bounds[1]+(op.bounds[2]-op.bounds[1])/2);
  return sfw_result(Array{Float64}(0),0.0,0.0,Array{Float64}(0),Array{Float64}(0),Array{Vector{Float64}}(0),Array{Float64}(0),u_start,Array{Vector{Float64}}(0),Array{Vector{Float64}}(0),
                     Array{Vector{Float64}}(0),Array{Vector{Float64}}(0),Array{Float64}(0),Array{Float64}(0),Array{Vector{Float64}}(0),Array{Vector{Float64}}(0),
                     Array{Vector{Float64}}(0),Array{Float64}(0),Array{Float64}(0),Array{Vector{Float64}}(0),Array{Int64}(0),Array{Int64}(0),Array{Float64}(0),
                     Array{Bool}(0),Array{Bool}(0),Array{Bool}(0),Array{Bool}(0),Array{Float64}(0),Array{Float64}(0),Array{Int64}(0),Array{Int64}(0),false);
end


function sfw4blasso(fobj::blasso.fobj,kernel::blasso.kernel,op::blasso.operator,o::sfw_options)
  # INIT variable results storing variable
  result=init_result(op);
  # STORE initial energy and gradient.
  if o.keep_trace
    update_result!(result;f_list=fobj.f(result.u),f_mainIter=fobj.f(result.u),g_mainIter=fobj.g(result.u),g_list=fobj.g(result.u),normG=norm(fobj.g(result.u)),f_bfgs=fobj.f(result.u),g_bfgs=fobj.g(result.u));
  else
    update_result!(result;f_mainIter=fobj.f(result.u),g_mainIter=fobj.g(result.u),f_bfgs=fobj.f(result.u),g_bfgs=fobj.g(result.u));
  end
  #

  for i in 1:o.max_mainIter
    ## - Compute the Next Position - ##
    t=time();
    correl_min=computeNextPosition!(result,kernel,op,o);
    dt=time()-t;

    # Check non-degeneracy
    if o.positivity
      if checkConvergence(-correl_min/fobj.lambda,result,o);
        break;
      end
    elseif !o.positivity
      if checkConvergence(abs(correl_min)/fobj.lambda,fobj,result,o);
        break;
      end
    end

    if o.show_mainIter
      println("---------------------------------------");
      println("Iteration no ",i," :");
    end
    # Show newPosition
    if o.show_newPos
      println("New Position : ",result.newPos[end]);
      if o.show_time_newPos
        println("--  Time in computeNextPosition : ",round(dt,5)," s");
      end
    end

    ## - Local Descent - ##
    t=time();
    localUpdate!(result,op,fobj,o);
    if o.show_time_localUpdate
      println("--  Time in localUpdate : ",round(time()-t,5)," s");
    end

    ## - Check Stopping Criteria - ##
    if checkStop(result,op,o)
      break;
    end
  end # end for

  # Rescaling amplitudes if phi discrete laplace and close to experiments model
  if :rl_model in fieldnames(op)
    if op.rl_model
      N=toolbox.lengthMeasure(result.u,d=op.dim);
      a,x=toolbox.decompAmpPos(result.u,d=op.dim);
      if N>0
        if op.dim==1
          for i in 1:N
            result.u[i]=result.u[i]/op.normPhi(result.u[i+N]);
          end
        else
          for i in 1:N
            result.u[i]=result.u[i]/op.normPhi(x[i][end]);
          end
        end
      end
    end
  end

  return result;
end

function checkConvergence(max_etaL::Float64,r::sfw_result,o::sfw_options)
  cv=false;

  if abs(max_etaL-1)<o.eta_tol
    if o.show_success
      println("\nSUCCEED! In ",length(r.newPos)-1," iterations.");
    end
    update_result!(r,blasso_converged=true);
    deleteat!(r.newPos,length(r.newPos));
    cv=true;
  elseif max_etaL<=1-2*1e-2
    if o.show_success
      println("\nFAILED! (max(etaL)<1)");
    end
    update_result!(r,blasso_converged=false);
    deleteat!(r.newPos,length(r.newPos));
    cv=true;
  end
  update_result!(r;max_etaL=max_etaL);

  return cv
end

function checkStop(result::sfw_result,op::blasso.operator,o::sfw_options)
  stop=false;

  i=length(result.newPos);

  if isnan(result.f)
    update_result!(result,blasso_converged=false);
    remove_result!(result);
    if o.show_success
      println("\nFAILED! (NaN)");
    end
    stop=true;
  end

  u_pruned,pruned=toolbox.pruneSpikes(result.u,d=op.dim);
  if o.keep_trace
    update_result!(result;u=u_pruned,u_list=u_pruned);
  else
    update_result!(result;u=u_pruned);
  end

  if i>=2
    if op.dim==1
      cond_bound=!(result.newPos[end][1] in [op.bounds[1][1],op.bounds[2][1]]);
    else
      cond_bound=true;
    end
    if norm(result.newPos[end]-result.newPos[end-1])<1e-3 && cond_bound
      if o.show_success
        println("\nFAILED! (Same newPos)");
      end
      update_result!(result,blasso_converged=false);
      remove_result!(result);
      stop=true;
    end
  end
  if i==o.max_mainIter
    update_result!(result,blasso_converged=false);
    if o.show_success
      println("\nFAILED! (max iter)");
    end
    stop=true;
  end

  return stop
end

function computeNextPosition!(r::sfw_result,kernel::blasso.kernel,op::blasso.operator,o::sfw_options)
  # Frank-Wolfe Step.
  # Minimization of -etaL if positivity constraints or -|etaL| otherwise.

  u=copy(r.u);
  max_iter=500;
  iter=0;

  argmin,argmin_previous=zeros(op.dim),zeros(op.dim);

  N=blasso.lengthMeasure(u,d=op.dim);
  ##
  Phiu=blasso.computePhiu(u,op);
  correl(x::Array{Float64,1})=op.correl(x,Phiu);
  d1correl(x::Array{Float64,1})=op.d1correl(x,Phiu);
  d2correl(x::Array{Float64,1})=op.d2correl(x,Phiu);

  #
  if o.store_correl
    Correl=zeros(100);
    gridcoarse=collect(linspace(kernel.grid[1][1],kernel.grid[1][end],100));
    for i in 1:100
      if o.positivity
        Correl[i]=correl([gridcoarse[i]]);
      else
        Correl[i]=-abs(correl([gridcoarse[i]]));
      end
    end
    update_result!(r,correl=Correl);
  end
  #

  # Minimizing on a grid.
  t=time();
  argmin_previous,correl_min=blasso.minCorrelOnGrid(Phiu,kernel,op,o.positivity);
  argmin_grid=copy(argmin_previous);
  Argmin=copy(argmin_grid);
  if o.show_time_newPos
    println("Time minimization on grid : ",round(time()-t,5))
  end

  x_low,x_up=blasso.setbounds(op,o.positivity,false);
  if !toolbox.isOnFrontier(argmin_previous,x_low,x_up)
    h_previous=d2correl(argmin_previous);
    if isposdef(h_previous) || !o.positivity
      g_previous=d1correl(argmin_previous);
      cv=false;
      # Newton Descent.
      while !cv
        d=-h_previous\g_previous;
        argmin=argmin_previous+d;

        argmin,outofbounds=toolbox.stayInDomain(argmin,x_low,x_up);

        if norm(argmin-argmin_previous)<o.x_tol || iter>=max_iter || outofbounds
          cv=true;
          if iter>=max_iter
            update_result!(r,warning_newPos=1);
            if o.show_warning
              println("# Max_iter in computeNextPosition, dx=",norm(argmin-argmin_previous)," #");
            end
          else
            update_result!(r,warning_newPos=0);
          end
          if outofbounds
            if o.positivity
              if correl(argmin)>correl(argmin_previous)
                argmin=copy(argmin_previous);
              end
            else
              if -abs(correl(argmin))>-abs(correl(argmin_previous))
                argmin=copy(argmin_previous);
              end
            end
          end # end outofbounds
          if o.positivity
            if correl(argmin)<correl_min
              correl_min=correl(argmin);
              Argmin=copy(argmin);
            end
          else
            if -abs(correl(argmin))<correl_min
              correl_min=-abs(correl(argmin));
              Argmin=copy(argmin);
            end
          end
          argmin_previous=copy(argmin);
        else
          if o.positivity
            if correl(argmin)<correl_min
              correl_min=correl(argmin);
              Argmin=copy(argmin);
            end
          else
            if -abs(correl(argmin))<correl_min
              correl_min=-abs(correl(argmin));
              Argmin=copy(argmin);
            end
          end
          argmin_previous=copy(argmin);
          g_previous=d1correl(argmin);
          h_previous=d2correl(argmin);
          iter+=1;
        end  # if stopping conditions
      end # while
      update_result!(r,newPos=Argmin);
    else # if h_previous not non-negative
      update_result!(r,newPos=Argmin);
    end
  else # argmin_previous on frontier of domain -> keep minimum on the grid and no Newton descent.
    update_result!(r,newPos=Argmin);
  end
  return correl(Argmin)
end

function localUpdate!(r::sfw_result,op::blasso.operator,fobj::blasso.fobj,o::sfw_options)
  # Compute new set of amplitudes for new sets of positions.
  u_previous=copy(computeNewAmplitudes(r,op,fobj.lambda,o.a_tol,o.positivity,"FISTA"));
  g_previous=copy(fobj.g(u_previous)); # Compute gradient at u_previous.
  f_previous=fobj.f(u_previous); # Compute objective function at u_previous.
  if o.show_mainIter
    println("### Before BFGS ###")
    println("Amp/Pos : ",u_previous);
    println("Energy : ",f_previous);
  end

  # STORE before local update
  if o.keep_trace
    update_result!(r;f_list=f_previous,
                     g_list=g_previous,
                     u_lasso=u_previous,
                     u_list=u_previous,
                     f_previous=f_previous,
                     g_previous=g_previous,
                     u_previous=u_previous,
                     normG=norm(g_previous));
  else
    update_result!(r;u_lasso=u_previous,
                     f_previous=f_previous,
                     g_previous=g_previous,
                     u_previous=u_previous);
  end

  # SHOW trace
  if o.show_trace
    println("Before BFGS : Amplitudes/Positions ---- Energy ---- Gradient ----");
    println(u_previous, " ## ", r.f_list[end], " ## ", r.g_list[end]);
  end

  if o.subblock_descent
    if length(r.newPos)>1
      o.subblock_descent=false;
      localDescent!(u_previous,r,op,fobj,o);
      o.subblock_descent=true;

      if r.warning_step[end]>0 || !(r.f_converged[end] || r.g_converged[end] || r.u_converged[end])
        if o.dist_subblock==0.0
          K=minimum([3,length(r.newPos)]);
          localDescent!(r.u,r,op,fobj,o,K,o.max_subblock_iter);
        else
          localDescent!(r.u,r,op,fobj,o,o.dist_subblock);
          localDescent!(r.u,r,op,fobj,o,3*o.dist_subblock);
          etaL,d1etaL,d2etaL=certificate.computeEtaL(r.u,op,fobj.lambda);
          a,x=toolbox.decompAmpPos(r.u,d=op.dim);
          m=1;
          #while (norm(norm.(d1etaL.(x)))>1e-4 || norm(sign(a)-etaL.(x))>1e-4) && m < 20
          while norm(sign(a)-etaL.(x))>1e-4 && m < 20
            localDescent!(r.u,r,op,fobj,o,3*o.dist_subblock);
            etaL,d1etaL,d2etaL=certificate.computeEtaL(r.u,op,fobj.lambda);
            a,x=toolbox.decompAmpPos(r.u,d=op.dim);
            m+=1;
          end
        end
        if o.show_mainIter
          println("### After Zone Descent ###");
          println("Amp/Pos : ",r.u);
          println("Energy : ",r.f);
        end
      end # end if
    else
      o.subblock_descent=false;
      localDescent!(u_previous,r,op,fobj,o);
      o.subblock_descent=true;
    end
  else
    localDescent!(u_previous,r,op,fobj,o);
  end

  u_cleaned=toolbox.deleteZeroAmpSpikes(r.u,o.show_warning,d=op.dim);

  etaL,d1etaL,d2etaL=certificate.computeEtaL(u_cleaned,op,fobj.lambda);
  a,x=toolbox.decompAmpPos(u_cleaned,d=op.dim);
  if op.dim==1
    update_result!(r;interpolation=norm(sign(a)-etaL(x)),derivatives=norm(norm.(d1etaL(x))));
  else
    update_result!(r;interpolation=norm(sign(a)-etaL.(x)),derivatives=norm(norm.(d1etaL.(x))));
  end

  if o.keep_trace
    update_result!(r;u=u_cleaned,u_list=u_cleaned,u_mainIter=u_cleaned,f_mainIter=r.f,g_mainIter=r.g);
  else
    update_result!(r;u=u_cleaned,u_mainIter=u_cleaned,f_mainIter=r.f,g_mainIter=r.g);
  end
end

function localDescent!(u_previous::Array{Float64,1},result::sfw.sfw_result,op::blasso.operator,fobj::blasso.fobj,options::sfw.sfw_options,K::Int64,L::Int64)
  # sub-block local descent
  # random sub-block form composed of K spikes.
  # bfgs on each sub-block.
  a_low,a_up,x_low,x_up=blasso.setbounds(op,options.positivity,true);
  N=blasso.lengthMeasure(u_previous,d=op.dim);
  f_previous=result.f;
  update_result!(result;f_subblock_iter=[f_previous]);

  cv,ii=false,1;
  while !cv
        n=toolbox.randOnList(collect(1:N),K);
        lw=zeros(K*(op.dim+1));
        up=zeros(K*(op.dim+1));
        lw[1:K]=a_low*ones(K);
        up[1:K]=a_up*ones(K);
        for i in 1:op.dim
            lw[1+i*K:(i+1)*K]=x_low[i]*ones(K);
            up[1+i*K:(i+1)*K]=x_up[i]*ones(K);
        end
        Ii=toolbox.selectInd(n,N,op.dim);
        function F(u::Array{Float64,1})
            U=copy(u_previous);
            U[Ii]=copy(u);
            return fobj.f(U)
        end
        function G(u::Array{Float64,1})
            U=copy(u_previous);
            U[Ii]=copy(u);
            return fobj.g(U,n)
        end
        function H(u::Array{Float64,1})
            return eye((1+op.dim)*K)
        end

        U_previous=copy(u_previous[Ii]);
        sfw.bfgs_b!(F,G,H,U_previous,lw,up,result,options,op.ker);
        u_previous[Ii]=result.u;

        f=result.f;
        if ii>=L
            cv=true;
        end
        if abs(f-f_previous)<options.f_tol && ii>2*convert(Int64,round(length(u_previous)/K))
            cv=true;
        end
        f_previous=f;
        ii+=1;
    end
    result.u=copy(u_previous);
    result.u_previous=copy(u_previous);
end
function localDescent!(u_previous::Array{Float64,1},result::sfw.sfw_result,op::blasso.operator,fobj::blasso.fobj,options::sfw.sfw_options,dist::Float64)
    # sub-block local descent
    # spikes seperated by less than dist are put together to form a block
    # bfgs on each sub-block.
    a_low,a_up,x_low,x_up=blasso.setbounds(op,options.positivity,true);
    N=blasso.lengthMeasure(u_previous,d=op.dim);
    a_previous,x_previous=toolbox.decompAmpPos(u_previous,d=op.dim);
    XY_previous=[[x_previous[i][1],x_previous[i][2]] for i in 1:N];
    f_previous=result.f;
    update_result!(result;f_subblockIter=[f_previous]);

    cv,ii=false,1;
    cluster=toolbox.clusterPoints(XY_previous,dist);
    for k in 1:length(cluster)
      n=cluster[k];
      K=length(n);

      lw=zeros(K*(op.dim+1));
      up=zeros(K*(op.dim+1));
      lw[1:K]=a_low*ones(K);
      up[1:K]=a_up*ones(K);
      for i in 1:op.dim
        lw[1+i*K:(i+1)*K]=x_low[i]*ones(K);
        up[1+i*K:(i+1)*K]=x_up[i]*ones(K);
      end
      Ii=toolbox.selectInd(n,N,op.dim);
      function F(u::Array{Float64,1})
        U=copy(u_previous);
        U[Ii]=copy(u);
        return fobj.f(U)
      end
      function G(u::Array{Float64,1})
        U=copy(u_previous);
        U[Ii]=copy(u);
        return fobj.g(U,n)
      end
      function H(u::Array{Float64,1})
        return eye((1+op.dim)*K)
      end

      U_previous=copy(u_previous[Ii]);
      sfw.bfgs_b!(F,G,H,U_previous,lw,up,result,options,op.ker);
      u_previous[Ii]=result.u;
    end
    result.u=copy(u_previous);
    result.u_previous=copy(u_previous);
end
function localDescent!(u_previous::Array{Float64,1},result::sfw.sfw_result,op::blasso.operator,fobj::blasso.fobj,options::sfw.sfw_options)
    # Standard local descent
    # Moves all spikes amplitudes and positions.
    a_low,a_up,x_low,x_up=blasso.setbounds(op,options.positivity,true);
    N=blasso.lengthMeasure(u_previous,d=op.dim);

    lw=zeros(N*(op.dim+1));
    up=zeros(N*(op.dim+1));
    lw[1:N]=a_low*ones(N);
    up[1:N]=a_up*ones(N);
    for i in 1:op.dim
        lw[1+i*N:(i+1)*N]=x_low[i]*ones(N);
        up[1+i*N:(i+1)*N]=x_up[i]*ones(N);
    end
    sfw.bfgs_b!(fobj.f,fobj.g,fobj.h,u_previous,lw,up,result,options,op.ker);
end

function computeCauchyPoint(x0::Array{Float64,1},g::Array{Float64,1},B::Array{Float64,2},lw::Array{Float64,1},up::Array{Float64,1})
  M=length(up);
  t,d=zeros(M),zeros(M);
  x=copy(x0);
  @inbounds for i in 1:M
    if g[i]<0.0
      if up[i]==Inf
        t[i]=Inf;
      else
        t[i]=(x[i]-up[i])/g[i];
      end
    elseif g[i]>0.0
      if lw[i]==-Inf
        t[i]=Inf;
      else
        t[i]=(x[i]-lw[i])/g[i];
      end
    else
      t[i]=Inf;
    end #if
    if t[i]>0.0
      d[i]=-g[i];
    end
  end #for
  ts=sort(t);

  F=[i for i in 1:M if t[i]>0];
  fp=-vecdot(d,d);
  fs=vecdot(d,B*d);
  dtm=-fp/fs;
  to=0.0;
  T=minimum(t[F]);
  b=F[indmin(t[F])];
  dt=T-to;

  while dtm>=dt
    if d[b]>0.0
      bo=up[b];
    else
      bo=lw[b];
    end
    d[b]=0.0;
    #z[b]=z[b]+bo-x[b];
    x[b]=bo;

    fp=-vecdot(d,d);
    fs=vecdot(d,B*d);

    dtm=-fp/fs;

    to=T;
    deleteat!(F,indmin(t[F]));
    T=minimum(t[F]);
    b=F[indmin(t[F])];
    dt=T-to;
  end #while

  dtm=maximum(vcat(dtm,0.0));
  to=to+dtm;

  @inbounds @simd for i in 1:M
    if t[i]>=T
      x[i]=x[i]+to*d[i];
    end
  end

  return x,F
end #function

function subspaceMin(xc::Array{Float64,1},Ac::Array{Int64,1},x::Array{Float64,1},g::Array{Float64,1},B::Array{Float64,2},lw::Array{Float64,1},up::Array{Float64,1})
  Z=zeros(length(x),length(Ac));
  @inbounds @simd for i in 1:length(Ac)
    Z[Ac[i],i]=1.0;
  end
  b=-Z'*g-Z'*B*(xc-x);
  A=Z'*B*Z;

  y=zeros(length(Ac));
  r=b-A*y;
  p=copy(r);

  xb=xc+Z*y;
  xbold=copy(xb);

  eps=minimum(vcat(1e-8,sqrt(norm(r))))*norm(r);
  if norm(r)<=eps
    convergence=true;
  else
    convergence=false;
  end
  i=1;
  while !convergence
    Ap=A*p;
    alpha=vecdot(r,r)/vecdot(p,Ap);
    y=y+alpha*p;
    rold=copy(r);

    r=r-alpha*Ap;

    xb=xc+Z*y;

    if !checkInBounds(xb,lw,up)
      #println("### Going out of bounds ! ###")
      break;
    elseif i>10
      break;
    elseif norm(r)<=eps
      #println("### Hit tol ###",norm(r))
      xbold=copy(xb);
      break;
    end

    beta=vecdot(r,r)/vecdot(rold,rold);
    p=r+beta*p;

    xbold=copy(xb);
    i+=1;
  end #while

  return(xbold)
end #function

function checkInBounds(x::Array{Float64,1},lw::Array{Float64,1},up::Array{Float64,1})
  N=length(x);
  inB=true;
  @inbounds for i in 1:N
    if x[i]<lw[i] || x[i]>up[i]
      inB=false;
      break;
    end
  end

  return inB
end #function

function lineSearch(d::Array{Float64,1},x0::Array{Float64,1},fx0::Float64,gx0::Array{Float64,1},f::Function,g::Function,lw::Array{Float64,1},up::Array{Float64,1},o::sfw_options,typekernel::DataType=blasso.kernel)
  al,be=.2,1.2;
  MAX_STEP=20;
  if typekernel<:blasso.cLaplace || typekernel<:blasso.dLaplace
    if typekernel<:blasso.dulaplace
      al,be=0.4,0.999;
    end
  else
    MAX_STEP=50;
  end

  i=1;
  tmin,t,tmax=0.0,1.0,10.0;
  M=length(d);
  maxIterHit=false;

  @inbounds for i in 1:M
    if d[i]>0.0
      if !(up[i]==Inf)
        tb=(up[i]-x0[i])/d[i];
        if tmax>tb
          tmax=tb;
        end
      end
    elseif d[i]<0.0
      if !(lw[i]==-Inf)
        tb=(lw[i]-x0[i])/d[i];
        if tmax>tb
          tmax=tb;
        end
      end
    end #if
  end #for

  a=tmin;
  b=tmax;

  if t>tmax
    t=(tmin+tmax)/2;
  end

  stop=false;
  while !stop
    x=x0+t*d;
    fx=f(x);

    if fx>fx0+al*t*dot(gx0,d)
      tmax=t;
      t=(tmin+tmax)/2;
    else
      gx=g(x);
      #if abs(vecdot(gx,d))>be*abs(vecdot(gx0,d))
      if dot(gx,d)<be*dot(gx0,d)
        tmin=t;
        t=(tmin+tmax)/2;
      else
        stop=true;
      end
    end #if
    i+=1;
    if i>MAX_STEP
      stop=true;
      maxIterHit=true;

      if o.show_warning
        println("### Max Iter in lineSearch ! ###", "d=",d,", -gx0=",-gx0)
      end
    end
  end #while

  return t,maxIterHit
end #function

function backtrackingLS(d::Array{Float64,1},x0::Array{Float64,1},fx0::Float64,f::Function,lw::Array{Float64,1},up::Array{Float64,1},o::sfw_options,typekernel::DataType=blasso.kernel)
  alpha=.5;
  t=1.0;
  x=x0+t*d;
  fx=f(x);
  MAX_STEP=15;
  i=1;
  success=false;

  while fx>fx0 && i<MAX_STEP && !checkInBounds(x,lw,up)
    t=alpha*t;
    x=x0+t*d;
    fx=f(x);
  end

  if i==MAX_STEP
    success=true;
  end

  return t,success
end

function findpointLS(d::Array{Float64,1},x0::Array{Float64,1},fx0::Float64,gx0::Array{Float64,1},f::Function,g::Function,lw::Array{Float64,1},up::Array{Float64,1},o::sfw_options,typekernel::DataType=blasso.kernel)
  if typekernel<:blasso.cLaplace || typekernel<:blasso.dLaplace
    al,be=1e-4,.985;
  else
    al,be=.1,.88;
  end

  j=0;
  k=0;
  l=0;

  tmin,tmax=0.0,10.0;
  M=length(d);

  @inbounds for i in 1:M
    if d[i]>0.0
      if !(up[i]==Inf)
        tb=(up[i]-x0[i])/d[i];
        if tmax>tb
          tmax=tb;
        end
      end
    elseif d[i]<0.0
      if !(lw[i]==-Inf)
        tb=(lw[i]-x0[i])/d[i];
        if tmax>tb
          tmax=tb;
        end
      end
    end #if
  end #for

  grid=collect(linspace(tmin,tmax,1000));
  for i in 1:length(grid)
    t=grid[i];
    x=x0+t*d;
    fx=f(x);
    gx=g(x);
    if fx<fx0
      j+=1;
      if dot(gx,d)>be*dot(gx0,d)
        l+=1;
      end
    end
    if dot(gx,d)>be*dot(gx0,d)
      k+=1;
    end
  end

  return j,k,l
end

function bfgs_b!(f::Function,g::Function,h::Function,x0::Array{Float64,1},lw::Array{Float64,1},up::Array{Float64,1},r::sfw_result,o::sfw_options,typekernel::DataType=blasso.kernel)
  fx0,gx0=f(x0),g(x0);
  N=length(x0);

  u_conv,g_conv,f_conv,max_luIter_hit,convergence=false,false,false,false,false;
  luIter,cIterHitLS=0,0;

  # HESSIAN
  # Bx0=h(x0);
  # if !isposdef(Bx0)
  #   Bx0=eye(N);
  # end
  Bx0=eye(N);

  if o.show_trace
    println("Inside BFGS-B : Amplitudes/Positions ---- Energy ---- Gradient ----");
  end

  stop=false;
  while !stop
    if norm(gx0)>1e-15
      #println("\n###### luIter=",luIter," ######")
      xc,Ac=computeCauchyPoint(x0,gx0,Bx0,lw,up);

      xb=subspaceMin(xc,Ac,x0,gx0,Bx0,lw,up);

      chgDDescent=false;
      if vecdot(xb-x0,gx0)>=0.0
        if o.show_warning
          println("### d=xb-x0 is not a descent direction ! ###")
        end
        gx0proj=toolbox.projgradient(gx0,Ac);
        d=copy(-gx0proj);
        chgDDescent=true;
      else
        d=copy(xb-x0);
      end

      t,maxIterHitLS=lineSearch(d,x0,fx0,gx0,f,g,lw,up,o,typekernel);
      #if luIter==2
      #  t=1e-4;
      #else
        if maxIterHitLS
          cIterHitLS+=1;
          if !chgDDescent
            chgDDescent=true;
            gx0proj=toolbox.projgradient(gx0,Ac);
            d=copy(-gx0proj);
            t,maxIterHitLS=lineSearch(d,x0,fx0,gx0,f,g,lw,up,o,typekernel);
            if maxIterHitLS
              if o.show_warning
                println("### Max iter in LS even after d=-gx0 ###")
              end
              t,backtracking_succeed=backtrackingLS(xb-x0,x0,fx0,f,lw,up,o,typekernel);
              if !backtracking_succeed
                t=0.0;
                if o.show_warning
                  println("### backtrackingLS failed ###")
                end
              end
            end
          else
            t=0.0;
            if o.show_warning
              println("### Already change d before LS ###")
            end
          end
        end
      #end


      s=t*d;
      Bs=Bx0*s;
      x=x0+s;

      fx,gx=f(x),g(x);
      if fx>fx0 && o.show_warning
        println("### No decrease of energy ! : df=",fx-fx0)
      end

      dg = gx-gx0;
      c1,c2 = vecdot(dg,s),vecdot(s,Bs);
      Bx = Bx0 + 1/c1 * dg * dg' - 1/c2 * Bs * Bs';

      # STORE
      if o.keep_trace
        update_result!(r;f_list=f(x),
                         normG=norm(gx),
                         f=fx,
                         f_previous=fx0,
                         u_list=x,
                         u=x,
                         u_previous=x0,
                         g=gx,
                         g_list=gx,
                         g_previous=gx0);
      else # only store energy, u and gradient of last two iterations for assessing convergence.
        update_result!(r;f=fx,
                         f_previous=fx0,
                         u=x,
                         u_previous=x0,
                         g=gx,
                         g_previous=gx0);
      end

      # SHOW trace
      if o.show_trace
        println(x, " ## ", r.f_list[end], " ## ",r.g_list[end]);
      end

      # Iterate counter loop.
      luIter+=1;

      # Assess convergence
      u_conv,g_conv,f_conv,max_luIter_hit,stop=assess_convergence(luIter,r,o);

      # If convergence update result accordingly.
      if stop
        if !o.subblock_descent
          update_result!(r;luIter=luIter,
                         u_converged=u_conv,
                         g_converged=g_conv,
                         f_converged=f_conv,
                         max_luIter_hits=max_luIter_hit,
                         f_bfgs=fx,
                         u_bfgs=x,
                         g_bfgs=gx,
                         warning_step=cIterHitLS);
        else
          update_result!(r;subblock_iter=luIter);
          append!(r.f_subblock_iter[end],[fx]);
        end
        if o.show_mainIter && !o.subblock_descent
          println("### After BFGS ###");
          println("Amp/Pos : ",r.u);
          println("Energy : ",r.f);
          #println("Gradient : ",r.g);
        end
      end

      x0,fx0,gx0,Bx0=copy(x),copy(fx),copy(gx),copy(Bx);
    else # norm gradient <1e-15
      fx=copy(fx0);
      gx=copy(gx0);
      x=copy(x0);
      u_conv,g_conv,f_conv,max_luIter_hit,stop=false,true,false,false,true;

      # STORE
      if o.keep_trace
        update_result!(r;f_list=fx,
                         normG=norm(gx),
                         f=fx,
                         f_previous=fx0,
                         u_list=x,
                         u=x,
                         u_previous=x0,
                         g=gx,
                         g_list=gx,
                         g_previous=gx0);
      else # only store energy, u and gradient of last two iterations for assessing convergence.
        update_result!(r;f=fx,
                         f_previous=fx0,
                         u=x,
                         u_previous=x0,
                         g=gx,
                         g_previous=gx0);
      end

      # SHOW trace
      if o.show_trace
        println(x, " ## ", r.f_list[end], " ## ",r.g_list[end]);
      end

      # If convergence, update result accordingly.
      if stop
        if !o.subblock_descent
          update_result!(r;luIter=luIter,
                         u_converged=u_conv,
                         g_converged=g_conv,
                         f_converged=f_conv,
                         max_luIter_hits=max_luIter_hit,
                         f_bfgs=fx,
                         u_bfgs=x,
                         g_bfgs=gx,
                         warning_step=cIterHitLS);
        else
          update_result!(r;subblock_iter=luIter);
          append!(r.f_subblock_iter[end],[fx]);
        end
        if o.show_mainIter && !o.subblock_descent
          println("### After BFGS ###");
          println("Amp/Pos : ",r.u);
          println("Energy : ",r.f);
          #println("Gradient : ",r.g);
        end
      end
    end
  end #while
end

function assess_convergence(iter::Integer,r::sfw_result,o::sfw_options)
  convergence=false;
  u_conv,g_conv,f_conv,max_iter_hit=false,false,false,false;

  if isnan(r.f)
    convergence=true;
  else
    if norm(r.u-r.u_previous)<o.u_tol
      convergence=true;
      u_conv=true;
    end
    if norm(r.g)<o.g_tol
      convergence=true;
      g_conv=true;
    end
    if abs(r.f-r.f_previous)<o.f_tol
      convergence=true;
      f_conv=true;
    end
    if iter>=o.max_luIter
      convergence=true;
      max_iter_hit=true;
    end
  end

  return u_conv,g_conv,f_conv,max_iter_hit,convergence
end

function computeNewAmplitudes(r::sfw_result,op::blasso.operator,lambda::Real,tol::Real,positive::Bool=true,algo::String="FISTA")
  if op.dim == 1
    x=Array{Float64}(1);
  else
    x=Array{Array{Float64,1}}(1);
  end

  newPos=r.newPos[end];
  sameNewPos=false;
  I=1;

  a_previous=Array{Float64}(1);
  if positive
    new_a=[1.0];
  else
    new_a=[0.0];
  end
  N=blasso.lengthMeasure(r.u,d=op.dim);
  if N==1 && length(r.newPos)==1
    if op.dim == 1
      x[1]=r.newPos[1][1];
    else
      x[1]=r.newPos[1];
    end
    a_previous=new_a;
    if algo=="FISTA"
      a=blasso.lassoFISTA(x,a_previous,op,lambda,tol,positive);
    else
      println(algo,": don't know this algo!")
    end
  else
    a,X=blasso.decompAmpPos(r.u,d=op.dim);

    xt=[X[i]-newPos for i in 1:N];
    for i in 1:N
      if norm(xt[i])<1e-6
        I=i;
        sameNewPos=true;
      end
    end
    if sameNewPos
      a=vcat(a,a[I]/2);
      a[I]=a[I]/2;
      if op.dim == 1
        x=vcat(X,newPos);
      else
        x=vcat(X,[newPos]);
      end
    else
      if op.dim == 1
        x=vcat(X,newPos);
      else
        x=vcat(X,[newPos]);
      end
      a_previous=vcat(a,new_a);
      if algo=="FISTA"
        a=blasso.lassoFISTA(x,a_previous,op,lambda,tol,positive);
      else
        println(algo,": don't know this algo!")
      end
    end
  end

  return toolbox.recompAmpPos(a,x,d=op.dim);
end

end # module
