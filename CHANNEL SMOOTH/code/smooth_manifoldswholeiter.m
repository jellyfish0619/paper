function [infortr,infobb,infocg,infobfgs,V_rtr,V_cg,V_bfgs]= smooth_manifoldswholeiter(V,chanEstSound,Q_I)

[subcarrierNum,Tx,STA] = size(chanEstSound);
V_rtr=V;
V_cg=V;
V_bfgs=V;
for j=1:4
        manifold = stiefelcomplexfactory(STA,STA,242);
        % Define the cost function
        problem.M = manifold;
%     
        problem.cost = @(Q)fcost(V,Q);
%         problem = manoptAD(problem);
        % Define the gradient (optional, Manopt can approximate it)
        problem.egrad = @(Q)fgrad(V,Q);
%         checkgradient(problem);
        % Define options
        options = struct('maxiter',10);
        options.verbosity=0;  % 0: no output, 1: a little output, 2: more output
    
        % Solve
        if j==1
        [Qopt, xcost, infortr, options]=trustregions(problem,Q_I,options);  % Pass options to trustregions
        Q=Qopt; 
        for i=1:242
            V_rtr(:,:,i)=V_rtr(:,:,i)*Q(:,:,i);
        end
        elseif j==2
        [Qopt, xcost, infobb, options]=barzilaiborwein(problem,Q_I,options);  % Pass options to trustregions

        elseif j==3
        [Qopt, xcost, infocg, options]=conjugategradient(problem,Q_I,options);  % Pass options to trustregions
        Q=Qopt; 
        for i=1:242
            V_cg(:,:,i)=V_cg(:,:,i)*Q(:,:,i);
        end

        elseif j==4
        [Qopt, xcost, infobfgs, options]=rlbfgs(problem,Q_I,options);  % Pass options to trustregions
        Q=Qopt; 
        for i=1:242
            V_bfgs(:,:,i)=V_bfgs(:,:,i)*Q(:,:,i);
        end

        end

end



