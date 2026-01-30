function [V,rho_allc, rho_all,Q_whole,infobb]= smooth_manifoldswhole(V,chanEstSound,Q_I)

[subcarrierNum,Tx,STA] = size(chanEstSound);
v_all = zeros(Tx,subcarrierNum);


V_c=V;
        manifold = stiefelcomplexfactory(STA,STA,242);
        % Define the cost function
        problem.M = manifold;
       
        problem.cost =  @(Q)fcost(V,Q);
%         @(Q)fcost(V,Q);
%         problem = manoptAD(problem);
        % Define the gradient (optional, Manopt can approximate it)
        problem.egrad = @(Q)fgrad(V,Q);
        %checkgradient(problem);
        % Define options
        options = struct('maxiter',100);
        options.verbosity=0;  % 0: no output, 1: a little output, 2: more output
        
        % Solve
        [Qopt,xcost, infobb,option]=barzilaiborwein(problem,Q_I,options);  % Pass options to trustregions
        Q=Qopt; 
        Q_whole=Qopt;
        fprintf("优化前：%d 优化后：%d\n",fcost(V,Q_I),fcost(V,Q));
        for i=1:242
            V(:,:,i)=V(:,:,i)*Q(:,:,i);
        end

v_all = zeros(Tx,subcarrierNum);
rho_all = zeros(subcarrierNum-1,1);

for sub_i=1:subcarrierNum
    v_all(:,sub_i)= V_c(:,1,sub_i);
%     v_all(:,sub_i) = v_tmp(:,1);
    if sub_i>1
        rho_all(sub_i-1) = real(v_all(:,sub_i)'*v_all(:,sub_i-1) );
    end
end


v_allc = zeros(Tx,subcarrierNum);
rho_allc = zeros(subcarrierNum-1,1);
for sub_i=1:subcarrierNum
    v_allc(:,sub_i)= V(:,1,sub_i);
%     v_all(:,sub_i) = v_tmp(:,1);
    if sub_i>1
        rho_allc(sub_i-1) = real(v_allc(:,sub_i)'*v_allc(:,sub_i-1) );
    end
end
 %figure;
% plot((1:subcarrierNum-1)',abs(rho_all));
% hold on;
% plot((1:subcarrierNum-1)',abs(rho_allc),'r');
% legend('优化前','优化后');
% hold off;
% % 
