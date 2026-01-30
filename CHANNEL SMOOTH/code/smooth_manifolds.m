function [V,rho_allc, rho_all,Q_all]= smooth_manifolds(V,chanEstSound,NumTxAnts)

[subcarrierNum,~,~] = size(chanEstSound);
v_all = zeros(NumTxAnts,subcarrierNum);
rho_all = zeros(subcarrierNum-1,1);
V_c=V;

        for i=1:242
               Q_all(:,:,i)=eye(NumTxAnts,NumTxAnts);
        end


for q =1:10
k=0;
for sub_i=1:subcarrierNum-1
    v_tmp= V(:,:,sub_i);
%   v_all(k,sub_i)=V(1,k,sub_i);
    v_all(:,sub_i) = v_tmp(:,1);
    if sub_i>1
        rho_all(sub_i-1) = real(v_all(:,sub_i)'*v_all(:,sub_i-1) );
        % 创建一个向量
        if (abs(rho_all(sub_i-1))<0.98)
            
        k=k+1;
        manifold = stiefelcomplexfactory(NumTxAnts,NumTxAnts);
        % Define the cost function
        problem.M = manifold;
        R = (V(:,:,sub_i)'*V(:,:,sub_i-1)+V(:,:,sub_i)'*V(:,:,sub_i+1));
        problem.cost = @(Q)-trace(R*Q'+R'*Q);
%         problem = manoptAD(problem);
        % Define the gradient (optional, Manopt can approximate it)
        problem.egrad = @(Q)-2*R;
%          checkgradient(problem);
        % Define options
        options = struct();
        options = struct('maxiter',20);
        options.verbosity=0;  % 0: no output, 1: a little output, 2: more output
        % Solve
       [Qopt,xcost, infobb,option] = steepestdescent(problem, [], options);  % Pass options to trustregions
        Q(:,:,sub_i)=Qopt;   
        V(:,:,sub_i)=V(:,:,sub_i)*Q(:,:,sub_i);
        Q_all(:,:,sub_i)=Q_all(:,:,sub_i)*Q(:,:,sub_i);
        end
    end
end
end

k=0;
for sub_i=1:subcarrierNum-1
    v_tmp= V(:,:,sub_i);
%   v_all(k,sub_i)=V(1,k,sub_i);
    v_all(:,sub_i) = v_tmp(:,1);
    if sub_i>1
        rho_all(sub_i-1) = real(v_all(:,sub_i)'*v_all(:,sub_i-1));
        if (abs(rho_all(sub_i-1))<0.98)
        k=k+1;
        end
    end
end
fprintf('k=%d',k);
% manifold = stiefelcomplexfactory(2,2,242);
% problem.M = manifold;
% problem.cost=@(Q) cost(Q,V);
% options = struct();
% options.verbosity=1;  % 0: no output, 1: a little output, 2: more output
%     
% % Solve
% Qopt = trustregions(problem, [], options);  % Pass options to trustregions


semilogy([infobb.iter], [infobb.gradnorm], '.-');
xlabel('Iteration number');
ylabel('Norm of the gradient of f');



[subcarrierNum,~,~] = size(chanEstSound);
v_all = zeros(NumTxAnts,subcarrierNum);
rho_all = zeros(subcarrierNum-1,1);
for sub_i=1:subcarrierNum
    v_all(:,sub_i)= V_c(:,1,sub_i);
%     v_all(:,sub_i) = v_tmp(:,1);
    if sub_i>1
        rho_all(sub_i-1) = real(v_all(:,sub_i)'*v_all(:,sub_i-1) );
    end
end

[subcarrierNum,~,~] = size(chanEstSound);
v_allc = zeros(NumTxAnts,subcarrierNum);
rho_allc = zeros(subcarrierNum-1,1);
for sub_i=1:subcarrierNum
    v_allc(:,sub_i)= V(:,1,sub_i);
%     v_all(:,sub_i) = v_tmp(:,1);
    if sub_i>1
        rho_allc(sub_i-1) = real(v_allc(:,sub_i)'*v_allc(:,sub_i-1) );
    end
end
% figure;
% plot((1:subcarrierNum-1)',abs(rho_all));
% hold on;
% plot((1:subcarrierNum-1)',abs(rho_allc),'r');
% legend('优化前','优化后');
% hold off;

