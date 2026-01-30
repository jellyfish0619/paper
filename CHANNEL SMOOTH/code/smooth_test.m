function [V,rho_all,rho_allc,Q]= smooth_test(V,chanEstSound)
[subcarrierNum,Tx,STA] = size(chanEstSound);
v_all = zeros(Tx,subcarrierNum);
rho_all = zeros(subcarrierNum-1,1);
V_c=V;

for z=1:242
               Q(:,:,z)=eye(STA,STA);
end

for sub_i=1:subcarrierNum-1
    v_tmp= V(:,:,sub_i);
    v_all(:,sub_i) = v_tmp(:,1);
    if sub_i>1
        rho_all(sub_i-1) = real(v_all(:,sub_i)'*v_all(:,sub_i-1) );
        if (abs(rho_all(sub_i-1))<0.99)
            R = V(:,:,sub_i)'*V(:,:,sub_i-1)+V(:,:,sub_i)'*V(:,:,sub_i+1);
%             *Q(:,:,sub_i-1)**Q(:,:,sub_i+1)
            [U1,D1,V1] = pagesvd(R,'econ'); %SVD分解
            Q(:,:,sub_i)=U1*V1';
            V(:,:,sub_i)=V(:,:,sub_i)*Q(:,:,sub_i);
        end
    end
end


% % 迭代的，解开注释就是迭代方法
% for j=1:5
% 
% for sub_i=1:subcarrierNum-1
%     v_tmp= V(:,:,sub_i);
%     v_all(:,sub_i) = v_tmp(:,1);
%     if sub_i>1
%         rho_all(sub_i-1) = real(v_all(:,sub_i)'*v_all(:,sub_i-1) );
%         if (abs(rho_all(sub_i-1))<0.98)
%             R = V(:,:,sub_i)'*Q(:,:,sub_i-1)*V(:,:,sub_i-1)+V(:,:,sub_i)'*Q(:,:,sub_i+1)*V(:,:,sub_i+1);
%            
%             [U1,D1,V1] = pagesvd(R,'econ'); %SVD分解
%             Q(:,:,sub_i)=U1*V1';
%             V(:,:,sub_i)=V(:,:,sub_i)*Q(:,:,sub_i);
%         end
%     end
% end
% 
% end

%画图

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

%  figure;
% plot((1:subcarrierNum-1)',abs(rho_all));
% hold on;
% plot((1:subcarrierNum-1)',abs(rho_allc),'r');
% legend('优化前','优化后');
% hold off;
% % 
