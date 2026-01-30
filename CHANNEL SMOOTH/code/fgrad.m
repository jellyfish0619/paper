
function fg=fgrad(V,Q)

for n=1:242
    if n==1
        R_n=V(1,:,n)*V(1,:,n+1)';
        fg(:,:,n)=-R_n'*Q(:,:,n+1);
    end  
    if n>1&&n<242
        R_n=V(1,:,n)*V(1,:,n+1)';
        R_n_1=V(1,:,n-1)*V(1,:,n)';
        fg(:,:,n)=-R_n'*Q(:,:,n+1)-Q(:,:,n-1)'*R_n_1';
    end
    if n==242
        R_n_1=V(1,:,n-1)*V(1,:,n)';
        fg(:,:,n)=-Q(:,:,n-1)'*R_n_1';
    end
end
% for n=1:242
%     if n==1
%         R_n=V(:,:,n+1)'*V(:,:,n);
%         fg(:,:,n)=-2*R_n'*Q(:,:,n+1);
%     end  
%     if n>1&&n<242
%         R_n=V(:,:,n+1)'*V(:,:,n);
%         R_n_1=V(:,:,n)'*V(:,:,n-1);
%         fg(:,:,n)=-2*R_n'*Q(:,:,n+1)-2*R_n_1*Q(:,:,n-1);
%     end
%     if n==242
%         R_n_1=V(:,:,n)'*V(:,:,n-1);
%         fg(:,:,n)=-2*R_n_1*Q(:,:,n-1);
%     end
% end


