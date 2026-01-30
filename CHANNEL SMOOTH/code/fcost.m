function F = fcost(V,Q)
F=0;
for  n=1:241
    R=V(1,:,n)*V(1,:,n+1)';
    f= -trace(R*Q(:,:,n+1)'*Q(:,:,n)+R'*Q(:,:,n)'*Q(:,:,n+1));
    F=F+f;
end
% 
% F=0;
% for n=1:241
%     R=V(1,:,n+1)'*V(1,:,n);
%     f=-trace(Q(:,:,n)*Q(:,:,n+1)'*R+Q(:,:,n+1)*Q(:,:,n)'*R');
%     F=F+f;
% end
