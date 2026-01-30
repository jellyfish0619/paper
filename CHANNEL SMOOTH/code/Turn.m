function [T]=Turn(V,NumSTS)
S=V;
  T=zeros(NumSTS,NumSTS,242);
        for index=1:242
            A=S(:,:,index)*inv(S(:,:,index)'*S(:,:,index));
            P=diag(1./vecnorm(A));
            T(:,:,index)=A*P;
        end
        T = permute(T(:,:,:),[3 2 1]); % Permute to Nst-by-Nsts-by-Nt
