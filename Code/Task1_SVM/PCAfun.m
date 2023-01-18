
function [U,Dmu] = PCAfun(Dtrain,Nd)

Ntrain = size(Dtrain,2);
Dmu = mean(Dtrain,2);
Dtrain = Dtrain-Dmu*ones(1,Ntrain);
Cov = Dtrain*Dtrain.'/Ntrain;
[U,~,~] = eigs(Cov,Nd);

end



