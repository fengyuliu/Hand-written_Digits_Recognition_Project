function U = LDAfun(Dtrain,Ltrain,Nc,Nd)

Nd0 = size(Dtrain,1);
mui = zeros(Nd0,Nc);

Ni = zeros(1,Nc);
for i = 1:Nc
    Ni(i) = sum(Ltrain == (i-1));
end

for i = 1:Nc
    mui(:,i) = sum(Dtrain(:,Ltrain == (i-1)),2)/Ni(i);
end

Sigmai = zeros(Nd0,Nd0,Nc);
for i = 1:Nc
    Sigmai(:,:,i) = (Dtrain(:,Ltrain == (i-1))-mui(:,i)*ones(1,Ni(i)))*(Dtrain(:,Ltrain == (i-1))-mui(:,i)*ones(1,Ni(i))).'/Ni(i);
end

Sigmaw = sum(Sigmai,3)/Nc;
Sigmaw = Sigmaw + eye(Nd0);

mu0 = sum(mui,2)/Nc;

Sigmab = (mui-mu0*ones(1,Nc))*(mui-mu0*ones(1,Nc)).'/Nc;

[U,~] = eigs(Sigmab,Sigmaw,Nd);

end