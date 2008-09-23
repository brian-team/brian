function W = conmat(N,conProba)

% we do this because some oddity of how MATLAB allocates memory
% for sprand makes this much quicker
if N<=8000
    W=sprand(N,N,conProba)>0;
else
    W1 = conmat(N/2,conProba);
    W2 = conmat(N/2,conProba);
    W3 = conmat(N/2,conProba);
    W4 = conmat(N/2,conProba);
    W = [W1 W2 ; W3 W4];
end
