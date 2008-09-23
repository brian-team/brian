function [avt, avn, runs] = cuba_average(N, repeats, best, use_connections, we)
runs = [];
for i=1:repeats
    if nargin<5
        [t, n] = cuba(N, use_connections);
    else
        [t, n] = cuba(N, use_connections, we);
    end
    runs = [runs ; [t n]];
    i
end
[sruns, i] = sort(runs);
runs = runs(i(:,1),:);
runs = runs(1:best,:);
m = mean(runs,1);
avt = m(1);
avn = m(2);