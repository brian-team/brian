function cuba_do_runs(Nvals,repeats,best,use_connections,vary_we,wevals)
fname = 'data/matlab_cuba_results.py';
if not(use_connections)
    fname = 'data/matlab_cuba_nospiking_results.py';
end
if vary_we
    fname = 'data/matlab_cuba_varywe_results.py';
end
fid=fopen(fname,'w');
fprintf(fid,'matlab_cuba = []\n');
if vary_we
    for we=wevals
        fprintf(fid,'cur_result = []\n');
        [t,n,r] = cuba_average(Nvals,repeats,best, use_connections, we);
        for s=r'
            fprintf(fid,'cur_result.append((%g,%g))\n',s(1),s(2));
        end
        fprintf(fid,'matlab_cuba.append((%d,%g,%g,cur_result))\n',Nvals,t,n);
        we
    end
else
    for N=Nvals
        fprintf(fid,'cur_result = []\n');
        [t,n,r] = cuba_average(N,repeats,best, use_connections);
        for s=r'
            fprintf(fid,'cur_result.append((%g,%g))\n',s(1),s(2));
        end
        fprintf(fid,'matlab_cuba.append((%d,%g,%g,cur_result))\n',N,t,n);
        N
    end
end
fclose(fid);