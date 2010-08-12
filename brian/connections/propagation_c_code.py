################################################################################
################## NO DELAYS ###################################################
################################################################################

################## SPARSE ######################################################

propagate_weave_code_sparse_vars = ['sv', 'rowinds', 'datas', 'spikes', 'nspikes']
propagate_weave_code_sparse = '''
    for(int j=0;j<nspikes;j++)
    {
        PyObject* _rowind = rowinds[j];
        PyArrayObject* _row = convert_to_numpy(_rowind, "row");
        conversion_numpy_check_type(_row, PyArray_LONG, "row");
        conversion_numpy_check_size(_row, 1, "row");
        //blitz::Array<long,1> row = convert_to_blitz<long,1>(_row,"row");
        long* row = (long*)_row->data;
        PyObject* _datasj = datas[j];
        PyArrayObject* _data = convert_to_numpy(_datasj, "data");
        conversion_numpy_check_type(_data, PyArray_DOUBLE, "data");
        conversion_numpy_check_size(_data, 1, "data");
        //blitz::Array<double,1> data = convert_to_blitz<double,1>(_data,"data");
        double* data = (double*)_data->data;
        //int m = row.numElements();
        int m = _row->dimensions[0];
        for(int k=0;k<m;k++)
            //sv(row(k)) += data(k);
            sv[row[k]] += data[k];
        Py_DECREF(_rowind);
        Py_DECREF(_datasj);
    }
    '''
    
propagate_weave_code_sparse_modulation_vars = ['sv', 'sv_pre', 'rowinds', 'datas', 'spikes', 'nspikes']
propagate_weave_code_sparse_modulation = '''
    for(int j=0;j<nspikes;j++)
    {
        PyObject* _rowind = rowinds[j];
        PyArrayObject* _row = convert_to_numpy(_rowind, "row");
        conversion_numpy_check_type(_row, PyArray_LONG, "row");
        conversion_numpy_check_size(_row, 1, "row");
        //blitz::Array<long,1> row = convert_to_blitz<long,1>(_row,"row");
        long* row = (long*)_row->data;
        PyObject* _datasj = datas[j];
        PyArrayObject* _data = convert_to_numpy(_datasj, "data");
        conversion_numpy_check_type(_data, PyArray_DOUBLE, "data");
        conversion_numpy_check_size(_data, 1, "data");
        //blitz::Array<double,1> data = convert_to_blitz<double,1>(_data,"data");
        double* data = (double*)_data->data;
        //int m = row.numElements();
        int m = _row->dimensions[0];
        //double mod = sv_pre(spikes(j));
        double mod = sv_pre[spikes[j]];
        for(int k=0;k<m;k++)
            //sv(row(k)) += data(k)*mod;
            sv[row[k]] += data[k]*mod;
        Py_DECREF(_rowind);
        Py_DECREF(_datasj);
    }
    '''

################## DENSE #######################################################

propagate_weave_code_dense_vars = ['sv', 'spikes', 'nspikes', 'N', 'rows']
propagate_weave_code_dense = '''
    for(int j=0;j<nspikes;j++)
    {
        PyObject* _rowsj = rows[j];
        PyArrayObject* _row = convert_to_numpy(_rowsj, "row");
        conversion_numpy_check_type(_row, PyArray_DOUBLE, "row");
        conversion_numpy_check_size(_row, 1, "row");
        //blitz::Array<double,1> row = convert_to_blitz<double,1>(_row,"row");
        double *row = (double *)_row->data;
        for(int k=0;k<N;k++)
            //sv(k) += row(k);
            sv[k] += row[k];
        Py_DECREF(_rowsj);
    }
    '''

propagate_weave_code_dense_modulation_vars = ['sv', 'sv_pre', 'spikes', 'nspikes', 'N', 'rows']
propagate_weave_code_dense_modulation = '''
    for(int j=0;j<nspikes;j++)
    {
        PyObject* _rowsj = rows[j];
        PyArrayObject* _row = convert_to_numpy(_rowsj, "row");
        conversion_numpy_check_type(_row, PyArray_DOUBLE, "row");
        conversion_numpy_check_size(_row, 1, "row");
        //blitz::Array<double,1> row = convert_to_blitz<double,1>(_row,"row");
        //double mod = sv_pre(spikes(j));
        double *row = (double *)_row->data;
        double mod = sv_pre[spikes[j]];
        for(int k=0;k<N;k++)
            sv[k] += row[k]*mod;
            //sv(k) += row(k)*mod;
        Py_DECREF(_rowsj);
    }
    '''

################################################################################
################## DELAYS ######################################################
################################################################################

################## SPARSE ######################################################

delay_propagate_weave_code_sparse_vars = [
    'rowinds', 'datas', 'spikes', 'nspikes', 'dvecrows', 'dr', 'cdi', 'idt',
    'md']
delay_propagate_weave_code_sparse = """
    for(int j=0;j<nspikes;j++)
    {
        PyObject* _rowind = rowinds[j];
        PyArrayObject* _row = convert_to_numpy(_rowind, "row");
        conversion_numpy_check_type(_row, PyArray_LONG, "row");
        conversion_numpy_check_size(_row, 1, "row");
        blitz::Array<long,1> row = convert_to_blitz<long,1>(_row,"row");
        PyObject* _datasj = datas[j];
        PyArrayObject* _data = convert_to_numpy(_datasj, "data");
        conversion_numpy_check_type(_data, PyArray_DOUBLE, "data");
        conversion_numpy_check_size(_data, 1, "data");
        blitz::Array<double,1> data = convert_to_blitz<double,1>(_data,"data");
        PyObject* _dvecrowsj = dvecrows[j];
        PyArrayObject* _dvecrow = convert_to_numpy(_dvecrowsj, "dvecrow");
        conversion_numpy_check_type(_dvecrow, PyArray_DOUBLE, "dvecrow");
        conversion_numpy_check_size(_dvecrow, 1, "dvecrow");
        blitz::Array<double,1> dvecrow = convert_to_blitz<double,1>(_dvecrow,"dvecrow");
        int m = row.numElements();
        for(int k=0;k<m;k++)
        {
            dr((cdi+(int)(idt*dvecrow(k)))%md, (int)row(k)) += data(k);
        }
        Py_DECREF(_rowind);
        Py_DECREF(_datasj);
        Py_DECREF(_dvecrowsj);
    }
    """
    
delay_propagate_weave_code_sparse_modulation_vars = [
    'sv_pre', 'rowinds', 'datas', 'spikes', 'nspikes', 'dvecrows', 'dr', 'cdi',
    'idt', 'md']
delay_propagate_weave_code_sparse_modulation = """
    for(int j=0;j<nspikes;j++)
    {
        PyObject* _rowind = rowinds[j];
        PyArrayObject* _row = convert_to_numpy(_rowind, "row");
        conversion_numpy_check_type(_row, PyArray_LONG, "row");
        conversion_numpy_check_size(_row, 1, "row");
        blitz::Array<long,1> row = convert_to_blitz<long,1>(_row,"row");
        PyObject* _datasj = datas[j];
        PyArrayObject* _data = convert_to_numpy(_datasj, "data");
        conversion_numpy_check_type(_data, PyArray_DOUBLE, "data");
        conversion_numpy_check_size(_data, 1, "data");
        blitz::Array<double,1> data = convert_to_blitz<double,1>(_data,"data");
        PyObject* _dvecrowsj = dvecrows[j];
        PyArrayObject* _dvecrow = convert_to_numpy(_dvecrowsj, "dvecrow");
        conversion_numpy_check_type(_dvecrow, PyArray_DOUBLE, "dvecrow");
        conversion_numpy_check_size(_dvecrow, 1, "dvecrow");
        blitz::Array<double,1> dvecrow = convert_to_blitz<double,1>(_dvecrow,"dvecrow");
        int m = row.numElements();
        double mod = sv_pre(spikes(j));
        for(int k=0;k<m;k++)
        {
            dr((cdi+(int)(idt*dvecrow(k)))%md, (int)row(k)) += data(k)*mod;
        }
        Py_DECREF(_rowind);
        Py_DECREF(_datasj);
        Py_DECREF(_dvecrowsj);
    }
    """

################## DENSE #######################################################

delay_propagate_weave_code_dense_vars = [
    'spikes', 'nspikes', 'N', 'rows', 'dr', 'cdi', 'idt', 'md', 'dvecrows']
delay_propagate_weave_code_dense = """
    for(int j=0;j<nspikes;j++)
    {
        PyObject* _rowsj = rows[j];
        PyArrayObject* _row = convert_to_numpy(_rowsj, "row");
        conversion_numpy_check_type(_row, PyArray_DOUBLE, "row");
        conversion_numpy_check_size(_row, 1, "row");
        blitz::Array<double,1> row = convert_to_blitz<double,1>(_row,"row");
        PyObject* _dvecrowsj = dvecrows[j];
        PyArrayObject* _dvecrow = convert_to_numpy(_dvecrowsj, "dvecrow");
        conversion_numpy_check_type(_dvecrow, PyArray_DOUBLE, "dvecrow");
        conversion_numpy_check_size(_dvecrow, 1, "dvecrow");
        blitz::Array<double,1> dvecrow = convert_to_blitz<double,1>(_dvecrow,"dvecrow");
        for(int k=0;k<N;k++)
            dr((cdi+(int)(idt*dvecrow(k)))%md, k) += row(k);
        Py_DECREF(_rowsj);
        Py_DECREF(_dvecrowsj);
    }
    """

delay_propagate_weave_code_dense_modulation_vars = [
    'sv_pre', 'spikes', 'nspikes', 'N', 'rows', 'dr', 'cdi', 'idt', 'md',
    'dvecrows']
delay_propagate_weave_code_dense_modulation = """
    for(int j=0;j<nspikes;j++)
    {
        PyObject* _rowsj = rows[j];
        PyArrayObject* _row = convert_to_numpy(_rowsj, "row");
        conversion_numpy_check_type(_row, PyArray_DOUBLE, "row");
        conversion_numpy_check_size(_row, 1, "row");
        blitz::Array<double,1> row = convert_to_blitz<double,1>(_row,"row");
        PyObject* _dvecrowsj = dvecrows[j];
        PyArrayObject* _dvecrow = convert_to_numpy(_dvecrowsj, "dvecrow");
        conversion_numpy_check_type(_dvecrow, PyArray_DOUBLE, "dvecrow");
        conversion_numpy_check_size(_dvecrow, 1, "dvecrow");
        blitz::Array<double,1> dvecrow = convert_to_blitz<double,1>(_dvecrow,"dvecrow");
        double mod = sv_pre(spikes(j));
        for(int k=0;k<N;k++)
            dr((cdi+(int)(idt*dvecrow(k)))%md, k) += row(k)*mod;
        Py_DECREF(_rowsj);
        Py_DECREF(_dvecrowsj);
    }
    """
