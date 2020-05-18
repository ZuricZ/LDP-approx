# LDP-approx

**The error reads:**

Traceback (most recent call last):
  File "C:\Users\...\Anaconda3\lib\site-packages\tensorflow\python\client\session.py", line 1356, in _do_call
    return fn(*args)
  File "C:\Users\...\Anaconda3\lib\site-packages\tensorflow\python\client\session.py", line 1341, in _run_fn
    options, feed_dict, fetch_list, target_list, run_metadata)
  File "C:\Users\...\Anaconda3\lib\site-packages\tensorflow\python\client\session.py", line 1429, in _call_tf_sessionrun
    run_metadata)
tensorflow.python.framework.errors_impl.InvalidArgumentError: Input matrix is not invertible.
	 [[{{node MatrixSolve}}]]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\...\Anaconda3\lib\contextlib.py", line 99, in __exit__
    self.gen.throw(type, value, traceback)
  File "C:\Users\...\Anaconda3\lib\site-packages\tensorflow\python\framework\ops.py", line 5652, in get_controller
    yield g
  File "C:/Users/.../code/brownian_motion_ldp_x.py", line 195, in <module>
    sess.run(update_gradients, feed_dict={X: x_grid})
  File "C:\Users\...\Anaconda3\lib\site-packages\tensorflow\python\client\session.py", line 950, in run
    run_metadata_ptr)
  File "C:\Users\...\Anaconda3\lib\site-packages\tensorflow\python\client\session.py", line 1173, in _run
    feed_dict_tensor, options, run_metadata)
  File "C:\Users\...\Anaconda3\lib\site-packages\tensorflow\python\client\session.py", line 1350, in _do_run
    run_metadata)
  File "C:\Users\...\Anaconda3\lib\site-packages\tensorflow\python\client\session.py", line 1370, in _do_call
    raise type(e)(node_def, op, message)
tensorflow.python.framework.errors_impl.InvalidArgumentError: Input matrix is not invertible.
	 [[node MatrixSolve (defined at /Users/.../code/brownian_motion_ldp_x.py:109) ]]

Errors may have originated from an input operation.
Input Source operations connected to node MatrixSolve:
 concat_9 (defined at /Users/.../code/brownian_motion_ldp_x.py:107)	
 concat_6 (defined at /Users/.../code/brownian_motion_ldp_x.py:106)

Original stack trace for 'MatrixSolve':
  File "\Program Files\JetBrains\PyCharm Community Edition 2019.1.3\plugins\python-ce\helpers\pydev\pydevd.py", line 2127, in <module>
    main()
  File "\Program Files\JetBrains\PyCharm Community Edition 2019.1.3\plugins\python-ce\helpers\pydev\pydevd.py", line 2118, in main
    globals = debugger.run(setup['file'], None, None, is_module)
  File "\Program Files\JetBrains\PyCharm Community Edition 2019.1.3\plugins\python-ce\helpers\pydev\pydevd.py", line 1427, in run
    return self._exec(is_module, entry_point_fn, module_name, file, globals, locals)
  File "\Program Files\JetBrains\PyCharm Community Edition 2019.1.3\plugins\python-ce\helpers\pydev\pydevd.py", line 1434, in _exec
    pydev_imports.execfile(file, globals, locals)  # execute the script
  File "\Program Files\JetBrains\PyCharm Community Edition 2019.1.3\plugins\python-ce\helpers\pydev\_pydev_imps\_pydev_execfile.py", line 18, in execfile
    exec(compile(contents+"\n", file, 'exec'), glob, loc)
  File "/Users/.../code/brownian_motion_ldp_x.py", line 170, in <module>
    update_gradients = opt().adam_tf(constraints, gradient_dict, C_gradient_dict)
  File "/Users/.../code/brownian_motion_ldp_x.py", line 109, in adam_tf
    out = tf.linalg.solve(matrix, rhs, adjoint=False, name=None)
  File "\Users\...\Anaconda3\lib\site-packages\tensorflow\python\ops\gen_linalg_ops.py", line 1515, in matrix_solve
    "MatrixSolve", matrix=matrix, rhs=rhs, adjoint=adjoint, name=name)
  File "\Users\...\Anaconda3\lib\site-packages\tensorflow\python\framework\op_def_library.py", line 788, in _apply_op_helper
    op_def=op_def)
  File "\Users\...\Anaconda3\lib\site-packages\tensorflow\python\util\deprecation.py", line 507, in new_func
    return func(*args, **kwargs)
  File "\Users\...\Anaconda3\lib\site-packages\tensorflow\python\framework\ops.py", line 3616, in create_op
    op_def=op_def)
  File "\Users\...\Anaconda3\lib\site-packages\tensorflow\python\framework\ops.py", line 2005, in __init__
    self._traceback = tf_stack.extract_stack()
