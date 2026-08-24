import sys, os, runpy, traceback
_shim = '/home/ecs-user/SMTR/results/marble/official_metric_profile/workspaces_smoke2/6c0020cb5e7a9e778afa24a8/engine_logs/runtime_shim/sitecustomize.py'
sys.path.insert(0, os.path.dirname(_shim))
try:
    with open(_shim) as _f:
        exec(compile(_f.read(), _shim, 'exec'), {'__file__': _shim, '__name__': 'sitecustomize'})
except Exception as _e:
    traceback.print_exc()
sys.argv[0] = '/home/ecs-user/MARBLE/marble/main.py'
runpy.run_path(sys.argv[0], run_name='__main__')
