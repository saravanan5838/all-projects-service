import sys, types, runpy

st = types.ModuleType('streamlit')

class Dummy:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    def __getattr__(self, name):
        return lambda *a, **k: None
    def __call__(self, *a, **k):
        return None

class Box:
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def __getattr__(self, name):
        return getattr(st, name, lambda *a, **k: None)
    def __getitem__(self, idx):
        return self
    def __iter__(self):
        return iter([])


def first_option(*a, **k):
    args = list(a)
    for item in args[1:]:
        if isinstance(item, (list, tuple)) and item:
            return item[0]
        if isinstance(item, dict) and item:
            return next(iter(item.keys()))
    if args and isinstance(args[0], (list, tuple)) and args[0]:
        return args[0][0]
    return None

def numeric_default(*a, **k):
    args = list(a)
    if len(args) >= 3:
        return args[2]
    return 0

st.set_page_config = lambda *a, **k: None
st.title = lambda *a, **k: None
st.caption = lambda *a, **k: None
st.error = lambda *a, **k: None
st.warning = lambda *a, **k: None
st.info = lambda *a, **k: None
st.success = lambda *a, **k: None
st.write = lambda *a, **k: None
st.markdown = lambda *a, **k: None
st.subheader = lambda *a, **k: None
st.radio = lambda *a, **k: first_option(*a, **k)
st.number_input = lambda *a, **k: numeric_default(*a, **k)
st.selectbox = lambda *a, **k: first_option(*a, **k)
st.toggle = lambda *a, **k: False
st.columns = lambda *a, **k: [Box() for _ in range(int(a[0]) if a and isinstance(a[0], int) else 2)]
st.tabs = lambda *a, **k: [Box() for _ in range(4)]
st.dataframe = lambda *a, **k: None
st.progress = lambda *a, **k: None
st.metric = lambda *a, **k: None
st.button = lambda *a, **k: False
st.cache_resource = lambda *a, **k: (lambda f: f)
st.spinner = lambda *a, **k: Dummy()
st.expander = lambda *a, **k: Dummy()
st.container = lambda *a, **k: Box()
st.multiselect = lambda *a, **k: (a[1] if len(a) > 1 and isinstance(a[1], list) else ['High', 'Medium'])
st.table = lambda *a, **k: None
st.header = lambda *a, **k: None
st.json = lambda *a, **k: None
st.code = lambda *a, **k: None
st.plotly_chart = lambda *a, **k: None
st.bar_chart = lambda *a, **k: None

class SessionState(dict):
    def setdefault(self, key, default=None):
        return super().setdefault(key, default)

st.session_state = SessionState()

sys.modules['streamlit'] = st
ns = runpy.run_path('app.py')
M = ns['train']()
print('keys', sorted(M.keys()))
print('test_auc', ns['roc_auc_score'](M['y_test'], M['p2_test']))
print('oof_gaps', M['oof_gaps'])
