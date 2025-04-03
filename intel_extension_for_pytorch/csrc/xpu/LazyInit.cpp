#include <torch/csrc/python_headers.h>
#include <mutex>

#include <core/PreInitHook.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>

namespace xpu {

static bool run_yet = false;

void lazy_init() {
  // Here is thread safety. There are two reasons for this:
  // 1. avoid circular calls.
  // 2. avoid GIL's overhead.
  if (run_yet)
    return;

  pybind11::gil_scoped_acquire g;
  // Protected by the GIL. We don't use call_once because under ASAN it
  // has a buggy implementation that deadlocks if an instance throws an
  // exception.  In any case, call_once isn't necessary, because we
  // have taken a lock.
  if (!run_yet) {
    // We set run_yet true in THPModule_initExtension(), which is invoked by
    // Python API's _lazy_init(), to avoid circular calls.
    auto module = THPObjectPtr(
        PyImport_ImportModule("intel_extension_for_pytorch.xpu.lazy_init"));
    if (!module)
      throw python_error();
    auto res =
        THPObjectPtr(PyObject_CallMethod(module.get(), "_lazy_init", ""));
    if (!res) {
      throw python_error();
      run_yet =
          false; // if python API's execution fails, restore tun_yet to FALSE.
    }
  }
}

// Here, register lazy_init to pre_init_hook. It makes possible calling
// lazy_init in back-end if necessary.
IPEX_REGISTER_PRE_INIT_HOOK(&lazy_init)

void set_run_yet_variable_to_false() {
  run_yet = false;
}

void set_run_yet_variable_to_true() {
  run_yet = true;
}

} // namespace xpu
