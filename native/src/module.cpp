#include "nimbleml/autograd.hpp"
#include "nimbleml/kernels.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(nimbleml_native, m) {
  m.doc() = "NimbleML native kernels and autograd helpers (required extension)";
  m.attr("with_cuda") =
#if defined(NIMBLEML_WITH_CUDA)
      true
#else
      false
#endif
      ;

  m.def(
      "gelu_forward",
      [](py::array_t<float> x) {
        auto xbuf = x.request();
        py::array_t<float> out(xbuf.size);
        py::array_t<float> tanh_u(xbuf.size);
        auto obuf = out.request();
        auto tbuf = tanh_u.request();
        nimbleml::gelu_forward_cpu(static_cast<const float*>(xbuf.ptr), static_cast<float*>(obuf.ptr),
                                   static_cast<float*>(tbuf.ptr), static_cast<std::size_t>(xbuf.size));
        return py::make_tuple(out, tanh_u);
      },
      py::arg("x"));

  m.def(
      "gelu_backward",
      [](py::array_t<float> grad, py::array_t<float> x, py::array_t<float> tanh_u) {
        auto gbuf = grad.request();
        auto xbuf = x.request();
        auto tbuf = tanh_u.request();
        py::array_t<float> grad_x(xbuf.size);
        auto out = grad_x.request();
        nimbleml::gelu_backward_cpu(static_cast<const float*>(gbuf.ptr), static_cast<const float*>(xbuf.ptr),
                                    static_cast<const float*>(tbuf.ptr), static_cast<float*>(out.ptr),
                                    static_cast<std::size_t>(xbuf.size));
        return grad_x;
      },
      py::arg("grad"), py::arg("x"), py::arg("tanh_u"));

  m.def(
      "rmsnorm_forward",
      [](py::array_t<float> x, py::array_t<float> gamma, float eps) {
        auto xbuf = x.request();
        auto gbuf = gamma.request();
        if (xbuf.ndim < 1) throw std::runtime_error("x ndim");
        const auto dim = static_cast<std::size_t>(xbuf.shape[xbuf.ndim - 1]);
        const auto rows = static_cast<std::size_t>(xbuf.size / static_cast<py::ssize_t>(dim));
        py::array_t<float> out(xbuf.shape);
        py::array_t<float> ms(static_cast<py::ssize_t>(rows));
        py::array_t<float> rms(static_cast<py::ssize_t>(rows));
        auto obuf = out.request();
        auto mbuf = ms.request();
        auto rbuf = rms.request();
        nimbleml::rmsnorm_forward_cpu(static_cast<const float*>(xbuf.ptr), static_cast<const float*>(gbuf.ptr),
                                      static_cast<float*>(obuf.ptr), static_cast<float*>(mbuf.ptr),
                                      static_cast<float*>(rbuf.ptr), rows, dim, eps);
        return py::make_tuple(out, ms, rms);
      },
      py::arg("x"), py::arg("gamma"), py::arg("eps") = 1e-5f);

  m.def(
      "rmsnorm_backward",
      [](py::array_t<float> grad, py::array_t<float> x, py::array_t<float> gamma, py::array_t<float> ms,
         py::array_t<float> rms, float eps) {
        auto xbuf = x.request();
        const auto dim = static_cast<std::size_t>(xbuf.shape[xbuf.ndim - 1]);
        const auto rows = static_cast<std::size_t>(xbuf.size / static_cast<py::ssize_t>(dim));
        py::array_t<float> grad_x(xbuf.shape);
        py::array_t<float> grad_gamma(static_cast<py::ssize_t>(dim));
        auto gx = grad_x.request();
        auto gg = grad_gamma.request();
        nimbleml::rmsnorm_backward_cpu(static_cast<const float*>(grad.request().ptr),
                                       static_cast<const float*>(xbuf.ptr),
                                       static_cast<const float*>(gamma.request().ptr),
                                       static_cast<const float*>(ms.request().ptr),
                                       static_cast<const float*>(rms.request().ptr),
                                       static_cast<float*>(gx.ptr), static_cast<float*>(gg.ptr), rows, dim,
                                       eps);
        return py::make_tuple(grad_x, grad_gamma);
      },
      py::arg("grad"), py::arg("x"), py::arg("gamma"), py::arg("ms"), py::arg("rms"),
      py::arg("eps") = 1e-5f);

  m.def(
      "embedding_scatter_add",
      [](py::array_t<float> grad_w, py::array_t<std::int64_t> ids, py::array_t<float> grad_out) {
        auto wbuf = grad_w.request();
        auto ibuf = ids.request();
        auto gbuf = grad_out.request();
        const auto dim = static_cast<std::size_t>(gbuf.shape[1]);
        const auto n = static_cast<std::size_t>(ibuf.size);
        const auto vocab = static_cast<std::size_t>(wbuf.shape[0]);
        nimbleml::embedding_scatter_add_cpu(static_cast<float*>(wbuf.ptr),
                                            static_cast<const std::int64_t*>(ibuf.ptr),
                                            static_cast<const float*>(gbuf.ptr), n, dim, vocab);
      },
      py::arg("grad_w"), py::arg("ids"), py::arg("grad_out"));

  m.def(
      "crossentropy_forward",
      [](py::array_t<float> logits, py::array_t<std::int64_t> labels) {
        auto lbuf = logits.request();
        auto ybuf = labels.request();
        const auto rows = static_cast<std::size_t>(lbuf.shape[0]);
        const auto classes = static_cast<std::size_t>(lbuf.shape[1]);
        py::array_t<float> max_vals(static_cast<py::ssize_t>(rows));
        py::array_t<float> sum_exp(static_cast<py::ssize_t>(rows));
        float loss = 0.0f;
        nimbleml::crossentropy_forward_cpu(static_cast<const float*>(lbuf.ptr),
                                           static_cast<const std::int64_t*>(ybuf.ptr),
                                           static_cast<float*>(max_vals.request().ptr),
                                           static_cast<float*>(sum_exp.request().ptr), &loss, rows, classes);
        return py::make_tuple(loss, max_vals, sum_exp);
      },
      py::arg("logits"), py::arg("labels"));

  m.def(
      "crossentropy_backward",
      [](float grad_scale, py::array_t<float> logits, py::array_t<std::int64_t> labels,
         py::array_t<float> max_vals, py::array_t<float> sum_exp) {
        auto lbuf = logits.request();
        const auto rows = static_cast<std::size_t>(lbuf.shape[0]);
        const auto classes = static_cast<std::size_t>(lbuf.shape[1]);
        py::array_t<float> grad({static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(classes)});
        nimbleml::crossentropy_backward_cpu(grad_scale, static_cast<const float*>(lbuf.ptr),
                                            static_cast<const std::int64_t*>(labels.request().ptr),
                                            static_cast<const float*>(max_vals.request().ptr),
                                            static_cast<const float*>(sum_exp.request().ptr),
                                            static_cast<float*>(grad.request().ptr), rows, classes);
        return grad;
      },
      py::arg("grad_scale"), py::arg("logits"), py::arg("labels"), py::arg("max_vals"),
      py::arg("sum_exp"));

  m.def(
      "sdpa_forward",
      [](py::array_t<float> q, py::array_t<float> k, py::array_t<float> v, float scale, bool causal) {
        auto qbuf = q.request();
        const auto bh = static_cast<std::size_t>(qbuf.shape[0]);
        const auto seq = static_cast<std::size_t>(qbuf.shape[1]);
        const auto dk = static_cast<std::size_t>(qbuf.shape[2]);
        py::array_t<float> out({static_cast<py::ssize_t>(bh), static_cast<py::ssize_t>(seq),
                                static_cast<py::ssize_t>(dk)});
        py::array_t<float> probs({static_cast<py::ssize_t>(bh), static_cast<py::ssize_t>(seq),
                                  static_cast<py::ssize_t>(seq)});
        nimbleml::sdpa_forward_cpu(static_cast<const float*>(qbuf.ptr),
                                   static_cast<const float*>(k.request().ptr),
                                   static_cast<const float*>(v.request().ptr),
                                   static_cast<float*>(out.request().ptr),
                                   static_cast<float*>(probs.request().ptr), bh, seq, dk, scale, causal);
        return py::make_tuple(out, probs);
      },
      py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"), py::arg("causal") = true);

  m.def(
      "sdpa_backward",
      [](py::array_t<float> grad_out, py::array_t<float> q, py::array_t<float> k, py::array_t<float> v,
         py::array_t<float> probs, float scale) {
        auto qbuf = q.request();
        const auto bh = static_cast<std::size_t>(qbuf.shape[0]);
        const auto seq = static_cast<std::size_t>(qbuf.shape[1]);
        const auto dk = static_cast<std::size_t>(qbuf.shape[2]);
        py::array_t<float> gq({static_cast<py::ssize_t>(bh), static_cast<py::ssize_t>(seq),
                               static_cast<py::ssize_t>(dk)});
        py::array_t<float> gk({static_cast<py::ssize_t>(bh), static_cast<py::ssize_t>(seq),
                               static_cast<py::ssize_t>(dk)});
        py::array_t<float> gv({static_cast<py::ssize_t>(bh), static_cast<py::ssize_t>(seq),
                               static_cast<py::ssize_t>(dk)});
        nimbleml::sdpa_backward_cpu(static_cast<const float*>(grad_out.request().ptr),
                                    static_cast<const float*>(qbuf.ptr),
                                    static_cast<const float*>(k.request().ptr),
                                    static_cast<const float*>(v.request().ptr),
                                    static_cast<const float*>(probs.request().ptr),
                                    static_cast<float*>(gq.request().ptr),
                                    static_cast<float*>(gk.request().ptr),
                                    static_cast<float*>(gv.request().ptr), bh, seq, dk, scale);
        return py::make_tuple(gq, gk, gv);
      },
      py::arg("grad_out"), py::arg("q"), py::arg("k"), py::arg("v"), py::arg("probs"),
      py::arg("scale"));

  m.def(
      "adamw_step",
      [](py::array_t<float> param, py::array_t<float> grad, py::array_t<float> m, py::array_t<float> v,
         float lr, float beta1, float beta2, float bias1, float bias2, float eps, float weight_decay) {
        auto pbuf = param.request();
        const auto n = static_cast<std::size_t>(pbuf.size);
        py::array_t<float> scratch(static_cast<py::ssize_t>(n));
        nimbleml::adamw_step_cpu(static_cast<float*>(pbuf.ptr), static_cast<float*>(grad.request().ptr),
                                 static_cast<float*>(m.request().ptr), static_cast<float*>(v.request().ptr),
                                 static_cast<float*>(scratch.request().ptr), n, lr, beta1, beta2, bias1,
                                 bias2, eps, weight_decay);
      },
      py::arg("param"), py::arg("grad"), py::arg("m"), py::arg("v"), py::arg("lr"), py::arg("beta1"),
      py::arg("beta2"), py::arg("bias1"), py::arg("bias2"), py::arg("eps"), py::arg("weight_decay"));

  m.def(
      "clip_grad_norm",
      [](std::vector<py::array_t<float>> grads, float max_norm) {
        std::vector<float*> ptrs;
        std::vector<std::size_t> sizes;
        for (auto& g : grads) {
          auto buf = g.request();
          ptrs.push_back(static_cast<float*>(buf.ptr));
          sizes.push_back(static_cast<std::size_t>(buf.size));
        }
        return nimbleml::clip_grad_norm_cpu(ptrs.data(), sizes.data(), ptrs.size(), max_norm);
      },
      py::arg("grads"), py::arg("max_norm"));

  m.def("autograd_clear", []() { nimbleml::engine().clear(); });
  m.def("autograd_size", []() { return nimbleml::engine().size(); });
  m.def(
      "autograd_run_backward",
      [](int root_id, bool retain_graph) { nimbleml::engine().run_backward(root_id, retain_graph); },
      py::arg("root_id"), py::arg("retain_graph") = false);
  m.def(
      "autograd_add_py_node",
      [](const std::vector<int>& parents, bool requires_grad, py::function backward) {
        return nimbleml::engine().add_node(parents, requires_grad, [backward]() {
          py::gil_scoped_acquire gil;
          backward();
        });
      },
      py::arg("parents"), py::arg("requires_grad"), py::arg("backward"));

  m.def("flash_sdpa_available", []() {
#if defined(NIMBLEML_WITH_CUDA)
    return true;
#else
    return false;
#endif
  });

#if defined(NIMBLEML_WITH_CUDA)
  // Device-pointer FA API for CuPy: pass contiguous float32 buffer data_ptrs.
  m.def(
      "flash_sdpa_forward_device",
      [](std::uintptr_t q, std::uintptr_t k, std::uintptr_t v, std::uintptr_t out, std::uintptr_t m,
         std::uintptr_t l, std::size_t bh, std::size_t seq, std::size_t dk, float scale) {
        nimbleml::flash_sdpa_forward_cuda(
            reinterpret_cast<const float*>(q), reinterpret_cast<const float*>(k),
            reinterpret_cast<const float*>(v), reinterpret_cast<float*>(out),
            reinterpret_cast<float*>(m), reinterpret_cast<float*>(l), bh, seq, dk, scale);
      },
      py::arg("q"), py::arg("k"), py::arg("v"), py::arg("out"), py::arg("m"), py::arg("l"),
      py::arg("bh"), py::arg("seq"), py::arg("dk"), py::arg("scale"));
  m.def(
      "flash_sdpa_backward_device",
      [](std::uintptr_t grad_out, std::uintptr_t q, std::uintptr_t k, std::uintptr_t v,
         std::uintptr_t m, std::uintptr_t l, std::uintptr_t grad_q, std::uintptr_t grad_k,
         std::uintptr_t grad_v, std::size_t bh, std::size_t seq, std::size_t dk, float scale) {
        nimbleml::flash_sdpa_backward_cuda(
            reinterpret_cast<const float*>(grad_out), reinterpret_cast<const float*>(q),
            reinterpret_cast<const float*>(k), reinterpret_cast<const float*>(v),
            reinterpret_cast<const float*>(m), reinterpret_cast<const float*>(l),
            reinterpret_cast<float*>(grad_q), reinterpret_cast<float*>(grad_k),
            reinterpret_cast<float*>(grad_v), bh, seq, dk, scale);
      },
      py::arg("grad_out"), py::arg("q"), py::arg("k"), py::arg("v"), py::arg("m"), py::arg("l"),
      py::arg("grad_q"), py::arg("grad_k"), py::arg("grad_v"), py::arg("bh"), py::arg("seq"),
      py::arg("dk"), py::arg("scale"));
  m.def(
      "rmsnorm_forward_device",
      [](std::uintptr_t x, std::uintptr_t gamma, std::uintptr_t out, std::uintptr_t ms,
         std::uintptr_t rms, std::size_t rows, std::size_t dim, float eps) {
        nimbleml::rmsnorm_forward_cuda(
            reinterpret_cast<const float*>(x), reinterpret_cast<const float*>(gamma),
            reinterpret_cast<float*>(out), reinterpret_cast<float*>(ms),
            reinterpret_cast<float*>(rms), rows, dim, eps);
      },
      py::arg("x"), py::arg("gamma"), py::arg("out"), py::arg("ms"), py::arg("rms"),
      py::arg("rows"), py::arg("dim"), py::arg("eps") = 1e-5f);
  m.def(
      "rmsnorm_backward_device",
      [](std::uintptr_t grad, std::uintptr_t x, std::uintptr_t gamma, std::uintptr_t ms,
         std::uintptr_t rms, std::uintptr_t grad_x, std::uintptr_t grad_gamma, std::size_t rows,
         std::size_t dim, float eps) {
        nimbleml::rmsnorm_backward_cuda(
            reinterpret_cast<const float*>(grad), reinterpret_cast<const float*>(x),
            reinterpret_cast<const float*>(gamma), reinterpret_cast<const float*>(ms),
            reinterpret_cast<const float*>(rms), reinterpret_cast<float*>(grad_x),
            reinterpret_cast<float*>(grad_gamma), rows, dim, eps);
      },
      py::arg("grad"), py::arg("x"), py::arg("gamma"), py::arg("ms"), py::arg("rms"),
      py::arg("grad_x"), py::arg("grad_gamma"), py::arg("rows"), py::arg("dim"),
      py::arg("eps") = 1e-5f);
#endif
}
