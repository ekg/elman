// Copyright 2020 LMNT, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include <torch/extension.h>

void elman_init(py::module&);
void elman_silu_init(py::module&);
void elman_variants_init(py::module&);
void gru_init(py::module&);
void gru_silu_init(py::module&);
void skip_elman_init(py::module&);
void indrnn_init(py::module&);
void lstm_init(py::module&);
void lstm_silu_init(py::module&);
void layer_norm_gru_init(py::module&);
void layer_norm_indrnn_init(py::module&);
void layer_norm_lstm_init(py::module&);
void multihead_elman_init(py::module&);
void init_elman_advanced(py::module&);
void init_multihead_triple_r(py::module&);
void diagonal_mhtr_init(py::module&);
void elman_ladder_init(py::module&);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  elman_init(m);
  elman_silu_init(m);
  elman_variants_init(m);
  gru_init(m);
  gru_silu_init(m);
  skip_elman_init(m);
  indrnn_init(m);
  lstm_init(m);
  lstm_silu_init(m);
  layer_norm_gru_init(m);
  layer_norm_indrnn_init(m);
  layer_norm_lstm_init(m);
  multihead_elman_init(m);
  init_elman_advanced(m);
  init_multihead_triple_r(m);
  diagonal_mhtr_init(m);
  elman_ladder_init(m);
}
