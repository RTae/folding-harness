# Weights notice for the pair-stack oracle fixtures

The 36 files under `fixtures/weights/` hold copies of trained weight tensors
of the RF3 model. They come from Pairformer block 24 of the checkpoint
`rf3_foundry_01_24_latest_remapped.ckpt`, published by the Institute for
Protein Design at the University of Washington, and they are the EMA shadow
weights that foundry's own inference runs. The activation files under
`fixtures/tensors/` are computed from those weights. The checkpoint itself is
not redistributed here, and must not be committed to this repository.

- Source URL:
  <https://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_latest_remapped.ckpt>
- Source SHA-256:
  `364ef592fd8042a9cf4176d045015190f8322f961ccca38d891b20ca578d3bb0`
- Source size: 3,038,876,446 bytes
- Extracted by: `tools/pairstack-oracle/generate_fixtures.py`
- Extracted slice: 36 tensors, one Pairformer block out of 48, holding
  363,520 float32 values (1,454,080 bytes), about 0.05% of the checkpoint

Foundry's `LICENSE.md` is the BSD 3-Clause Licence, and its copyright line
states that the licence covers both the source code and the model weights
referenced for download in the README. The fixtures are a redistribution of a
small derived slice of those weights, so the licence text follows here, as the
BSD 3-Clause Licence requires for a redistribution in binary form. Nothing in
this repository is endorsed by the copyright holder.

---

BSD 3-Clause License

Copyright (c) 2025, Institute for Protein Design, University of Washington.
This copyright and license covers both the source code and model
weights referenced for download in the README file.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
