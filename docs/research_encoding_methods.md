# Literature Review — Data Encoding Methods for Photonic Quantum Reservoir Computing

*Compiled 2026-07-17. Context: NSL-KDD intrusion detection via QRC on Quandela hardware (Perceval).*
*Cross-reference: our implementation in `02_data_encoding.ipynb`.*

---

## 1. The paper closest to our setup — QORC on Ascella

**"Photonic Quantum-Accelerated Machine Learning" (QORC)** — [arXiv:2512.08318](https://arxiv.org/html/2512.08318), Dec 2025, verified on Quandela's **Ascella** QPU.

**Encoding pipeline (near-identical to our notebook 02, scheme A):**
1. PCA on input data → first **M** principal components (M = number of modes)
2. PCA values encoded as **phases on phase shifters** between two circuit blocks
3. Architecture: **random interferometer → phase-encoding layer → second random interferometer** (the sandwich; theirs sometimes reuses the same random interferometer twice)
4. Hardware run: **N=3 photons, M=12 modes**; simulations N=1–5, M=12/20/24 with rule-of-thumb **M > N²**
5. Readout: multi-photon **coincidence distribution** (threshold detectors, no PNR needed) = "quantum fingerprint"
6. Fingerprint **standardized to zero mean / unit variance**, then **concatenated with the original classical features**, fed to a linear classifier

**Key result for us:** QORC showed its strongest improvements on **imbalanced and complex datasets** (artificially imbalanced MNIST, MedMNISTv2). NSL-KDD's 5-class task is heavily imbalanced (U2R=52 samples) — this is direct evidence the approach suits our problem.

**Takeaways to adopt:**
- ✅ We already do: PCA→phase sandwich, standardization before readout
- 🔲 To add: **concatenate quantum fingerprint with classical features** (hybrid readout) — cheap and reportedly effective
- 🔲 To check: their M > N² guideline → for N=2 photons our M=8 satisfies it (8 > 4)

---

## 2. Quandela's own QRP paper — Belenos hardware

**"Quantum Reservoir Processing"** — [arXiv:2605.10471](https://arxiv.org/abs/2605.10471), QUONDENSATE collaboration, on Quandela's **Belenos** QPU ([blog](https://www.quandela.com/resources/blog/photonic-quantum-machine-learning/)).

- Hardware: **24 modes, up to 12 photons**, universal **Bell-Walmsley** interferometer of mode couplers + thermal phase shifters; PNR detectors on 16 modes; photon purity ≈0.98, indistinguishability ≈0.92
- Classical inputs encoded into **a selected subset of the mesh's phase shifters / couplings** (same principle as ours)
- Readout: photon-count probability distribution → **ridge regression (L2)**
- Classical task (spiral classification): in-silico ~98% → raw hardware **79.7%** → with mitigation, above classical baseline

**The gem — hardware-aware training:** during readout training they **perturb the reservoir unitary with Gaussian noise (ε ≈ 0.075**, matched to measured transpilation error). The readout becomes robust to sim-vs-hardware deviation. **This is the single most important trick for our cloud deployment step** — train the readout on an ensemble of noise-perturbed simulated reservoirs before submitting to real hardware.

---

## 3. Photon number-resolving QRC — the encoding-design study

**"Photon Number-Resolving Quantum Reservoir Computing"** — [arXiv:2402.06339](https://arxiv.org/html/2402.06339v4).

- Encoding in the **polarization degree of freedom**: each input port gets `cos θ_m(x)·a†_H + sin θ_m(x)·e^{iφ_m(x)}·a†_V` — data sets Poincaré-sphere angles (θ, φ) per mode
- Compares **three encoding maps**: uniform linear (θ = l·x for all modes), **multi-slope linear (θ_m = l_m·x, different slope per mode)**, and spiral (data wound into both θ and φ)
- Reservoir: fixed random network of variable BS + waveplates + PS; ~4 photons; output dim = C(N+M−1, N)
- Readout: Fock-state expectation values → linear regression
- Analyzes feature quality via **conditioned rank** (# linearly independent features above noise floor) and Fourier spectrum of the feature map

**Takeaways to adopt:**
- 🔲 **Multi-slope encoding**: instead of a single scale π for all 8 phases, give each phase its own scale `φ_i = l_i·π·z_i` (e.g. l_i ∈ {0.5, 1, 1.5, 2}). Different slopes → richer Fourier spectrum of the feature map → more independent features. Trivial to add to our `PhotonicEncoder`.
- 🔲 **Conditioned rank** is a better sanity metric than our TV distance — measures how many of our 36 features are actually independent above (shot-)noise.

---

## 4. GBS reservoir — a different photonic paradigm (continuous-variable)

**"Large-scale QRC using a Gaussian Boson Sampler"** — [arXiv:2505.13695](https://arxiv.org/html/2505.13695v1) / [npj Quantum Information](https://www.nature.com/articles/s41534-026-01251-w).

- Encoding: input vector → **spectral phases of a pump pulse** via SLM (~100 wavelength bins); squeezed light, not single photons
- Readout: **mean photon numbers ⟨n_i⟩ + covariance matrix ⟨n_i n_j⟩ − ⟨n_i⟩⟨n_j⟩**
- Results: 97% on 7-class vowels; 58% on MNIST-10; explicitly **static (no temporal memory)**

**Takeaway:** not our hardware paradigm (Quandela = discrete single photons), but the readout idea transfers: 🔲 alongside the 36 exact probabilities, also derive **per-mode marginals ⟨n_i⟩ (8 features) and pairwise correlations (28 features)** — on real hardware these are far more shot-noise-robust than individual Fock-state probabilities.

---

## 5. Baselines initiative — encoding zoo on Quandela hardware

**"Establishing Baselines for Photonic QML"** — [arXiv:2510.25839](https://arxiv.org/html/2510.25839v1), community initiative incl. a 16-mode/3-photon deployment on Ascella.

Encoding strategies compared across contributed models:
- **Phase encoding with learnable scaling**: S(x) = λx with trainable λ before the phase shifters
- **Pixel-to-angle maps**: −2πx, 2πx, sin(2πx), (0.5−x)·π/2 — nonlinear pre-maps into the phase range
- **Amplitude encoding on qudits** (two-photon states) for structured 2D data
- **Quantum kernel**: 20 modes / 5 photons, PCA-20 → phases; accuracy improved monotonically with photon number
- Notably: **parallel quantum⊕classical integration beat sequential** — matches QORC's "concatenate fingerprint with raw features" finding

**Takeaways:**
- 🔲 Try a **sin/periodic pre-map** variant of the encoding as a cheap ablation
- ✅ Confirms phase encoding into Clements/rectangular MZI meshes as the standard Quandela-native pattern
- 📌 More photons → better accuracy (but exponentially costlier simulation and more hardware loss); N=2→3 is our natural upgrade path

---

## 6. Temporal QRC — for the streaming/anomaly-detection endgame

Our current design is **static** (one connection record → one circuit run). True reservoir *memory* would let the system see attack *sequences* (e.g. a slow scan) instead of relying on NSL-KDD's hand-crafted 2-second window features (`count`, `serror_rate`, …).

- **"Experimental memory control in CV optical QRC"** — [arXiv:2506.07279](https://arxiv.org/html/2506.07279) / [Nature Photonics](https://www.nature.com/articles/s41566-026-01880-9): fading memory via **electro-optic feedback**; memory strength is tunable
- **"Scalable Photonic Platform for Real-Time QRC"** — [PRApplied 20, 014051](https://link.aps.org/doi/10.1103/PhysRevApplied.20.014051): temporal + spectral multiplexing
- **"Time-series forecasting with multiphoton states and integrated photonics"** — [arXiv:2512.02928](https://arxiv.org/pdf/2512.02928)

**Practical route on gate-style Quandela hardware (no physical feedback loop):** *output-feedback QRC* — feed a summary of sample *t*'s output distribution back as extra phases for sample *t+1*. Software-level recurrence, hardware-compatible. Worth a dedicated experiment after the static pipeline works.

---

## 7. QML for intrusion detection specifically (context, mostly qubit-based)

- [Survey: QML for Network Intrusion Detection](https://www.mdpi.com/1999-5903/18/5/234) (Future Internet, 2026) — NSL-KDD is the standard QML-IDS testbed; dominant encoding is **angle encoding** (one feature → one qubit rotation), which hits a qubit-count wall on 41-dim data → everyone reduces dims first (PCA/autoencoder), exactly our approach
- [Towards Quantum-Enhanced ML for NIDS](https://www.dpss.inesc-id.pt/~mpc/pubs/Quantum_NIDS_final.pdf) — QSVM on NSL-KDD, competitive but gated by hardware noise
- [Security intrusion detection using QML](https://pmc.ncbi.nlm.nih.gov/articles/PMC9226281/)
- **Gap we can claim:** essentially all QML-IDS work is qubit-circuit-based (QSVM/VQC on IBM). **Photonic QRC on NSL-KDD appears unpublished** — our project sits in an open niche.

---

## Action list distilled for notebook 03+

| Priority | Change | Source |
|---|---|---|
| ★★★ | Concatenate quantum fingerprint + classical/PCA features before readout | QORC, Baselines |
| ★★★ | Hardware-aware training: noise-perturbed reservoir ensemble (ε≈0.075) before cloud runs | Quandela QRP |
| ★★ | Multi-slope phase scaling (per-phase l_i) | PNR-QRC |
| ★★ | Add marginals ⟨n_i⟩ + pair correlations as shot-noise-robust features | GBS-QRC |
| ★★ | Ridge regression readout (their standard) alongside logistic | Quandela QRP |
| ★ | Conditioned-rank diagnostic of the 36 features | PNR-QRC |
| ★ | sin(2πx)-style periodic pre-map ablation | Baselines |
| ★ | N=2→3 photons upgrade path (M>N² ⇒ need M≥10) | QORC, Baselines |
| ☆ later | Output-feedback temporal QRC for streaming detection | CV-QRC / PRApplied |
