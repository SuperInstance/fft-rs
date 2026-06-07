//! # fft-rs
//!
//! A pure-Rust library for Fast Fourier Transform (FFT) algorithms.
//!
//! Provides Cooley-Tukey radix-2 recursive FFT, iterative FFT, inverse FFT,
//! circular convolution via FFT, and Discrete Cosine Transform (DCT-II).
//!
//! All operations work on `Vec<f64>` in natural order. Callers are responsible
//! for padding inputs to power-of-two lengths where required.

/// Complex number representation for FFT operations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex {
    /// Create a new complex number.
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Complex scalar multiplication.
    pub fn scale(self, s: f64) -> Self {
        Self { re: self.re * s, im: self.im * s }
    }

    /// Complex conjugate.
    pub fn conj(self) -> Self {
        Self { re: self.re, im: -self.im }
    }

    /// Magnitude.
    pub fn mag(self) -> f64 {
        self.re.hypot(self.im)
    }
}

impl std::fmt::Display for Complex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}+{}i", self.re, self.im)
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self { re: self.re + other.re, im: self.im + other.im }
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self { re: self.re - other.re, im: self.im - other.im }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

/// Bit-reversal permutation of length `n` (must be a power of two).
fn bit_reverse_permute(data: &mut [Complex]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    let bits = (n as f64).log2() as usize;
    for i in 0..n {
        let mut rev = 0usize;
        let mut val = i;
        for _ in 0..bits {
            rev = (rev << 1) | (val & 1);
            val >>= 1;
        }
        if i < rev {
            data.swap(i, rev);
        }
    }
}

/// Cooley-Tukey radix-2 recursive FFT.
///
/// Input length must be a power of two. Returns the DFT in natural order.
pub fn fft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    if n == 1 {
        return input.to_vec();
    }
    assert!(n.is_power_of_two(), "Input length must be a power of two");

    let mut even: Vec<Complex> = Vec::with_capacity(n / 2);
    let mut odd: Vec<Complex> = Vec::with_capacity(n / 2);
    for (i, &x) in input.iter().enumerate() {
        if i % 2 == 0 {
            even.push(x);
        } else {
            odd.push(x);
        }
    }

    let even_fft = fft(&even);
    let odd_fft = fft(&odd);

    let mut result = vec![Complex::new(0.0, 0.0); n];
    for k in 0..n / 2 {
        let angle = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
        let twiddle = Complex::new(angle.cos(), angle.sin());
        let t = twiddle * odd_fft[k];
        result[k] = even_fft[k] + t;
        result[k + n / 2] = even_fft[k] - t;
    }
    result
}

/// Iterative (in-place) Cooley-Tukey FFT.
///
/// Input length must be a power of two. Uses bit-reversal permutation
/// followed by butterfly operations.
pub fn fft_iterative(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    assert!(n.is_power_of_two(), "Input length must be a power of two");
    if n == 1 {
        return input.to_vec();
    }

    let mut data = input.to_vec();
    bit_reverse_permute(&mut data);

    let mut size = 2usize;
    while size <= n {
        let half = size / 2;
        let angle_step = -2.0 * std::f64::consts::PI / size as f64;
        for start in (0..n).step_by(size) {
            for k in 0..half {
                let angle = angle_step * k as f64;
                let twiddle = Complex::new(angle.cos(), angle.sin());
                let even = data[start + k];
                let odd = twiddle * data[start + k + half];
                data[start + k] = even + odd;
                data[start + k + half] = even - odd;
            }
        }
        size *= 2;
    }
    data
}

/// Inverse FFT (IFFT) using the conjugate trick.
///
/// Computes `x = IFFT(X)` such that `IFFT(FFT(x)) = x`.
pub fn inverse_fft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    assert!(n.is_power_of_two(), "Input length must be a power of two");

    // Conjugate, FFT, conjugate, scale
    let conj_input: Vec<Complex> = input.iter().map(|c| c.conj()).collect();
    let mut result = fft(&conj_input);
    for c in result.iter_mut() {
        *c = c.conj().scale(1.0 / n as f64);
    }
    result
}

/// Circular convolution via FFT.
///
/// Both inputs must have the same length, which must be a power of two.
/// Returns `a (*) b` where `(*)` denotes circular convolution.
pub fn convolution(a: &[Complex], b: &[Complex]) -> Vec<Complex> {
    assert_eq!(a.len(), b.len(), "Inputs must have the same length");
    let n = a.len();
    assert!(n.is_power_of_two(), "Input length must be a power of two");

    let fa = fft(a);
    let fb = fft(b);
    let fc: Vec<Complex> = fa.iter().zip(fb.iter()).map(|(&x, &y)| x * y).collect();
    inverse_fft(&fc)
}

/// Linear convolution via FFT with zero-padding.
///
/// Pads both inputs to the next power of two >= `a.len() + b.len() - 1`,
/// performs circular convolution, and trims the result.
pub fn linear_convolution(a: &[f64], b: &[f64]) -> Vec<f64> {
    let result_len = a.len() + b.len() - 1;
    let n = result_len.next_power_of_two();

    let mut pa = vec![Complex::new(0.0, 0.0); n];
    let mut pb = vec![Complex::new(0.0, 0.0); n];
    for (i, &v) in a.iter().enumerate() {
        pa[i] = Complex::new(v, 0.0);
    }
    for (i, &v) in b.iter().enumerate() {
        pb[i] = Complex::new(v, 0.0);
    }

    let conv = convolution(&pa, &pb);
    conv[..result_len].iter().map(|c| c.re).collect()
}

/// Discrete Cosine Transform (DCT-II).
///
/// Computes the Type-II DCT of a real-valued input sequence.
/// The DCT-II is defined as:
///
/// `X[k] = sum_{n=0}^{N-1} x[n] * cos(pi/N * (n + 0.5) * k)` for `k = 0..N`
pub fn dct_ii(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    let mut output = vec![0.0; n];
    for (k, out) in output.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (i, &x) in input.iter().enumerate() {
            sum += x * (std::f64::consts::PI * (i as f64 + 0.5) * k as f64 / n as f64).cos();
        }
        *out = sum;
    }
    output
}

/// Inverse DCT-II (using the identity that DCT-III is the inverse of DCT-II up to scaling).
pub fn idct_ii(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    let mut output = vec![0.0; n];
    for (k, out) in output.iter_mut().enumerate() {
        let mut sum = input[0] / 2.0;
        for (i, &val) in input.iter().enumerate().skip(1) {
            sum += val * (std::f64::consts::PI * i as f64 * (k as f64 + 0.5) / n as f64).cos();
        }
        *out = sum;
    }
    output
}

/// Naive DFT for correctness verification.
/// O(n²) implementation for testing purposes.
pub fn naive_dft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    let mut output = Vec::with_capacity(n);
    for k in 0..n {
        let mut sum = Complex::new(0.0, 0.0);
        for (i, &x) in input.iter().enumerate() {
            let angle = -2.0 * std::f64::consts::PI * k as f64 * i as f64 / n as f64;
            let twiddle = Complex::new(angle.cos(), angle.sin());
            sum = sum + twiddle * x;
        }
        output.push(sum);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    fn approx_eq_complex(a: Complex, b: Complex, tol: f64) -> bool {
        (a.re - b.re).abs() < tol && (a.im - b.im).abs() < tol
    }

    fn c(re: f64, im: f64) -> Complex {
        Complex::new(re, im)
    }

    #[test]
    fn test_complex_add() {
        let a = c(1.0, 2.0);
        let b = c(3.0, -1.0);
        assert_eq!(a + b, c(4.0, 1.0));
    }

    #[test]
    fn test_complex_sub() {
        let a = c(5.0, 3.0);
        let b = c(2.0, 1.0);
        assert_eq!(a - b, c(3.0, 2.0));
    }

    #[test]
    fn test_complex_mul() {
        // (1+2i)(3+4i) = 3+4i+6i+8i^2 = -5+10i
        let a = c(1.0, 2.0);
        let b = c(3.0, 4.0);
        assert_eq!(a * b, c(-5.0, 10.0));
    }

    #[test]
    fn test_complex_conj() {
        let a = c(3.0, -4.0);
        assert_eq!(a.conj(), c(3.0, 4.0));
    }

    #[test]
    fn test_complex_scale() {
        let a = c(2.0, 3.0);
        assert_eq!(a.scale(2.0), c(4.0, 6.0));
    }

    #[test]
    fn test_complex_magnitude() {
        let a = c(3.0, 4.0);
        assert!((a.mag() - 5.0).abs() < TOL);
    }

    #[test]
    fn test_fft_single_element() {
        let input = vec![c(5.0, 0.0)];
        let output = fft(&input);
        assert_eq!(output.len(), 1);
        assert!(approx_eq_complex(output[0], c(5.0, 0.0), TOL));
    }

    #[test]
    fn test_fft_two_elements() {
        let input = vec![c(1.0, 0.0), c(2.0, 0.0)];
        let output = fft(&input);
        assert!(approx_eq_complex(output[0], c(3.0, 0.0), TOL));
        assert!(approx_eq_complex(output[1], c(-1.0, 0.0), TOL));
    }

    #[test]
    fn test_fft_four_elements() {
        let input = vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)];
        let output = fft(&input);
        let expected = naive_dft(&input);
        for (a, b) in output.iter().zip(expected.iter()) {
            assert!(approx_eq_complex(*a, *b, 1e-8));
        }
    }

    #[test]
    fn test_fft_eight_elements() {
        let input: Vec<Complex> = (0..8).map(|i| c(i as f64, 0.0)).collect();
        let output = fft(&input);
        let expected = naive_dft(&input);
        for (a, b) in output.iter().zip(expected.iter()) {
            assert!(approx_eq_complex(*a, *b, 1e-8));
        }
    }

    #[test]
    fn test_fft_complex_input() {
        let input = vec![c(1.0, 2.0), c(3.0, -1.0), c(0.0, 4.0), c(-2.0, 0.0)];
        let output = fft(&input);
        let expected = naive_dft(&input);
        for (a, b) in output.iter().zip(expected.iter()) {
            assert!(approx_eq_complex(*a, *b, 1e-8));
        }
    }

    #[test]
    fn test_fft_dc_signal() {
        // Constant signal: FFT should have energy only at DC (bin 0)
        let input = vec![c(5.0, 0.0); 8];
        let output = fft(&input);
        assert!(approx_eq_complex(output[0], c(40.0, 0.0), TOL));
        for k in 1..8 {
            assert!(output[k].mag() < TOL, "Non-DC bin {} has magnitude {}", k, output[k].mag());
        }
    }

    #[test]
    fn test_fft_nyquist() {
        // Alternating signal [1, -1, 1, -1]: energy only at Nyquist (bin n/2)
        let input = vec![c(1.0, 0.0), c(-1.0, 0.0), c(1.0, 0.0), c(-1.0, 0.0)];
        let output = fft(&input);
        assert!(output[0].mag() < TOL);
        assert!(approx_eq_complex(output[2], c(4.0, 0.0), TOL));
    }

    #[test]
    fn test_fft_iterative_matches_recursive() {
        let input: Vec<Complex> = (0..16).map(|i| c(i as f64, (i % 3) as f64)).collect();
        let recursive = fft(&input);
        let iterative = fft_iterative(&input);
        for (a, b) in recursive.iter().zip(iterative.iter()) {
            assert!(approx_eq_complex(*a, *b, 1e-8));
        }
    }

    #[test]
    fn test_fft_iterative_single() {
        let input = vec![c(42.0, 0.0)];
        let output = fft_iterative(&input);
        assert!(approx_eq_complex(output[0], c(42.0, 0.0), TOL));
    }

    #[test]
    fn test_fft_iterative_power_of_2_sizes() {
        for &size in &[2, 4, 8, 16, 32, 64] {
            let input: Vec<Complex> = (0..size).map(|i| c(i as f64, 0.0)).collect();
            let rec = fft(&input);
            let itr = fft_iterative(&input);
            for (a, b) in rec.iter().zip(itr.iter()) {
                assert!(approx_eq_complex(*a, *b, 1e-8), "Mismatch at size {}", size);
            }
        }
    }

    #[test]
    fn test_inverse_fft_roundtrip() {
        let input = vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)];
        let transformed = fft(&input);
        let recovered = inverse_fft(&transformed);
        for (a, b) in input.iter().zip(recovered.iter()) {
            assert!(approx_eq_complex(*a, *b, 1e-8));
        }
    }

    #[test]
    fn test_inverse_fft_roundtrip_complex() {
        let input = vec![c(1.0, 2.0), c(3.0, -1.0), c(0.0, 4.0), c(-2.0, 3.0)];
        let transformed = fft(&input);
        let recovered = inverse_fft(&transformed);
        for (a, b) in input.iter().zip(recovered.iter()) {
            assert!(approx_eq_complex(*a, *b, 1e-8));
        }
    }

    #[test]
    fn test_inverse_fft_larger() {
        let input: Vec<Complex> = (0..32).map(|i| c((i as f64).sin(), (i as f64).cos())).collect();
        let recovered = inverse_fft(&fft(&input));
        for (a, b) in input.iter().zip(recovered.iter()) {
            assert!(approx_eq_complex(*a, *b, 1e-7));
        }
    }

    #[test]
    fn test_convolution_delta() {
        let a = vec![c(3.0, 0.0), c(4.0, 0.0), c(5.0, 0.0), c(6.0, 0.0)];
        let delta = vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)];
        let result = convolution(&a, &delta);
        for (x, y) in a.iter().zip(result.iter()) {
            assert!(approx_eq_complex(*x, *y, 1e-8));
        }
    }

    #[test]
    fn test_convolution_matches_naive() {
        let a = vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)];
        let b = vec![c(5.0, 0.0), c(6.0, 0.0), c(7.0, 0.0), c(8.0, 0.0)];
        let result = convolution(&a, &b);
        let n = a.len();
        for k in 0..n {
            let mut expected = 0.0;
            for i in 0..n {
                expected += a[i].re * b[(k + n - i) % n].re;
            }
            assert!((result[k].re - expected).abs() < 1e-8, "Mismatch at index {}", k);
        }
    }

    #[test]
    fn test_linear_convolution_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let result = linear_convolution(&a, &b);
        assert_eq!(result.len(), 4);
        assert!((result[0] - 4.0).abs() < 1e-8);
        assert!((result[1] - 13.0).abs() < 1e-8);
        assert!((result[2] - 22.0).abs() < 1e-8);
        assert!((result[3] - 15.0).abs() < 1e-8);
    }

    #[test]
    fn test_linear_convolution_single_element() {
        let a = vec![5.0];
        let b = vec![3.0];
        let result = linear_convolution(&a, &b);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 15.0).abs() < 1e-8);
    }

    #[test]
    fn test_dct_ii_constant() {
        let input = vec![5.0; 4];
        let output = dct_ii(&input);
        assert!((output[0] - 20.0).abs() < TOL);
        for k in 1..4 {
            assert!(output[k].abs() < TOL, "DCT bin {} = {}", k, output[k]);
        }
    }

    #[test]
    fn test_dct_ii_known_values() {
        let input = vec![1.0, 0.0];
        let output = dct_ii(&input);
        assert!((output[0] - 1.0).abs() < TOL);
        assert!((output[1] - std::f64::consts::FRAC_1_SQRT_2).abs() < TOL);
    }

    #[test]
    fn test_dct_ii_four_elements() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = dct_ii(&input);
        assert_eq!(output.len(), 4);
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_idct_roundtrip() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let transformed = dct_ii(&input);
        let recovered = idct_ii(&transformed);
        for (a, b) in input.iter().zip(recovered.iter()) {
            let expected = a * input.len() as f64 / 2.0;
            assert!((expected - b).abs() < 1e-8, "DCT roundtrip mismatch: {} vs {}", expected, b);
        }
    }

    #[test]
    fn test_dft_vs_fft_correctness() {
        for &size in &[2, 4, 8, 16] {
            let input: Vec<Complex> = (0..size)
                .map(|i| c((i as f64 * 0.5).sin(), (i as f64 * 0.3).cos()))
                .collect();
            let fft_result = fft(&input);
            let dft_result = naive_dft(&input);
            for (a, b) in fft_result.iter().zip(dft_result.iter()) {
                assert!(approx_eq_complex(*a, *b, 1e-8), "FFT vs DFT mismatch at size {}", size);
            }
        }
    }

    #[test]
    fn test_fft_parseval_theorem() {
        let input: Vec<Complex> = (0..8).map(|i| c(i as f64, 0.0)).collect();
        let output = fft(&input);
        let input_energy: f64 = input.iter().map(|c| c.re * c.re + c.im * c.im).sum();
        let output_energy: f64 = output.iter().map(|c| c.mag() * c.mag()).sum();
        assert!((output_energy - 8.0 * input_energy).abs() < 1e-8,
            "Parseval's theorem violated: {} vs {}", output_energy, 8.0 * input_energy);
    }

    #[test]
    fn test_fft_linearity() {
        let a: Vec<Complex> = (0..4).map(|i| c(i as f64, 0.0)).collect();
        let b: Vec<Complex> = (0..4).map(|i| c(0.0, i as f64)).collect();
        let combined: Vec<Complex> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
        let fft_a = fft(&a);
        let fft_b = fft(&b);
        let fft_combined = fft(&combined);
        for i in 0..4 {
            let expected = fft_a[i] + fft_b[i];
            assert!(approx_eq_complex(fft_combined[i], expected, 1e-8));
        }
    }
}
